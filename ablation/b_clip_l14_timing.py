"""B step 1: time CLIP ViT-L/14 (HF openai/clip-vit-large-patch14, fp16) on one 256-image batch and project the total for all images used by T2 and T3.

Images = exactly the images in v2's CLIP cache (artifacts/emb/clip_{humanitarian,damage}_{train,dev,test}.ids.json), embedded once per unique image id
(damage images are a subset of the humanitarian ones except one). A random sample of 256 of those images (seed 0) is loaded and preprocessed with the
model's deterministic eval processor (resize 224, centre crop, normalise), then pushed through get_image_features in fp16. Nothing is saved except the timing JSON.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from common import ART, DATA, SPLIT_DIR, cc, write_row

MODEL = "openai/clip-vit-large-patch14"
BATCH = 256

# unique images, in a stable order, with the cache-id check
imgs = {}
for task in ("humanitarian", "damage"):
    for s in ("train", "dev", "test"):
        df = cc.load_split(task, s, DATA, SPLIT_DIR, verbose=False)[0]
        assert json.load(open(ART / "emb" / f"clip_{task}_{s}.ids.json")) == df.image_id.tolist()
        imgs.update(dict(zip(df.image_id, df.img_path)))
ids = list(imgs)
n = len(ids)
print(f"unique images in v2's CLIP cache: {n}", flush=True)

from transformers import CLIPModel, CLIPProcessor
t0 = time.time()
proc = CLIPProcessor.from_pretrained(MODEL)
model = CLIPModel.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()
t_load = time.time() - t0
print(f"model download + load: {t_load:.1f} s | image embedding dim {model.config.projection_dim} | patch {model.config.vision_config.patch_size}, "
      f"image size {model.config.vision_config.image_size} | params {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M | VRAM after load {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)

rng = np.random.default_rng(0)
sample = [imgs[i] for i in rng.choice(ids, BATCH, replace=False)]

t0 = time.time()
with ThreadPoolExecutor(8) as ex:
    loaded = list(ex.map(lambda p: cc._open_rgb(p)[0], sample))
t_open = time.time() - t0
t0 = time.time()
px = proc(images=loaded, return_tensors="pt")["pixel_values"]
t_proc = time.time() - t0
assert px.shape == (BATCH, 3, 224, 224)
print(f"256 images: open+convert {t_open:.2f} s (8 threads) | processor (resize/crop/normalise, single thread) {t_proc:.2f} s", flush=True)

x = px.half().cuda()
with torch.no_grad():
    for _ in range(2):                                         # warm-up
        model.get_image_features(pixel_values=x)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(3):
        t0 = time.time(); f = model.get_image_features(pixel_values=x); torch.cuda.synchronize(); ts.append(time.time() - t0)
t_gpu = float(np.median(ts))
peak = torch.cuda.max_memory_allocated() / 2**30
assert f.shape == (BATCH, 768) and torch.isfinite(f).all()
print(f"GPU forward, 256-image fp16 batch: {np.round(ts, 3).tolist()} s (median {t_gpu:.3f}) = {BATCH / t_gpu:.0f} img/s | peak VRAM {peak:.2f} GiB", flush=True)

batches = int(np.ceil(n / BATCH))
seq = batches * (t_open + t_proc + t_gpu)                     # load, preprocess, forward one after another (what cc.embed_images does)
ovl = batches * max(t_open + t_proc, t_gpu)                   # loading/preprocessing overlapped with the GPU forward (lower bound)
print(f"projected for {n} images = {batches} batches of {BATCH}: sequential {seq / 60:.1f} min | overlapped lower bound {ovl / 60:.1f} min "
      f"| GPU-only {batches * t_gpu / 60:.1f} min | limit 40 min -> {'OVER, stop' if seq / 60 > 40 else 'within'}", flush=True)
write_row("X", "b_clip_l14_timing", 42, dict(task="T2/T3", tier="gpu-small", model=MODEL, dtype="fp16", n_unique_images=n, batch=BATCH, batches=batches,
                                              model_load_s=t_load, open_s=t_open, processor_s=t_proc, gpu_forward_s_runs=ts, gpu_forward_s_median=t_gpu,
                                              peak_vram_gib=peak, projected_sequential_min=seq / 60, projected_overlapped_min=ovl / 60,
                                              projected_gpu_only_min=batches * t_gpu / 60, limit_min=40, within_limit=bool(seq / 60 <= 40)))
