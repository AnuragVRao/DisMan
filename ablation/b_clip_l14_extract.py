"""B: extract CLIP ViT-L/14 image embeddings (HF openai/clip-vit-large-patch14, fp16) for exactly the images in v2's CLIP cache.

Keyed exactly like v2's cache: for every (task, split) in {humanitarian, damage} x {train, dev, test} one array in v2's row order plus the same
`.ids.json`, so every downstream join (by position or by id) is identical to the B/32 pipeline. Each unique FILE is embedded once with the model's
deterministic eval preprocessing (resize 224, centre crop, normalise); the one file that appears under two image_id keys therefore gets identical
vectors under both keys. Saved to artifacts/ablation/clip_l14/ (gitignored):
  clip_<task>_<split>.npy       L2-normalised float32 [N, 768]   (the embeddings used downstream)
  clip_<task>_<split>_raw.npy   un-normalised float32 [N, 768]   (control: B/32's cached features are raw)
  clip_<task>_<split>.ids.json  v2's ids (asserted equal)
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from common import ABL_ART, ART, DATA, SPLIT_DIR, cc

MODEL, BATCH = "openai/clip-vit-large-patch14", 256
OUT = ABL_ART / "clip_l14"
OUT.mkdir(parents=True, exist_ok=True)

files = {}                                          # (task, split) -> (ids, paths) in v2's cache order
for task in ("humanitarian", "damage"):
    for s in ("train", "dev", "test"):
        df = cc.load_split(task, s, DATA, SPLIT_DIR, verbose=False)[0]
        ids = json.load(open(ART / "emb" / f"clip_{task}_{s}.ids.json"))
        assert ids == df.image_id.tolist()
        files[(task, s)] = (ids, df.img_path.tolist())
paths = sorted({p for _, ps in files.values() for p in ps})
print(f"{len(paths)} unique files for {sum(len(v[0]) for v in files.values())} cached rows", flush=True)

from transformers import CLIPModel, CLIPProcessor
proc = CLIPProcessor.from_pretrained(MODEL)
model = CLIPModel.from_pretrained(MODEL, dtype=torch.float16).to("cuda").eval()
assert model.config.vision_config.patch_size == 14 and model.config.projection_dim == 768

raw, bad, t0 = {}, 0, time.time()
with ThreadPoolExecutor(8) as ex, torch.no_grad():
    for i in range(0, len(paths), BATCH):
        chunk = paths[i:i + BATCH]
        loaded = list(ex.map(cc._open_rgb, chunk))
        bad += sum(b for _, b in loaded)
        px = proc(images=[im for im, _ in loaded], return_tensors="pt")["pixel_values"].half().cuda()
        f = model.get_image_features(pixel_values=px)
        f = (f if isinstance(f, torch.Tensor) else f.image_embeds).float().cpu().numpy()
        for p, v in zip(chunk, f):
            raw[p] = v
        if (i // BATCH) % 10 == 0:
            print(f"  {min(i + BATCH, len(paths))}/{len(paths)} files, {time.time() - t0:.0f} s", flush=True)
print(f"done in {(time.time() - t0) / 60:.1f} min; unreadable images replaced by grey: {bad}", flush=True)

for (task, s), (ids, ps) in files.items():
    r = np.stack([raw[p] for p in ps]).astype("float32")
    assert np.isfinite(r).all()
    np.save(OUT / f"clip_{task}_{s}_raw.npy", r)
    np.save(OUT / f"clip_{task}_{s}.npy", cc.l2(r).astype("float32"))
    (OUT / f"clip_{task}_{s}.ids.json").write_text(json.dumps(ids))
    print(f"  saved {task}/{s}: {r.shape}, raw norm mean {np.linalg.norm(r, axis=1).mean():.2f}", flush=True)
