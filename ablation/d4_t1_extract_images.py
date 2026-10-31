"""Day 4: frozen image features for the TASK 1 (informative) rows. The earlier caches cover the humanitarian rows only (dev overlap with T1 is 344/2237), so T1 needs its own.
Same recipe as d2_extract_images.py / b_clip_l14_extract.py: model's own HF processor, fp16 forward, each unique file embedded once.
  resnet50  microsoft/resnet-50           2048-d pooled          dinov2  facebook/dinov2-base  768-d pooler_output
  clip_l14  openai/clip-vit-large-patch14 768-d image_embeds, L2-normalised
Saved in d4_t1_data row order to artifacts/ablation/d4_feats/<name>/informative_<split>.npy (+ .ids.json). Usage: python d4_t1_extract_images.py resnet50 dinov2 clip_l14"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import d4_t1_data as d4
from common import ABL_ART, cc

SPECS = {"dinov2": ("facebook/dinov2-base", 768), "resnet50": ("microsoft/resnet-50", 2048), "clip_l14": ("openai/clip-vit-large-patch14", 768)}
BATCH = 128


def main(name):
    model_id, dim = SPECS[name]
    out = ABL_ART / "d4_feats" / name; out.mkdir(parents=True, exist_ok=True)
    D = d4.data(); files = {s: (D["ids"][s], D["S"][s].img_path.tolist()) for s in D["S"]}
    paths = sorted({p for _, ps in files.values() for p in ps})
    print(f"{name}: {len(paths)} unique files for {sum(len(v[0]) for v in files.values())} rows", flush=True)
    from transformers import AutoImageProcessor, AutoModel, CLIPModel
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = (CLIPModel if name == "clip_l14" else AutoModel).from_pretrained(model_id, dtype=torch.float16).to("cuda").eval()
    raw, bad, t0 = {}, 0, time.time()
    with ThreadPoolExecutor(8) as ex, torch.no_grad():
        for i in range(0, len(paths), BATCH):
            chunk = paths[i:i + BATCH]
            loaded = list(ex.map(cc._open_rgb, chunk)); bad += sum(b for _, b in loaded)
            px = proc(images=[im for im, _ in loaded], return_tensors="pt")["pixel_values"].half().cuda()
            if name == "clip_l14":
                f = model.get_image_features(pixel_values=px).float(); f = f / f.norm(dim=-1, keepdim=True)
            else:
                f = model(pixel_values=px).pooler_output.reshape(len(chunk), -1).float()
            f = f.cpu().numpy(); assert f.shape[1] == dim, f.shape
            for p, v in zip(chunk, f):
                raw[p] = v
            if (i // BATCH) % 20 == 0:
                print(f"  {min(i + BATCH, len(paths))}/{len(paths)} files, {time.time() - t0:.0f} s", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min; unreadable images replaced by grey: {bad}", flush=True)
    for s, (ids, ps) in files.items():
        r = np.stack([raw[p] for p in ps]).astype("float32")
        assert np.isfinite(r).all() and r.shape == (len(ids), dim)
        np.save(out / f"informative_{s}.npy", r); (out / f"informative_{s}.ids.json").write_text(json.dumps(ids))
        print(f"  saved {s}: {r.shape}", flush=True)
    json.dump(dict(model=model_id, dim=dim, unreadable=bad, minutes=(time.time() - t0) / 60), open(out / "meta.json", "w"))
    del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    for n in sys.argv[1:]:
        main(n)
