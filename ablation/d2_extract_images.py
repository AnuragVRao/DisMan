"""Day 2, step 1: cache frozen image features of the two NEW backbones for the humanitarian splits (the L/14 extraction path, ablation/b_clip_l14_extract.py).

  dinov2   facebook/dinov2-base   (ViT-B/14, 86.6M)  768-d `pooler_output` (CLS token after the final layer norm), raw
  resnet50 microsoft/resnet-50    (23.5M, ImageNet-1k) 2048-d global-average-pooled feature, raw
Each unique image file is embedded once with the model's own HF image processor (deterministic eval preprocessing), fp16 forward. Saved, in v2's humanitarian
row order with v2's ids (asserted equal), to artifacts/ablation/d2_feats/<name>/humanitarian_<split>.npy + .ids.json (gitignored).
Usage: python d2_extract_images.py dinov2|resnet50
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from common import ABL_ART, ART, DATA, SPLIT_DIR, cc

SPECS = {"dinov2": ("facebook/dinov2-base", 768), "resnet50": ("microsoft/resnet-50", 2048)}
BATCH = 128


def main(name):
    model_id, dim = SPECS[name]
    out = ABL_ART / "d2_feats" / name
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for s in ("train", "dev", "test"):
        df = cc.load_split("humanitarian", s, DATA, SPLIT_DIR, verbose=False)[0]
        ids = json.load(open(ART / "emb" / f"clip_humanitarian_{s}.ids.json"))
        assert ids == df.image_id.tolist()
        files[s] = (ids, df.img_path.tolist())
    paths = sorted({p for _, ps in files.values() for p in ps})
    print(f"{name}: {len(paths)} unique files for {sum(len(v[0]) for v in files.values())} rows", flush=True)

    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, dtype=torch.float16).to("cuda").eval()
    print("processor:", {k: v for k, v in proc.to_dict().items() if k in ("size", "crop_size", "do_resize", "do_center_crop", "resample")}, flush=True)

    raw, bad, t0 = {}, 0, time.time()
    with ThreadPoolExecutor(8) as ex, torch.no_grad():
        for i in range(0, len(paths), BATCH):
            chunk = paths[i:i + BATCH]
            loaded = list(ex.map(cc._open_rgb, chunk))
            bad += sum(b for _, b in loaded)
            px = proc(images=[im for im, _ in loaded], return_tensors="pt")["pixel_values"].half().cuda()
            o = model(pixel_values=px)
            f = o.pooler_output.reshape(len(chunk), -1).float().cpu().numpy()
            assert f.shape[1] == dim, f.shape
            for p, v in zip(chunk, f):
                raw[p] = v
            if (i // BATCH) % 20 == 0:
                print(f"  {min(i + BATCH, len(paths))}/{len(paths)} files, {time.time() - t0:.0f} s", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min; unreadable images replaced by grey: {bad}", flush=True)
    for s, (ids, ps) in files.items():
        r = np.stack([raw[p] for p in ps]).astype("float32")
        assert np.isfinite(r).all() and r.shape == (len(ids), dim)
        np.save(out / f"humanitarian_{s}.npy", r)
        (out / f"humanitarian_{s}.ids.json").write_text(json.dumps(ids))
        print(f"  saved {s}: {r.shape}, mean norm {np.linalg.norm(r, axis=1).mean():.2f}", flush=True)
    json.dump(dict(model=model_id, dim=dim, n_unique_files=len(paths), unreadable=bad, minutes=(time.time() - t0) / 60,
                   params_m=sum(p.numel() for p in model.parameters()) / 1e6), open(out / "meta.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1])
