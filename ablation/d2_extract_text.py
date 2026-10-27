"""Day 2, step 2: cache frozen sentence embeddings of the humanitarian tweet text for the two candidate text encoders (same text the fine-tuned models read).

  minilm  sentence-transformers/all-MiniLM-L6-v2  (v2's local copy artifacts/sbert), 384-d, L2-normalised (cc.embed_texts)
  mpnet   sentence-transformers/all-mpnet-base-v2 768-d, L2-normalised (cc.embed_texts)
Saved in v2's row order to artifacts/ablation/d2_feats/<name>/humanitarian_<split>.npy (+ ids). Usage: python d2_extract_text.py minilm|mpnet
"""
import json
import sys
import time

import numpy as np
import torch

from common import ABL_ART, ART, DATA, SPLIT_DIR, cc

SPECS = {"minilm": (ART / "sbert", 384), "mpnet": ("sentence-transformers/all-mpnet-base-v2", 768)}


def main(name):
    src, dim = SPECS[name]
    out = ABL_ART / "d2_feats" / name
    out.mkdir(parents=True, exist_ok=True)
    st = cc.load_sbert(torch.device("cuda"), src)
    t0 = time.time()
    for s in ("train", "dev", "test"):
        df = cc.load_split("humanitarian", s, DATA, SPLIT_DIR, verbose=False)[0]
        ids = json.load(open(ART / "emb" / f"clip_humanitarian_{s}.ids.json"))
        assert ids == df.image_id.tolist()
        e = np.asarray(cc.embed_texts(df["text"].tolist(), st), dtype="float32")
        assert e.shape == (len(df), dim) and np.isfinite(e).all(), e.shape
        np.save(out / f"humanitarian_{s}.npy", e)
        (out / f"humanitarian_{s}.ids.json").write_text(json.dumps(ids))
        print(f"{name} {s}: {e.shape}, mean norm {np.linalg.norm(e, axis=1).mean():.3f}", flush=True)
    json.dump(dict(model=str(src), dim=dim, minutes=(time.time() - t0) / 60, params_m=sum(p.numel() for p in st.parameters()) / 1e6), open(out / "meta.json", "w"))
    print(f"{name} done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
