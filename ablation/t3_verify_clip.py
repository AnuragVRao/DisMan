"""Which CLIP did v1's Task 3 use? Recompute the cell-27 demo's top-5 neighbour scores.

capstone-final.ipynb cell 27 printed `Scores: [1. 0.474 0.454 0.45 0.449]` (labels all "Don't know or can't judge") for a random
index row queried against the 18,082-row index. We rebuild that index with a candidate CLIP and look for index rows whose
rounded top-5 scores equal that vector. A wrong embedding model will not reproduce it by chance.
"""
import sys

import numpy as np
import torch

from common import ABL_EMB, DONT_KNOW, cc, load_annotations

TARGET = np.array([1.0, 0.474, 0.454, 0.45, 0.449])
img_file = sys.argv[1] if len(sys.argv) > 1 else "clip_annot_hf.npy"

ann = load_annotations()
V = cc.t3_vectors(np.load(ABL_EMB / img_file), np.load(ABL_EMB / "sbert_annot.npy"))
print(img_file, V.shape)
X = torch.tensor(V, device="cuda")
labels = ann["label_v1"].to_numpy()
tops, tidx = [], []
for i in range(0, len(X), 1024):
    s, ix = (X[i:i + 1024] @ X.T).topk(5, dim=1)
    tops.append(s.cpu().numpy()); tidx.append(ix.cpu().numpy())
tops, tidx = np.concatenate(tops).astype("float64"), np.concatenate(tidx)

exact = np.where((np.round(tops, 3) == TARGET).all(1))[0]
loose = np.where((np.abs(tops - TARGET) <= 0.0006).all(1))[0]
print(f"rows whose rounded top-5 scores == {TARGET.tolist()}: {len(exact)} (within 0.0006: {len(loose)})")
for i in exact:
    nl = labels[tidx[i]]
    print(f"  query row {i} image_id={ann.image_id[i]} split={ann.split[i]} label={labels[i]!r} "
          f"neighbour labels all DK: {bool((nl == DONT_KNOW).all())}")
    print("   scores", np.round(tops[i], 4))
# how discriminative is the target? count rows matching only the non-self part, for context
rest = (np.round(tops[:, 1:], 3) == TARGET[1:]).all(1).sum()
print(f"rows matching the 4 non-self scores: {rest}")
