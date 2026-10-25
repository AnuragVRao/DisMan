"""B: Task 2 image-only on CLIP ViT-L/14 embeddings (extension experiment; nothing existing is changed).

Same head and hyperparameters as Part B image_only (the d_lodo_t2.train_fold recipe: 10 epochs, lr 1e-4, 3 warm-up + cosine, AdamW, batch 32, inverse-frequency
class weights, fp16 autocast, v2's shuffle convention), seeds 42/43/44, checkpoint chosen on dev. Three input conditions:
  B32_raw      v2's cached B/32 features (512-d, raw)       -> validation: must reproduce the stored Part B image_only results exactly
  L14_norm     L/14 L2-normalised (768-d)                   -> the experiment (as specified)
  L14_raw      L/14 un-normalised (768-d)                   -> control: same input convention as B/32's raw features
Writes results/ablation/X/b_t2_imageonly_<condition>_seed<k>.json (with the per-row test predictions).
"""
import json

import numpy as np
import torch

from common import ABL_ART, ART, DATA, RES, SPLIT_DIR, cc, write_row
from d_lodo_t2 import predict, score, train_fold

L14 = ABL_ART / "clip_l14"


def no_miss():
    raise RuntimeError("cache miss")


def main():
    S = {s: cc.load_split("humanitarian", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "dev", "test")}
    labels = sorted(S["train"].label.unique()); l2i = {l: i for i, l in enumerate(labels)}; NC = len(labels); ALL = list(range(NC))
    Y = {s: torch.tensor([l2i[l] for l in S[s].label]) for s in S}
    feats = {
        "B32_raw": {s: torch.tensor(cc.cached(ART / "emb" / f"clip_humanitarian_{s}.npy", S[s].image_id.tolist(), no_miss), dtype=torch.float32) for s in S},
        "L14_norm": {s: torch.tensor(cc.cached(L14 / f"clip_humanitarian_{s}.npy", S[s].image_id.tolist(), no_miss), dtype=torch.float32) for s in S},
        "L14_raw": {s: torch.tensor(np.load(L14 / f"clip_humanitarian_{s}_raw.npy"), dtype=torch.float32) for s in S},      # same rows/ids as the normalised file
    }
    for s in S:   # the raw and normalised L/14 files share one ids file, which must equal the humanitarian split's image ids (v2's cache order)
        assert json.load(open(L14 / f"clip_humanitarian_{s}.ids.json")) == S[s].image_id.tolist()
        assert np.load(L14 / f"clip_humanitarian_{s}_raw.npy").shape[0] == len(S[s])
    y = Y["test"].numpy()
    for cond, X in feats.items():
        dim = X["train"].shape[1]
        for seed in (42, 43, 44):
            model, dev_f1, best_ep = train_fold(X["train"], Y["train"], X["dev"], Y["dev"], seed, NC, ALL, img_dim=dim)
            p = predict(model, X["test"])
            m = score(y, p, ALL)
            row = dict(task="T2", tier="gpu-small", condition=cond, input_dim=int(dim), dev_macro_f1=dev_f1, best_epoch=best_ep, n_test=int(len(y)),
                       test_acc=m["acc"], test_macro_f1=m["macro_f1_all8"], test_weighted_f1=m["weighted_f1"], pred=p.tolist())
            write_row("X", f"b_t2_imageonly_{cond}", seed, row)
            print(f"{cond:9s} seed {seed}: dev F1 {dev_f1:.4f} @ep{best_ep} | test acc {m['acc']:.4f} macro-F1 {m['macro_f1_all8']:.4f} weighted-F1 {m['weighted_f1']:.4f}", flush=True)
            if cond == "B32_raw":      # validation: the same recipe must reproduce the stored Part B image_only run
                st = json.load(open(RES / "B" / f"t2_image_only_seed{seed}.json"))
                d = max(abs(m["acc"] - st["test_acc"]), abs(m["macro_f1_all8"] - st["test_macro_f1"]), abs(dev_f1 - st["dev_macro_f1"]))
                print(f"          validation vs stored Part B image_only seed {seed}: max |difference| {d:.2e}", flush=True)
                assert d < 1e-4, "the new script does not reproduce Part B image_only"


if __name__ == "__main__":
    main()
