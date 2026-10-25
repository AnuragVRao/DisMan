"""B: paired bootstraps (10,000 resamples of test rows, percentile 95% CI; first minus second; bootstrap on CPU, L4a scored once on the GPU as in E1).

  t2_imageonlyL14_vs_B32       T2 image_only, L/14 (L2-normalised, as specified) - B/32 (v2's raw cached features), seed 42, macro-F1, 2,237 rows
  t2_imageonlyL14_vs_L4a       T2 image_only L/14 - L4a (the full model on B/32), seed 42, macro-F1
  t3_probeL14_vs_B32_fused     T3 logistic probe on the fused embedding, L/14 - B/32, macro-F1, 529 queries
  t3_knnL14_vs_B32_fused       T3 5-NN majority on the fused embedding, L/14 - B/32, macro-F1
  t2_imageonlyL14raw_vs_B32    control: L/14 with the same raw-input convention as B/32
Writes results/ablation/X/e1_<name>_seed42.json (same row format as the E1 bootstraps).
"""
import json

import numpy as np
import torch
from transformers import DistilBertConfig, DistilBertTokenizerFast

from bootstrap_utils import boot, confusion, macro_f1
from common import ABL_ART, ART, DATA, RES, SPLIT_DIR, cc

X = RES / "X"
load = lambda name: np.array(json.load(open(X / f"{name}_seed42.json"))["pred"])


def no_miss():
    raise RuntimeError("cache miss")


def main():
    tr, te = (cc.load_split("humanitarian", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "test"))
    labels = sorted(tr["label"].unique()); l2i = {l: i for i, l in enumerate(labels)}; ALL = list(range(len(labels)))
    y2 = np.array([l2i[l] for l in te["label"]])
    p_l14, p_b32, p_l14raw = load("b_t2_imageonly_L14_norm"), load("b_t2_imageonly_B32_raw"), load("b_t2_imageonly_L14_raw")

    # L4a seed 42 on the cached B/32 vectors (as in e1_bootstrap.py), asserted to reproduce the stored result
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = DistilBertTokenizerFast.from_pretrained(ART / "tokenizer"); bcfg = DistilBertConfig.from_pretrained(ART / "distilbert_config")
    ids, mask = cc.tokenize(tok, te.text.tolist(), 128)
    vec = cc.cached(ART / "emb" / "clip_humanitarian_test.npy", te.image_id.tolist(), no_miss)
    model = cc.DualStreamClassifier(len(labels), bert_config=bcfg).to(device)
    model.load_state_dict(torch.load(ABL_ART / "ckpt" / "t2_L4a_seed42.pt", map_location=device)); model.eval()
    p_l4a = cc.t2_predict_proba(model, vec, ids, mask, device).argmax(1)
    assert abs(macro_f1(confusion(y2, p_l4a, 8), ALL) - json.load(open(RES / "A" / "t2_L4a_seed42.json"))["test_macro_f1"]) < 1e-9

    boot("t2_imageonlyL14_vs_B32", y2, p_l14, p_b32, 8, ALL, "macro_f1")
    boot("t2_imageonlyL14_vs_L4a", y2, p_l14, p_l4a, 8, ALL, "macro_f1")
    boot("t2_imageonlyL14raw_vs_B32", y2, p_l14raw, p_b32, 8, ALL, "macro_f1")

    te3 = cc.load_split("damage", "test", DATA, SPLIT_DIR, verbose=False)[0]
    l3 = {l: i for i, l in enumerate(cc.T3_LABELS)}
    y3 = np.array([l3[l] for l in te3.label])
    boot("t3_probeL14_vs_B32_fused", y3, load("b_t3_L14_fused_probe"), load("b_t3_B32_fused_probe"), 3, [0, 1, 2], "macro_f1")
    boot("t3_knnL14_vs_B32_fused", y3, load("b_t3_L14_fused_knn5"), load("b_t3_B32_fused_knn5"), 3, [0, 1, 2], "macro_f1")


if __name__ == "__main__":
    main()
