"""E1: paired bootstrap confidence intervals (10,000 resamples of test rows, percentile 95% CI; bootstrap on CPU).

Every comparison scores two prediction vectors on the SAME resampled test rows and reports (first minus second):
  t3_probe_vs_knn_fused   macro-F1: logistic probe - 5-NN, fused embedding         (529 test queries)
  t3_probe_vs_knn_image   macro-F1: logistic probe - 5-NN, image embedding         (529)
  t3_L4_vs_L1             macro-F1: ladder L4 (= v2 index) - L1 (v1 index, test rows removed)   (529)
  t2_imageonly_vs_L4a     macro-F1: image_only - L4a, seed 42                       (2,237 test rows)
  t2_agreed_acc_L4a_vs_imageonly   accuracy: L4a - image_only on the agreed subset, seed 42  (955 rows)
Predictions are regenerated exactly as in t3_partb.py / t3_ladder.py / a1_agreement.py and asserted to reproduce the stored scores.
(The two T2 checkpoints are scored once with the GPU in the same autocast mode as training, as in A1; the bootstrap itself is numpy on CPU.)
Writes results/ablation/X/e1_<comparison>_seed42.json.
"""
import json

import faiss
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from transformers import DistilBertConfig, DistilBertTokenizerFast

from common import ABL_ART, ABL_EMB, ART, DATA, RES, SPLIT_DIR, T3_LABELS_V1, cc, load_annotations, write_row
from t2_models import Variant

N_BOOT, RNG_SEED = 10_000, 20241025
faiss.omp_set_num_threads(4)


def no_miss():
    raise RuntimeError("cache miss")


def stored(part, name):
    return json.load(open(RES / part / f"{name}_seed42.json"))


def macro_f1(cm, classes):
    f = []
    for c in classes:
        tp = cm[c, c]; fn = cm[c].sum() - tp; fp = cm[:, c].sum() - tp
        f.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0)
    return float(np.mean(f))


def confusion(yt, yp, k):
    return np.bincount(yt * k + yp, minlength=k * k).reshape(k, k)


def boot(name, yt, pa, pb, k, classes, metric, mask=None):
    """Paired percentile bootstrap of metric(pa) - metric(pb) over rows (restricted to `mask` if given)."""
    if mask is not None:
        yt, pa, pb = yt[mask], pa[mask], pb[mask]
    n = len(yt)
    fn = (lambda y, p: macro_f1(confusion(y, p, k), classes)) if metric == "macro_f1" else (lambda y, p: float((y == p).mean()))
    point = fn(yt, pa) - fn(yt, pb)
    rng = np.random.default_rng(RNG_SEED)
    d = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, n, n)
        d[b] = fn(yt[i], pa[i]) - fn(yt[i], pb[i])
    lo, hi = np.percentile(d, [2.5, 97.5])
    row = dict(task=name.split("_")[0].upper(), tier="cpu", comparison=name, metric=metric, n_rows=int(n), n_boot=N_BOOT, rng_seed=RNG_SEED,
               score_a=fn(yt, pa), score_b=fn(yt, pb), diff=point, ci95_low=float(lo), ci95_high=float(hi),
               ci_excludes_zero=bool(lo > 0 or hi < 0), boot_mean=float(d.mean()), frac_boot_diff_positive=float((d > 0).mean()))
    write_row("X", f"e1_{name}", 42, row)
    print(f"{name:34s} n={n:4d} {metric:8s} a={row['score_a']:.4f} b={row['score_b']:.4f} diff {point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  excludes 0: {row['ci_excludes_zero']}", flush=True)
    return row


# ── Task 3: probe vs 5-NN (fused, image), exactly as t3_partb.py ──────────────────────────────────────────
S = {s: cc.load_split("damage", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "test")}
img = {s: cc.cached(ART / "emb" / f"clip_damage_{s}.npy", S[s].image_id.tolist(), no_miss) for s in S}
txt = {s: cc.cached(ART / "emb" / f"sbert_damage_{s}.npy", S[s].image_id.tolist(), no_miss) for s in S}
VEC = {"fused": {s: cc.t3_vectors(img[s], txt[s]) for s in S}, "image": {s: cc.l2(img[s]).astype("float32") for s in S}}
keep = ~S["train"].tweet_id.isin(set(S["test"].tweet_id)).to_numpy()
ylab = S["train"].label.to_numpy()
l3 = {l: i for i, l in enumerate(cc.T3_LABELS)}
yt3 = np.array([l3[l] for l in S["test"].label])
knn, probe = {}, {}
for emb in ("fused", "image"):
    index = faiss.IndexFlatIP(VEC[emb]["train"].shape[1]); index.add(VEC[emb]["train"][keep])
    sims, nn_ = index.search(VEC[emb]["test"], 5)
    knn[emb] = np.array([l3[cc.knn_vote(list(ylab[keep][r]), list(s))[0]] for r, s in zip(nn_, sims)])
    C = stored("B", f"t3_lr_{emb}")["C"]
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(VEC[emb]["train"][keep], ylab[keep])
    probe[emb] = np.array([l3[l] for l in clf.predict(VEC[emb]["test"])])
    for nm, p, ref in (("5-NN", knn[emb], stored("B", f"t3_{emb}_k5_majority")), ("probe", probe[emb], stored("B", f"t3_lr_{emb}"))):
        assert abs(macro_f1(confusion(yt3, p, 3), [0, 1, 2]) - ref["test_macro_f1"]) < 1e-9, f"{emb} {nm} != stored"
rows = [boot("t3_probe_vs_knn_fused", yt3, probe["fused"], knn["fused"], 3, [0, 1, 2], "macro_f1"),
        boot("t3_probe_vs_knn_image", yt3, probe["image"], knn["image"], 3, [0, 1, 2], "macro_f1")]

# ── Task 3 ladder L1 vs L4, exactly as t3_ladder.py ───────────────────────────────────────────────────────
ann = load_annotations()
V = cc.t3_vectors(np.load(ABL_EMB / "clip_annot_hf.npy"), np.load(ABL_EMB / "sbert_annot.npy"))
test_mask = (ann.split == "test").to_numpy(); tr_mask = (ann.split == "train").to_numpy()
sibling = ann.tweet_id.isin(set(ann.tweet_id[test_mask])).to_numpy() & tr_mask
tidx = np.where(test_mask)[0]
l4 = {l: i for i, l in enumerate(T3_LABELS_V1)}
yt_l = np.array([l4[l] for l in ann.label_v1.to_numpy()[tidx]])
lad = {}
for name, m in (("L1", ~test_mask), ("L4", tr_mask & ~sibling)):
    r = np.where(m)[0]
    ix = faiss.IndexFlatIP(V.shape[1]); ix.add(V[r]); sims, nn_ = ix.search(V[tidx], 5)
    labs = ann.label_v1.to_numpy()[r[nn_]]
    lad[name] = np.array([l4[cc.knn_vote(list(a), list(s), T3_LABELS_V1)[0]] for a, s in zip(labs, sims)])
    assert abs(macro_f1(confusion(yt_l, lad[name], 4), [0, 1, 2]) - stored("A", f"t3_{name}")["test_macro_f1"]) < 1e-9
rows.append(boot("t3_L4_vs_L1", yt_l, lad["L4"], lad["L1"], 4, [0, 1, 2], "macro_f1"))

# ── Task 2: image_only vs L4a (seed 42) and the agreed subset, exactly as a1_agreement.py ──────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tr2, te2 = (cc.load_split("humanitarian", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "test"))
raw = pd.read_csv(SPLIT_DIR / "task_humanitarian_text_img_test.tsv", sep="\t", dtype=str)
te2 = te2.merge(raw[["image_id", "label_text"]], on="image_id", how="left")
agreed = (te2["label_text"].map(cc.t2_normalise) == te2["label"]).to_numpy()
assert int(agreed.sum()) == 955
labels = sorted(tr2["label"].unique()); l2i = {l: i for i, l in enumerate(labels)}
yt2 = np.array([l2i[l] for l in te2["label"]])
vec = cc.cached(ART / "emb" / "clip_humanitarian_test.npy", te2.image_id.tolist(), no_miss)
tok = DistilBertTokenizerFast.from_pretrained(ART / "tokenizer"); bcfg = DistilBertConfig.from_pretrained(ART / "distilbert_config")
ids, mask = cc.tokenize(tok, te2.text.tolist(), 128)
pred2 = {}
for nm, cfg, var in (("L4a", "t2_L4a", None), ("image_only", "t2_image_only", "image_only")):
    model = (cc.DualStreamClassifier(len(labels), bert_config=bcfg) if var is None else Variant(var, len(labels), bert_config=bcfg)).to(DEVICE)
    model.load_state_dict(torch.load(ABL_ART / "ckpt" / f"{cfg}_seed42.pt", map_location=DEVICE))
    pred2[nm] = cc.t2_predict_proba(model, vec, ids, mask, DEVICE).argmax(1)
    ref = stored("A" if var is None else "B", cfg)
    assert abs(macro_f1(confusion(yt2, pred2[nm], 8), list(range(8))) - ref["test_macro_f1"]) < 1e-9, f"{nm} != stored"
rows.append(boot("t2_imageonly_vs_L4a", yt2, pred2["image_only"], pred2["L4a"], 8, list(range(8)), "macro_f1"))
rows.append(boot("t2_agreed_acc_L4a_vs_imageonly", yt2, pred2["L4a"], pred2["image_only"], 8, list(range(8)), "accuracy", mask=agreed))
