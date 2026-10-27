"""G: T3 5-NN on the fused embeddings, class-balanced vs majority voting: paired bootstrap on the test rows (CPU).

Re-runs exactly the t3_partb.py index/query/vote code for embedding=fused, k=5 (test: damage-train index minus rows sharing a tweet_id with the test
split), asserts the macro-F1 of both votes equals the stored Part B results, then bootstraps (class_balanced - majority) macro-F1 over the 529 test rows
(10,000 paired resamples, fixed 3-class label set). Deterministic, single run. Writes results/ablation/G/g_t3_knn_vote_boot_seed42.json.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import json

import faiss
import numpy as np

from bootstrap_utils import macro_f1_vec, paired_diff_ci, resample_idx
from common import ART, DATA, RES, SPLIT_DIR, cc, write_row
from extra_metrics import macro_f1_ci
from g_analyze_imbalance import verdict

faiss.omp_set_num_threads(4)
LABELS = cc.T3_LABELS; K = 3; KNN = 5


def no_miss():
    raise RuntimeError("cache miss")


def vote(labs, sims, freq, how):          # identical to t3_partb.vote
    if how == "majority":
        return cc.knn_vote(list(labs), list(sims))[0]
    score = {l: 0.0 for l in LABELS}; tot = {l: 0.0 for l in LABELS}
    for l, s in zip(labs, sims):
        score[l] += 1.0 / freq[l]; tot[l] += float(s)
    return max(LABELS, key=lambda l: (score[l], tot[l]))


def main():
    S = {s: cc.load_split("damage", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "dev", "test")}
    img = {s: cc.cached(ART / "emb" / f"clip_damage_{s}.npy", S[s].image_id.tolist(), no_miss) for s in S}
    txt = {s: cc.cached(ART / "emb" / f"sbert_damage_{s}.npy", S[s].image_id.tolist(), no_miss) for s in S}
    V = {s: cc.t3_vectors(img[s], txt[s]) for s in S}
    keep = ~S["train"].tweet_id.isin(set(S["test"].tweet_id)).to_numpy(); rows = np.where(keep)[0]
    ylab = S["train"].label.to_numpy(); yt = S["test"].label.to_numpy()
    index = faiss.IndexFlatIP(V["train"].shape[1]); index.add(V["train"][rows])
    sims, nn_ = index.search(V["test"], KNN); labs = ylab[rows][nn_]
    freq = {l: int((ylab[rows] == l).sum()) for l in LABELS}
    l2i = {l: i for i, l in enumerate(LABELS)}; y = np.array([l2i[l] for l in yt])
    pred = {h: np.array([l2i[vote(labs[i], sims[i], freq, h)] for i in range(len(sims))]) for h in ("majority", "class_balanced")}
    cmp_ = {}
    for h in pred:
        stored = json.load(open(RES / "B" / f"t3_fused_k5_{h}_seed42.json"))["test_macro_f1"]
        mine = float(cc.metrics(yt, np.array([LABELS[i] for i in pred[h]]), LABELS)["macro_f1"])
        assert abs(mine - stored) < 1e-9, f"{h}: {mine} != stored {stored}"
        cmp_[h] = dict(macro_f1=mine, ci=macro_f1_ci(y, pred[h], K))
        print(f"{h:15s} test macro-F1 {mine:.4f}  (matches stored)  test-sampling 95% CI [{cmp_[h]['ci']['ci95_low']:.3f}, {cmp_[h]['ci']['ci95_high']:.3f}]")
    idx = resample_idx(len(y), 10_000)
    v = {h: {0: macro_f1_vec(y, pred[h], K, idx)} for h in pred}
    d = paired_diff_ci(v["class_balanced"], v["majority"], point_diff=cmp_["class_balanced"]["macro_f1"] - cmp_["majority"]["macro_f1"]); d["verdict"] = verdict(d)
    print(f"class_balanced - majority: {d['diff']:+.4f} [{d['ci95_low']:+.4f}, {d['ci95_high']:+.4f}] -> {d['verdict']}  (n=529, 10,000 paired resamples; reporting only)")
    print(f"predictions differing between the two votes: {int((pred['majority'] != pred['class_balanced']).sum())} of {len(y)}")
    write_row("G", "g_t3_knn_vote_boot", 42, dict(task="T3", tier="cpu", n_test=int(len(y)), votes=cmp_, class_balanced_minus_majority=d,
                                                 n_predictions_differing=int((pred["majority"] != pred["class_balanced"]).sum())))


if __name__ == "__main__":
    main()
