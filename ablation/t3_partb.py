"""Part B, Task 3 component ablation (CPU only, deterministic). ABLATION_PLAN.md.

Index/queries follow v2: the index is damage TRAIN minus rows sharing a tweet_id with the evaluated split (test: 2,361 rows; dev: train minus
dev siblings), cached v2 embeddings (HF CLIP QuickGELU + SBERT). Grid: embedding {image, text, fused} x k {1,3,5,10,20} x vote
{majority, similarity-weighted, class-balanced}; headline = fused, k=5, majority (reproduces v2 exactly: acc 0.6673, macro-F1 0.4914).

  majority         cc.knn_vote: vote share, ties by summed similarity (v2)
  sim_weighted     each neighbour votes with max(cosine, 0); ties by vote count
  class_balanced   each neighbour votes 1 / (class frequency in the index); ties by summed similarity

Logistic-regression probe (class_weight="balanced", C tuned on dev from {0.01,0.1,1,10,100}) on the same embeddings, trained on the same index rows.
Writes results/ablation/B/t3_<emb>_k<k>_<vote>_seed42.json and t3_lr_<emb>_seed42.json.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CPU only: never compete with the GPU batch
import faiss
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

from common import ART, DATA, SPLIT_DIR, cc, write_row

faiss.omp_set_num_threads(4)
KS, VOTES, EMBS = (1, 3, 5, 10, 20), ("majority", "sim_weighted", "class_balanced"), ("image", "text", "fused")
LABELS = cc.T3_LABELS


def no_miss():
    raise RuntimeError("cache miss")


S = {s: cc.load_split("damage", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "dev", "test")}
img = {s: cc.cached(ART / "emb" / f"clip_damage_{s}.npy", S[s].image_id.tolist(), no_miss) for s in S}
txt = {s: cc.cached(ART / "emb" / f"sbert_damage_{s}.npy", S[s].image_id.tolist(), no_miss) for s in S}
VEC = {"fused": {s: cc.t3_vectors(img[s], txt[s]) for s in S},
       "image": {s: cc.l2(img[s]).astype("float32") for s in S},
       "text": {s: txt[s].astype("float32") for s in S}}

keep = {sp: ~S["train"].tweet_id.isin(set(S[sp].tweet_id)).to_numpy() for sp in ("dev", "test")}
y = {sp: S[sp].label.to_numpy() for sp in ("dev", "test")}
ylab = S["train"].label.to_numpy()
print({sp: int(keep[sp].sum()) for sp in keep}, "index rows (dev / test)", flush=True)


def vote(labs, sims, freq, how):
    if how == "majority":
        return cc.knn_vote(list(labs), list(sims))[0]
    score = {l: 0.0 for l in LABELS}; cnt = {l: 0 for l in LABELS}; tot = {l: 0.0 for l in LABELS}
    for l, s in zip(labs, sims):
        w = max(float(s), 0.0) if how == "sim_weighted" else 1.0 / freq[l]
        score[l] += w; cnt[l] += 1; tot[l] += float(s)
    sec = tot if how == "class_balanced" else cnt
    return max(LABELS, key=lambda l: (score[l], sec[l]))


def knn_eval(emb, split, k_max=max(KS)):
    rows = np.where(keep[split])[0]
    index = faiss.IndexFlatIP(VEC[emb]["train"].shape[1]); index.add(VEC[emb]["train"][rows])
    sims, nn_ = index.search(VEC[emb][split], k_max)
    return sims, ylab[rows][nn_], {l: int((ylab[rows] == l).sum()) for l in LABELS}


res = {}
for emb in EMBS:
    for sp in ("dev", "test"):
        res[(emb, sp)] = knn_eval(emb, sp)
for emb in EMBS:
    for k in KS:
        for how in VOTES:
            m = {}
            for sp in ("dev", "test"):
                sims, labs, freq = res[(emb, sp)]
                pred = np.array([vote(labs[i, :k], sims[i, :k], freq, how) for i in range(len(sims))])
                m[sp] = cc.metrics(y[sp], pred, LABELS)
                if sp == "test":
                    pcf = f1_score(y[sp], pred, labels=LABELS, average=None, zero_division=0)
                    cm = confusion_matrix(y[sp], pred, labels=LABELS)
            headline = (emb, k, how) == ("fused", 5, "majority")
            if headline:
                assert abs(m["test"]["acc"] - 0.6673) < 5e-5 and abs(m["test"]["macro_f1"] - 0.4914) < 5e-5, "headline != v2"
            write_row("B", f"t3_{emb}_k{k}_{how}", 42, dict(
                task="T3", tier="cpu", embedding=emb, k=k, vote=how, headline=headline,
                index_rows_test=int(keep["test"].sum()), index_rows_dev=int(keep["dev"].sum()),
                dev_macro_f1=m["dev"]["macro_f1"], dev_acc=m["dev"]["acc"], test_acc=m["test"]["acc"],
                test_macro_f1=m["test"]["macro_f1"], test_weighted_f1=m["test"]["weighted_f1"],
                per_class_f1=dict(zip(LABELS, map(float, pcf))), confusion_matrix=cm.tolist(), labels=LABELS,
                n_test=int(len(y["test"])), n_dev=int(len(y["dev"]))))
            print(f"{emb:6s} k={k:2d} {how:14s} dev F1 {m['dev']['macro_f1']:.4f} | test acc {m['test']['acc']:.4f} F1 {m['test']['macro_f1']:.4f}", flush=True)

# ── logistic-regression probe: same embeddings, same index rows ──────────────────────────────────────────────
for emb in EMBS:
    X = VEC[emb]
    best = None
    for C in (0.01, 0.1, 1, 10, 100):
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(X["train"][keep["dev"]], ylab[keep["dev"]])
        f1 = cc.metrics(y["dev"], clf.predict(X["dev"]), LABELS)["macro_f1"]
        if best is None or f1 > best[1]:
            best = (C, f1)
    clf = LogisticRegression(C=best[0], class_weight="balanced", max_iter=3000).fit(X["train"][keep["test"]], ylab[keep["test"]])
    pred = clf.predict(X["test"]); m = cc.metrics(y["test"], pred, LABELS)
    pcf = f1_score(y["test"], pred, labels=LABELS, average=None, zero_division=0)
    write_row("B", f"t3_lr_{emb}", 42, dict(
        task="T3", tier="cpu", embedding=emb, probe="logistic_regression", C=best[0], dev_macro_f1=best[1],
        test_acc=m["acc"], test_macro_f1=m["macro_f1"], test_weighted_f1=m["weighted_f1"],
        per_class_f1=dict(zip(LABELS, map(float, pcf))), confusion_matrix=confusion_matrix(y["test"], pred, labels=LABELS).tolist(),
        labels=LABELS, n_test=int(len(y["test"])), index_rows_test=int(keep["test"].sum())))
    print(f"LR probe {emb:6s} C={best[0]:<6} dev F1 {best[1]:.4f} | test acc {m['acc']:.4f} F1 {m['macro_f1']:.4f}", flush=True)
