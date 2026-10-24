"""C2 leave-one-disaster-out for Task 1 and Task 3 (CPU only). ABLATION_PLAN.md, Part C.

For each of the 7 events: train / build the index on the OTHER 6 events' TRAIN split, test on the held-out event's TEST split.
Asserted for every fold: the held-out event contributes zero rows to the training data / index (by event_name, tweet_id and image_id).
T1: TF-IDF + logistic regression exactly as v2. T3: v2's fused vectors, exact inner-product index, k=5, v2's vote.
Per-event macro-F1 averages over the classes present in that event's labels or predictions (v2's per-event convention for T2); the
fixed-class-set macro-F1 is stored as well. `in_dist_*` is the standard model (trained on all 7 events) on the same held-out rows.
T2 is out of scope (future work).
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from common import ART, DATA, SPLIT_DIR, cc, write_row

faiss.omp_set_num_threads(4)


def no_miss():
    raise RuntimeError("cache miss")


def score(y_true, y_pred, fixed_labels):
    present = sorted(set(y_true) | set(y_pred))
    m = cc.metrics(y_true, y_pred, present)
    return dict(acc=m["acc"], macro_f1=m["macro_f1"], weighted_f1=m["weighted_f1"],
                macro_f1_fixed_labels=cc.metrics(y_true, y_pred, fixed_labels)["macro_f1"], n=int(len(y_true)))


def t1_model():
    return Pipeline([("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
                     ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=cc.SEED))])


def run_t1():
    tr, te = (cc.load_split("informative", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "test"))
    events = sorted(te.event_name.unique()); assert len(events) == 7, events
    full = t1_model().fit(tr.text, tr.label)
    rows = []
    for e in events:
        fit = tr[tr.event_name != e]
        assert (fit.event_name == e).sum() == 0 and not set(fit.tweet_id) & set(tr[tr.event_name == e].tweet_id), "LEAK"
        held = te[te.event_name == e]
        m = score(held.label.to_numpy(), t1_model().fit(fit.text, fit.label).predict(held.text), cc.T1_LABELS)
        ind = score(held.label.to_numpy(), full.predict(held.text), cc.T1_LABELS)
        rows.append((e, m, ind, len(fit)))
    return rows


def run_t3():
    S = {s: cc.load_split("damage", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "test")}
    V = {s: cc.t3_vectors(cc.cached(ART / "emb" / f"clip_damage_{s}.npy", S[s].image_id.tolist(), no_miss),
                          cc.cached(ART / "emb" / f"sbert_damage_{s}.npy", S[s].image_id.tolist(), no_miss)) for s in S}
    tr, te = S["train"], S["test"]
    labs = tr.label.to_numpy()
    # standard (in-distribution) index = v2's tweet-disjoint train index
    keep_v2 = ~tr.tweet_id.isin(set(te.tweet_id)).to_numpy()
    ix = faiss.IndexFlatIP(V["train"].shape[1]); ix.add(V["train"][keep_v2])
    sims, nn_ = ix.search(V["test"], 5)
    pred_in = np.array([cc.knn_vote(list(labs[keep_v2][r]), list(s))[0] for r, s in zip(nn_, sims)])
    rows = []
    for e in sorted(te.event_name.unique()):
        keep = (tr.event_name != e).to_numpy()
        held = (te.event_name == e).to_numpy()
        idx_meta = tr[keep]
        assert (idx_meta.event_name == e).sum() == 0 and not set(idx_meta.tweet_id) & set(te[held].tweet_id) \
            and not set(idx_meta.image_id) & set(te[held].image_id), "LEAK"
        index = faiss.IndexFlatIP(V["train"].shape[1]); index.add(V["train"][keep])
        sims, nn_ = index.search(V["test"][held], 5)
        pred = [cc.knn_vote(list(labs[keep][r]), list(s))[0] for r, s in zip(nn_, sims)]
        y = te.label.to_numpy()[held]
        rows.append((e, score(y, pred, cc.T3_LABELS), score(y, pred_in[held], cc.T3_LABELS), int(keep.sum())))
    return rows


for task, rows in (("t1", run_t1()), ("t3", run_t3())):
    avg = {k: float(np.mean([r[1][k] for r in rows])) for k in ("acc", "macro_f1", "weighted_f1", "macro_f1_fixed_labels")}
    avg_in = {k: float(np.mean([r[2][k] for r in rows])) for k in ("acc", "macro_f1", "weighted_f1", "macro_f1_fixed_labels")}
    for e, m, ind, n_fit in rows:
        write_row("C", f"c2_{task}_{e}", 42, dict(task=task.upper(), tier="cpu", held_out_event=e, n_train_or_index=n_fit,
                                                  heldout_rows_in_training=0, n=m["n"], test_acc=m["acc"], test_macro_f1=m["macro_f1"],
                                                  test_weighted_f1=m["weighted_f1"], test_macro_f1_fixed_labels=m["macro_f1_fixed_labels"],
                                                  in_dist_acc=ind["acc"], in_dist_macro_f1=ind["macro_f1"], in_dist_weighted_f1=ind["weighted_f1"]))
        print(f"{task.upper()} {e:22s} n={m['n']:4d} LODO acc {m['acc']:.4f} F1 {m['macro_f1']:.4f} | in-dist acc {ind['acc']:.4f} F1 {ind['macro_f1']:.4f}")
    write_row("C", f"c2_{task}_macro_avg", 42, dict(task=task.upper(), tier="cpu", folds=len(rows), test_acc=avg["acc"],
                                                     test_macro_f1=avg["macro_f1"], test_weighted_f1=avg["weighted_f1"],
                                                     test_macro_f1_fixed_labels=avg["macro_f1_fixed_labels"],
                                                     in_dist_acc=avg_in["acc"], in_dist_macro_f1=avg_in["macro_f1"],
                                                     in_dist_weighted_f1=avg_in["weighted_f1"], n=int(sum(r[1]["n"] for r in rows))))
    print(f"{task.upper()} macro average over {len(rows)} events: LODO acc {avg['acc']:.4f} F1 {avg['macro_f1']:.4f} | in-dist acc {avg_in['acc']:.4f} F1 {avg_in['macro_f1']:.4f}")
