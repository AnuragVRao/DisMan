"""G: Step 5 of the imbalance study (CPU only).
(a) class-count table for T1 / T2 / T3 (train/dev/test, max:min ratio) from the same loaders the models use (crisis_common.load_split);
(b) T1: TF-IDF + logistic regression, class_weight "balanced" (current) vs None (v1/v2 recipe otherwise, C=1);
(c) T3: logistic-regression probe on the image / text / fused embeddings, class_weight "balanced" (current) vs None, C tuned on DEV per arm over
    {0.01, 0.1, 1, 10, 100} (strict '>' on dev macro-F1), dev rows = damage-dev with its tweet siblings removed from the index, as t3_partb.py;
    plus the existing T3 5-NN majority vs class-balanced vote read from results/ablation/B/.
Both are deterministic single runs (no seed variation). Test is scored once per arm; the paired bootstrap (10,000 resamples) is for reporting only.
Writes results/ablation/G/g_class_counts_seed42.json, g_t1_balance_seed42.json, g_t3_balance_seed42.json.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from bootstrap_utils import macro_f1_vec, paired_diff_ci, resample_idx
from common import ART, DATA, RES, SPLIT_DIR, cc, write_row
from extra_metrics import full_metrics, macro_f1_ci

N_BOOT = 10_000
verdict = lambda ci: "better" if ci["ci95_low"] > 0 else "worse" if ci["ci95_high"] < 0 else "not distinguishable"


def no_miss():
    raise RuntimeError("cache miss")


def compare(task, labels, y, preds, score):
    """Metrics for each arm in `preds` plus the paired bootstrap of (balanced - none) macro-F1 on the test rows."""
    k = len(labels); arms = {}
    for a, p in preds.items():
        m = full_metrics(y, p, labels, score[a]); m["macro_f1_ci"] = macro_f1_ci(y, p, k)
        arms[a] = m
    idx = resample_idx(len(y), N_BOOT)
    v = {a: {0: macro_f1_vec(y, p, k, idx)} for a, p in preds.items()}
    d = paired_diff_ci(v["balanced"], v["none"]); d["verdict"] = verdict(d)
    print(f"\n{task}: {'arm':9s} {'acc':>6s} {'macroF1':>8s} {'test CI':>16s} {'balAcc':>7s} {'MCC':>6s} {'PR-AUC':>7s} | min-class recall")
    for a, m in arms.items():
        rec = min(v_["recall"] for v_ in m["per_class"].values())
        print(f"{task}: {a:9s} {m['acc']:6.3f} {m['macro_f1']:8.3f} [{m['macro_f1_ci']['ci95_low']:.3f}, {m['macro_f1_ci']['ci95_high']:.3f}] "
              f"{m['balanced_acc']:7.3f} {m['mcc']:6.3f} {m['macro_pr_auc']:7.3f} | {rec:.3f}")
    print(f"{task}: balanced - none macro-F1 (test, reporting only): {d['diff']:+.4f} [{d['ci95_low']:+.4f}, {d['ci95_high']:+.4f}] -> {d['verdict']}")
    return arms, d


def main():
    S = {t: {s: cc.load_split(t, s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "dev", "test")} for t in ("informative", "humanitarian", "damage")}
    # ---- (a) class counts ----
    counts = {}
    for t, d in S.items():
        labs = sorted(d["train"].label.unique())
        tab = {l: {s: int((d[s].label == l).sum()) for s in d} for l in labs}
        tr = [tab[l]["train"] for l in labs]
        counts[t] = dict(table=tab, n={s: len(d[s]) for s in d}, ratio_train=max(tr) / min(tr), ratio_test=max(tab[l]["test"] for l in labs) / max(1, min(tab[l]["test"] for l in labs)))
        print(f"\n{t}: n = {counts[t]['n']}  train max:min = {counts[t]['ratio_train']:.1f}:1  (test {counts[t]['ratio_test']:.1f}:1)")
        for l in sorted(labs, key=lambda l: -tab[l]["train"]):
            print(f"   {l:42s} train {tab[l]['train']:5d}  dev {tab[l]['dev']:5d}  test {tab[l]['test']:5d}")
    write_row("G", "g_class_counts", 42, dict(task="T1/T2/T3", tier="cpu", counts=counts))

    # ---- (b) T1 ----
    d = S["informative"]; labs = sorted(d["train"].label.unique()); l2i = {l: i for i, l in enumerate(labs)}
    ydv, yte = (np.array([l2i[l] for l in d[s].label]) for s in ("dev", "test"))
    preds, score, dev_f1 = {}, {}, {}
    for name, cw in (("balanced", "balanced"), ("none", None)):
        pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
                         ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight=cw, random_state=cc.SEED))]).fit(d["train"].text, d["train"].label)
        preds[name] = np.array([l2i[l] for l in pipe.predict(d["test"].text)]); score[name] = pipe.predict_proba(d["test"].text)
        dev_f1[name] = float(cc.metrics(d["dev"].label, pipe.predict(d["dev"].text), labs)["macro_f1"])
    arms, diff = compare("T1", labs, yte, preds, score)
    write_row("G", "g_t1_balance", 42, dict(task="T1", tier="cpu", labels=labs, dev_macro_f1=dev_f1, arms=arms, balanced_minus_none=diff))

    # ---- (c) T3 ----
    dm = S["damage"]; LAB = cc.T3_LABELS; l3 = {l: i for i, l in enumerate(LAB)}
    img = {s: cc.cached(ART / "emb" / f"clip_damage_{s}.npy", dm[s].image_id.tolist(), no_miss) for s in dm}
    txt = {s: cc.cached(ART / "emb" / f"sbert_damage_{s}.npy", dm[s].image_id.tolist(), no_miss) for s in dm}
    VEC = {"fused": {s: cc.t3_vectors(img[s], txt[s]) for s in dm}, "image": {s: cc.l2(img[s]).astype("float32") for s in dm},
           "text": {s: txt[s].astype("float32") for s in dm}}
    keep = {sp: ~dm["train"].tweet_id.isin(set(dm[sp].tweet_id)).to_numpy() for sp in ("dev", "test")}
    ylab = dm["train"].label.to_numpy(); ydv, yte = dm["dev"].label.to_numpy(), dm["test"].label.to_numpy()
    out = {}
    for emb in ("fused", "image", "text"):
        X = VEC[emb]; preds, score, chosen, devbest = {}, {}, {}, {}
        for name, cw in (("balanced", "balanced"), ("none", None)):
            best = None
            for C in (0.01, 0.1, 1, 10, 100):
                clf = LogisticRegression(C=C, class_weight=cw, max_iter=3000).fit(X["train"][keep["dev"]], ylab[keep["dev"]])
                f1 = cc.metrics(ydv, clf.predict(X["dev"]), LAB)["macro_f1"]
                if best is None or f1 > best[1]:
                    best = (C, f1)
            clf = LogisticRegression(C=best[0], class_weight=cw, max_iter=3000).fit(X["train"][keep["test"]], ylab[keep["test"]])
            assert list(clf.classes_) == sorted(LAB)
            P = clf.predict_proba(X["test"]); order = [list(clf.classes_).index(l) for l in LAB]
            preds[name] = np.array([l3[l] for l in clf.predict(X["test"])]); score[name] = P[:, order]
            chosen[name], devbest[name] = best[0], float(best[1])
        arms, diff = compare(f"T3[{emb}]", LAB, np.array([l3[l] for l in yte]), preds, score)
        out[emb] = dict(C=chosen, dev_macro_f1=devbest, arms=arms, balanced_minus_none=diff)
    knn = {v: json.load(open(RES / "B" / f"t3_fused_k5_{v}_seed42.json")) for v in ("majority", "class_balanced")}
    print("\nT3 5-NN fused (existing runs): majority test macro-F1 %.4f vs class_balanced %.4f" % (knn["majority"]["test_macro_f1"], knn["class_balanced"]["test_macro_f1"]))
    write_row("G", "g_t3_balance", 42, dict(task="T3", tier="cpu", probes=out,
                                           knn_fused_k5=dict(majority=knn["majority"]["test_macro_f1"], class_balanced=knn["class_balanced"]["test_macro_f1"])))


if __name__ == "__main__":
    main()
