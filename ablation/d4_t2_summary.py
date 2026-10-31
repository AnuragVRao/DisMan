"""Day 4 step 5 summary (Task 2 text models so far): provenance audit of every T2 run JSON, 3-seed tables (mean ± std, dev and test), the paired dev bootstrap between the two BERTweet arms
(SELECTION_RULE_DAY4.md rule 13 selection), and the comparison with the DistilBERT text-only runs (results/ablation/B, no stored predictions -> descriptive only).
Writes results/ablation/D4_T2_text_summary.{json,md}."""
import json

import numpy as np

import d4_prov
import metrics_all as M
from bootstrap_utils import N_BOOT, resample_idx
from common import RES

SEEDS = (42, 43, 44)
KEYS = ("macro_f1", "weighted_f1", "acc", "balanced_acc", "mcc", "top2_acc", "macro_roc_auc", "ece")
MODELS = [("t2_majority", "majority class ('Not humanitarian') (post-hoc)", 1), ("t2_tfidf_lr", "TF-IDF + LR (class-weighted)", 1),
          ("t2_tfidf_svm", "TF-IDF + LinearSVC (class-weighted, CV-calibrated on train)", 1), ("t2_tfidf_cnb", "TF-IDF + ComplementNB, unweighted (reference)", 1),
          ("t2_tfidf_cnb_bal", "TF-IDF + ComplementNB, balanced sample weights (post-hoc)", 1),
          ("bertweet_text_matched", "BERTweet text-only, DistilBERT-matched recipe (10 ep) [primary]", 3), ("bertweet_text_rule8", "BERTweet text-only, rule-8 recipe (4 ep, lr 2e-5)", 3)]


def macro_boot(y, p, idx, k=8):
    yt, yp = y[idx], p[idx]; f = 0.0
    for c in range(k):
        tp = ((yt == c) & (yp == c)).sum(1); s = (yt == c).sum(1); npd = (yp == c).sum(1)
        f = f + np.where(s + npd > 0, 2 * tp / np.maximum(s + npd, 1), 0.0)
    return f / k


if __name__ == "__main__":
    out = dict(audit={}, models={})
    for name, label, ns in MODELS:
        runs = [M.load_run("T2", name, s) for s in SEEDS[:ns]]
        for s, r in zip(SEEDS, runs):
            d4_prov.assert_run(r, "humanitarian", f"{name} seed{s}"); out["audit"][f"{name}_seed{s}"] = "PASS"
        row = dict(label=label, n_seeds=ns)
        for sp in ("dev", "test"):
            for k in KEYS:
                v = np.array([r[sp][k] for r in runs]); row[f"{sp}_{k}"] = [float(v.mean()), float(v.std(ddof=1)) if ns > 1 else None]
        row["test_per_class_f1"] = {l: float(np.mean([r["test"]["per_class"][l]["f1"] for r in runs])) for l in runs[0]["labels"]}
        row["test_per_class_recall"] = {l: float(np.mean([r["test"]["per_class"][l]["recall"] for r in runs])) for l in runs[0]["labels"]}
        row["best_epochs"] = [r.get("best_epoch") for r in runs]
        out["models"][name] = row
    # dev-based arm selection (rule 13)
    a, b = "bertweet_text_matched", "bertweet_text_rule8"
    ra, rb = [M.load_run("T2", a, s) for s in SEEDS], [M.load_run("T2", b, s) for s in SEEDS]
    y = ra[0]["y_dev"]; idx = resample_idx(len(y), N_BOOT)
    g = np.mean([macro_boot(y, ra[i]["probs_dev"].argmax(1), idx) - macro_boot(y, rb[i]["probs_dev"].argmax(1), idx) for i in range(3)], 0); lo, hi = np.percentile(g, [2.5, 97.5])
    pt = float(np.mean([ra[i]["dev"]["macro_f1"] - rb[i]["dev"]["macro_f1"] for i in range(3)]))
    tie = lo <= 0 <= hi
    out["arm_selection"] = dict(comparison="matched minus rule8, dev macro-F1, seed-averaged paired bootstrap", n_boot=N_BOOT, point=pt, ci95=[float(lo), float(hi)], tied=bool(tie),
                                pick=("bertweet_text_rule8 (tied on dev -> simpler, rule 13)" if tie else (a if pt > 0 else b)))
    ds = [json.load(open(RES / "B" / f"t2_text_only_seed{s}.json")) for s in SEEDS]
    out["distilbert_text_only"] = dict(source="results/ablation/B/t2_text_only_seed{42,43,44}.json (no stored predictions: no paired bootstrap possible)",
                                       test_macro_f1=[float(np.mean([d["test_macro_f1"] for d in ds])), float(np.std([d["test_macro_f1"] for d in ds], ddof=1))],
                                       dev_macro_f1=[float(np.mean([d["dev_macro_f1"] for d in ds])), float(np.std([d["dev_macro_f1"] for d in ds], ddof=1))],
                                       test_acc=float(np.mean([d["test_acc"] for d in ds])), test_weighted_f1=float(np.mean([d["test_weighted_f1"] for d in ds])),
                                       per_class_f1={l: float(np.mean([d["per_class_f1"][l] for d in ds])) for l in ds[0]["per_class_f1"]})
    (RES / "D4_T2_text_summary.json").write_text(json.dumps(out, indent=1))
    f = lambda v: f"{v[0]:.4f}" + (f" ± {v[1]:.4f}" if v[1] is not None else "")
    md = "# Task 2 text-only models (image label; mean ± std over seeds, deterministic models have 1 seed)\n\n| Model | dev macro-F1 | test macro-F1 | test wF1 | test acc | test bal-acc | test MCC | test top-2 | test ROC-AUC | test ECE |\n|---|---|---|---|---|---|---|---|---|---|\n"
    for n, r in out["models"].items():
        md += f"| {r['label']} | {f(r['dev_macro_f1'])} | {f(r['test_macro_f1'])} | {f(r['test_weighted_f1'])} | {f(r['test_acc'])} | {f(r['test_balanced_acc'])} | {f(r['test_mcc'])} | {f(r['test_top2_acc'])} | {f(r['test_macro_roc_auc'])} | {f(r['test_ece'])} |\n"
    d = out["distilbert_text_only"]
    md += f"| DistilBERT text-only (existing, Part B) | {d['dev_macro_f1'][0]:.4f} ± {d['dev_macro_f1'][1]:.4f} | {d['test_macro_f1'][0]:.4f} ± {d['test_macro_f1'][1]:.4f} | {d['test_weighted_f1']:.4f} | {d['test_acc']:.4f} | n/a | n/a | n/a | n/a | n/a |\n"
    s = out["arm_selection"]
    md += f"\nArm selection (rule 13): matched − rule8, dev macro-F1 {s['point']:+.4f}, 95% CI [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] -> {'tied' if s['tied'] else 'resolved'}; pick: {s['pick']}.\n"
    w = {n: out["models"][n]["dev_macro_f1"][0] for n in ("t2_tfidf_lr", "t2_tfidf_svm", "t2_tfidf_cnb_bal")}
    md += (f"\nClass-weighted classical models, selected on dev macro-F1: " + ", ".join(f"{n} {v:.4f}" for n, v in w.items()) + f" -> best: {max(w, key=w.get)}. "
           f"The unweighted ComplementNB (dev {out['models']['t2_tfidf_cnb']['dev_macro_f1'][0]:.4f}) is kept as a reference row; weighting did not help it.\n")
    bt = [r["dev"]["macro_f1"] for r in ra]; dd = [x["dev_macro_f1"] for x in ds]
    every = min(bt) > max(dd)
    out["bertweet_vs_distilbert_dev"] = dict(bertweet_matched_per_seed=bt, distilbert_per_seed=dd, every_bertweet_seed_exceeds_every_distilbert_seed=bool(every))
    md += ("\nBERTweet (matched recipe) vs DistilBERT text-only, per-seed dev macro-F1: BERTweet " + ", ".join(f"{v:.4f}" for v in bt) + "; DistilBERT " + ", ".join(f"{v:.4f}" for v in dd) + ". "
           + ("Better on dev at every seed (every BERTweet seed exceeds every DistilBERT seed; no paired CI: DistilBERT stored no predictions); test not resolvable (means differ by less than the seed spread). "
              if every else "Nominally better, not established. ")
           + "Confound: BERTweet reads the raw tweet text with its own normalisation, DistilBERT read the cleaned text, and the encoders differ in size, so the gap is not attributable to the encoder alone.\n")
    (RES / "D4_T2_text_summary.json").write_text(json.dumps(out, indent=1))
    md += "\n## Test per-class F1 (mean over seeds)\n\n| Class | " + " | ".join(["TF-IDF LR", "BERTweet matched", "BERTweet rule8", "DistilBERT"]) + " |\n|---|---|---|---|---|\n"
    for l in out["models"]["t2_tfidf_lr"]["test_per_class_f1"]:
        md += f"| {l} | {out['models']['t2_tfidf_lr']['test_per_class_f1'][l]:.3f} | {out['models'][a]['test_per_class_f1'][l]:.3f} | {out['models'][b]['test_per_class_f1'][l]:.3f} | {d['per_class_f1'][l]:.3f} |\n"
    (RES / "D4_T2_text_summary.md").write_text(md, encoding="utf-8"); print(md); print("audit:", set(out["audit"].values()), len(out["audit"]), "runs")
