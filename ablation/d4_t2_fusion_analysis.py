"""Day 4 step 5c/5d analysis, Task 2 (SELECTION_RULE_DAY4.md rules 15-19). Reads only stored run JSONs (dev and test probabilities).
 - results table (mean ± std over seeds), dev and test
 - pre-registered comparisons on DEV (paired bootstrap, seed-averaged, 10,000 resamples, seed 20241025; macro-F1 headline, weighted F1 secondary), test reported afterwards:
     (i) core model (BERTweet+ResNet50 early fusion) vs L4a;  (ii) core model vs image-only CLIP B/32 head
 - post-hoc (rule 19): BERTweet text-only (matched) vs DistilBERT text-only, with the re-inferred DistilBERT predictions
 - fusion gain = fused - best single modality (per seed, best on dev macro-F1 among BERTweet rule8 text and the dev-chosen image head), macro-F1 and weighted F1, dev/test,
   whole test set and agreed / disagreed subsets (normalised label_text == label_image); subset numbers are descriptive.
Writes results/ablation/D4_T2_fusion.{json,md}."""
import json

import numpy as np
import pandas as pd

import d4_prov
import d4_t2_data as d4
import metrics_all as M
from bootstrap_utils import N_BOOT, resample_idx
from common import RES, SPLIT_DIR, cc

SEEDS = (42, 43, 44)
TEXT, IMG, IMG_B32 = "bertweet_text_rule8", "t2_img_clip_l14_inv", "t2_img_clip_b32"
LATE, CORE, L4A = "late_bertweet_text_rule8+t2_img_clip_l14_inv", "t2_early_bertweet_resnet50", "t2_L4a"
TABLE = [("t2_majority", "majority class ('Not humanitarian')", 1), ("t2_tfidf_lr", "TF-IDF + LR (class-weighted)", 1), ("bertweet_text_rule8", "BERTweet text-only (rule-8 recipe; rule-13 pick)", 3),
         ("bertweet_text_matched", "BERTweet text-only (DistilBERT-matched recipe)", 3), ("t2_distilbert_text", "DistilBERT text-only (re-inferred)", 3),
         ("t2_img_clip_b32", "image-only head, CLIP B/32", 3), ("t2_img_clip_l14_inv", "image-only head, CLIP L/14", 3), ("t2_L4a", "L4a: CLIP B/32 + DistilBERT cross-attention (demo model)", 3),
         (LATE, "late fusion (calibrated avg): BERTweet + CLIP L/14 head", 3), (CORE, "early fusion BERTweet (fine-tuned) + ResNet50 (frozen) [guide's core model]", 3)]
KEYS = ("macro_f1", "weighted_f1", "acc", "balanced_acc", "mcc", "top2_acc", "macro_roc_auc")


def boot(y, p, idx, k=8):
    yt, yp = y[idx], p[idx]; fs = []; sup = []
    for c in range(k):
        tp = ((yt == c) & (yp == c)).sum(1); s = (yt == c).sum(1); npd = (yp == c).sum(1)
        fs.append(np.where(s + npd > 0, 2 * tp / np.maximum(s + npd, 1), 0.0)); sup.append(s)
    fs, sup = np.stack(fs), np.stack(sup)
    return dict(macro_f1=fs.mean(0), weighted_f1=(fs * sup).sum(0) / idx.shape[1])


def load(name, s):
    r = M.load_run("T2", name, s); d4_prov.assert_run(r, "humanitarian", f"{name} seed{s}"); return r


def compare(a, b, y, split):
    idx = resample_idx(len(y[split]), N_BOOT); R = {n: [load(n, s) for s in SEEDS] for n in (a, b)}
    d = {k: np.mean([boot(y[split], R[a][i][f"probs_{split}"].argmax(1), idx)[k] - boot(y[split], R[b][i][f"probs_{split}"].argmax(1), idx)[k] for i in range(3)], 0) for k in ("macro_f1", "weighted_f1")}
    out = {}
    for k in d:
        lo, hi = np.percentile(d[k], [2.5, 97.5])
        out[k] = dict(point=float(np.mean([R[a][i][split][k] - R[b][i][split][k] for i in range(3)])), ci95=[float(lo), float(hi)], verdict="better" if lo > 0 else "worse" if hi < 0 else "not distinguishable")
    return out


def subset(y, p, mask):
    ev = M.evaluate(y[mask], p[mask], d4_labels); return dict(n=int(mask.sum()), macro_f1=ev["macro_f1"], weighted_f1=ev["weighted_f1"], acc=ev["acc"])


if __name__ == "__main__":
    D = d4.data(); d4_labels = D["labels"]; y = {sp: D["y"][sp] for sp in ("dev", "test")}; out = dict(models={}, comparisons={}, posthoc={}, gains={}, subsets={})
    for n, label, ns in TABLE:
        runs = [load(n, s) for s in SEEDS[:ns]]; row = dict(label=label, n_seeds=ns)
        for sp in ("dev", "test"):
            for k in KEYS:
                v = np.array([r[sp][k] for r in runs]); row[f"{sp}_{k}"] = [float(v.mean()), float(v.std(ddof=1)) if ns > 1 else None]
        row["test_per_class_f1"] = {l: float(np.mean([r["test"]["per_class"][l]["f1"] for r in runs])) for l in d4_labels}
        row["test_confusion_row_norm"] = np.mean([np.array(r["test"]["confusion_row_norm"]) for r in runs], 0).round(4).tolist()
        out["models"][n] = row
    # pre-registered comparisons (dev decides)
    for key, (a, b) in {"core_vs_L4a": (CORE, L4A), "core_vs_image_only_clip_b32": (CORE, IMG_B32)}.items():
        out["comparisons"][key] = dict(a=a, b=b, dev=compare(a, b, y, "dev"), test=compare(a, b, y, "test"))
    out["posthoc"]["bertweet_matched_vs_distilbert_text"] = dict(a="bertweet_text_matched", b="t2_distilbert_text", dev=compare("bertweet_text_matched", "t2_distilbert_text", y, "dev"), test=compare("bertweet_text_matched", "t2_distilbert_text", y, "test"))
    # fusion gain
    raw = {sp: pd.read_csv(SPLIT_DIR / f"task_humanitarian_text_img_{sp}.tsv", sep="\t", dtype=str).drop_duplicates(subset=["image_id"]).set_index("image_id") for sp in ("dev", "test")}
    agree = {sp: (raw[sp].loc[D["ids"][sp], "label_text"].map(cc.t2_normalise).values == raw[sp].loc[D["ids"][sp], "label_image"].map(cc.t2_normalise).values) for sp in raw}
    out["agreement"] = {sp: dict(n_agree=int(agree[sp].sum()), n_disagree=int((~agree[sp]).sum()), not_humanitarian_share_agree=float((y[sp][agree[sp]] == d4_labels.index("Not humanitarian")).mean()),
                                 not_humanitarian_share_disagree=float((y[sp][~agree[sp]] == d4_labels.index("Not humanitarian")).mean())) for sp in agree}
    best_single = {}
    for s in SEEDS:
        t, i = load(TEXT, s), load(IMG, s); best_single[s] = t if t["dev"]["macro_f1"] >= i["dev"]["macro_f1"] else i
    for n in (LATE, CORE):
        per = []
        for s in SEEDS:
            f = load(n, s); b = best_single[s]
            row = {sp: {k: f[sp][k] - b[sp][k] for k in ("macro_f1", "weighted_f1")} for sp in ("dev", "test")}
            row["test_agreed"] = {k: subset(y["test"], f["probs_test"], agree["test"])[k] - subset(y["test"], b["probs_test"], agree["test"])[k] for k in ("macro_f1", "weighted_f1")}
            row["test_disagreed"] = {k: subset(y["test"], f["probs_test"], ~agree["test"])[k] - subset(y["test"], b["probs_test"], ~agree["test"])[k] for k in ("macro_f1", "weighted_f1")}
            row["best_single"] = b["name"]; per.append(row)
        keys = ("dev", "test", "test_agreed", "test_disagreed")
        out["gains"][n] = dict(per_seed=per, mean={g: {k: float(np.mean([p[g][k] for p in per])) for k in ("macro_f1", "weighted_f1")} for g in keys},
                               std={g: {k: float(np.std([p[g][k] for p in per], ddof=1)) for k in ("macro_f1", "weighted_f1")} for g in keys})
    for sp_name, m in (("agreed", agree["test"]), ("disagreed", ~agree["test"])):
        out["subsets"][sp_name] = {n: {k: float(np.mean([subset(y["test"], load(n, s)["probs_test"], m)[k] for s in SEEDS])) for k in ("macro_f1", "weighted_f1", "acc")} for n in (TEXT, IMG, IMG_B32, L4A, LATE, CORE)}
        out["subsets"][sp_name]["n"] = int(m.sum())
    (RES / "D4_T2_fusion.json").write_text(json.dumps(out, indent=1))
    f = lambda v: f"{v[0]:.4f}" + (f" ± {v[1]:.4f}" if v[1] is not None else ""); sg = lambda v: f"{v:+.4f}"
    md = "# Task 2 results, text / image / fusion (image label, 8 classes; mean ± std over seeds)\n\n| Model | dev macro-F1 | test macro-F1 | test wF1 | test acc | test bal-acc | test MCC | test top-2 | test ROC-AUC |\n|---|---|---|---|---|---|---|---|---|\n"
    for n, r in out["models"].items():
        md += f"| {r['label']} | {f(r['dev_macro_f1'])} | {f(r['test_macro_f1'])} | {f(r['test_weighted_f1'])} | {f(r['test_acc'])} | {f(r['test_balanced_acc'])} | {f(r['test_mcc'])} | {f(r['test_top2_acc'])} | {f(r['test_macro_roc_auc'])} |\n"
    md += "\n## Pre-registered comparisons (dev paired bootstrap, seed-averaged, 10,000 resamples; dev decides, test is reporting only)\n\n| Comparison | split | metric | point | 95% CI | verdict |\n|---|---|---|---|---|---|\n"
    for key, c in {**out["comparisons"], **out["posthoc"]}.items():
        for sp in ("dev", "test"):
            for k, v in c[sp].items():
                md += f"| {c['a']} − {c['b']} ({key}{', post-hoc' if key in out['posthoc'] else ''}) | {sp} | {k} | {sg(v['point'])} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | {v['verdict']} |\n"
    md += "\n## Fusion gain = fused − best single modality (per seed, best on dev macro-F1 among BERTweet text-only and the CLIP L/14 image head)\n\n| Fused model | dev gain macro-F1 | dev gain wF1 | test gain macro-F1 | test gain wF1 |\n|---|---|---|---|---|\n"
    for n, g in out["gains"].items():
        md += f"| {n} | {sg(g['mean']['dev']['macro_f1'])} ± {g['std']['dev']['macro_f1']:.4f} | {sg(g['mean']['dev']['weighted_f1'])} ± {g['std']['dev']['weighted_f1']:.4f} | {sg(g['mean']['test']['macro_f1'])} ± {g['std']['test']['macro_f1']:.4f} | {sg(g['mean']['test']['weighted_f1'])} ± {g['std']['test']['weighted_f1']:.4f} |\n"
    a = out["agreement"]["test"]
    md += (f"\n## Test set by label agreement (normalised label_text == label_image): agreed n={a['n_agree']}, disagreed n={a['n_disagree']}; 'Not humanitarian' share {a['not_humanitarian_share_agree']:.2f} (agreed) vs "
           f"{a['not_humanitarian_share_disagree']:.2f} (disagreed). Macro-F1 over all 8 classes (classes absent from a subset score 0), descriptive only.\n\n| Model | agreed macro-F1 | agreed wF1 | agreed acc | disagreed macro-F1 | disagreed wF1 | disagreed acc |\n|---|---|---|---|---|---|---|\n")
    for n in (TEXT, IMG, IMG_B32, L4A, LATE, CORE):
        md += f"| {n} | " + " | ".join(f"{out['subsets'][g][n][k]:.4f}" for g in ("agreed", "disagreed") for k in ("macro_f1", "weighted_f1", "acc")) + " |\n"
    md += "\n### Fusion gain on the subsets (test; fused − best single modality of that seed)\n\n| Fused model | agreed macro-F1 | agreed wF1 | disagreed macro-F1 | disagreed wF1 |\n|---|---|---|---|---|\n"
    for n, g in out["gains"].items():
        md += f"| {n} | {sg(g['mean']['test_agreed']['macro_f1'])} | {sg(g['mean']['test_agreed']['weighted_f1'])} | {sg(g['mean']['test_disagreed']['macro_f1'])} | {sg(g['mean']['test_disagreed']['weighted_f1'])} |\n"
    (RES / "D4_T2_fusion.md").write_text(md, encoding="utf-8"); print(md)
