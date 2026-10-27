"""Aggregate results/ablation/{A,B,C}/*.json into summary.csv, tables.tex and ladder_t2.pdf (ABLATION_PLAN.md deliverables).

mean +- sample std (ddof=1) over seeds; `n` = number of runs aggregated (deterministic T1/T3 rows have n = 1 and no std).
Tables use IEEEtran conventions (booktabs, caption above, \\footnotesize); no IEEE report/template exists in the repo to copy from.
"""
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results" / "ablation"
METRICS = ["dev_macro_f1", "test_acc", "test_macro_f1", "test_weighted_f1"]

# ── load ──────────────────────────────────────────────────────────────────────
runs = []
for f in sorted(glob.glob(str(RES / "[ABC]" / "*.json"))):
    d = json.load(open(f))
    if d.get("aborted"):
        continue
    runs.append(d)


def agg(rows, keys):
    out = {"n": len(rows)}
    for k in keys:
        v = np.array([r[k] for r in rows if k in r and r[k] is not None], dtype=float)
        out[k + "_mean"] = float(v.mean()) if len(v) else np.nan
        out[k + "_std"] = float(v.std(ddof=1)) if len(v) > 1 else np.nan
    return out


groups = defaultdict(list)
for r in runs:
    groups[(r["part"], r["task"], r["config"])].append(r)

# C1 rows carry several scores per run; expose each as its own summary config
c1 = [r for r in runs if r["config"].startswith("c1_")]
c1_groups = defaultdict(list)
for r in c1:
    for cond in ("full", "image_grey", "text_empty"):
        c1_groups[(r["part"], r["task"], f"{r['config']}_{cond}")].append(
            {"test_acc": r[f"{cond}_acc"], "test_macro_f1": r[f"{cond}_macro_f1"], "test_weighted_f1": r[f"{cond}_weighted_f1"]})
rows = []
for (part, task, cfg), rs in sorted(groups.items()):
    if cfg.startswith("c1_"):
        continue
    extra = {"n_params": rs[0].get("n_params"), "tier": rs[0].get("tier"), "n_test": rs[0].get("n_test"),
             "index_rows": rs[0].get("index_rows")}
    rows.append({"part": part, "task": task, "config": cfg, **extra, **agg(rs, METRICS)})
for (part, task, cfg), rs in sorted(c1_groups.items()):
    rows.append({"part": part, "task": task, "config": cfg, **agg(rs, METRICS)})
# A1 (results/ablation/X/a1_*): one summary config per (model, subset)
x1 = {m: [json.load(open(f)) for f in sorted(glob.glob(str(RES / "X" / f"a1_{m}_seed*.json")))] for m in ("L4a", "image_only", "text_only")}
for m, rs_ in x1.items():
    for sub in ("agreed", "disagreed", "all"):
        if rs_:
            rows.append({"part": "X", "task": "T2", "config": f"a1_{m}_{sub}", "n_test": rs_[0]["subsets"][sub]["n"],
                         **agg([{"test_acc": r["subsets"][sub]["acc"], "test_macro_f1": r["subsets"][sub]["macro_f1"],
                                 "test_weighted_f1": r["subsets"][sub]["weighted_f1"]} for r in rs_], METRICS)})
# D (results/ablation/X/d_lodo_t2_*): Task 2 leave-one-disaster-out with the image-only proxy; per-seed macro averages over the 7 events and the pooled 2,237-row scores
D_EV = sorted({Path(f).name.split("_seed")[0].removeprefix("d_lodo_t2_") for f in glob.glob(str(RES / "X" / "d_lodo_t2_*_seed*.json"))} - {"overall", "perclass"})
dres = {e: [json.load(open(f)) for f in sorted(glob.glob(str(RES / "X" / f"d_lodo_t2_{e}_seed*.json")))] for e in D_EV}
dov = json.load(open(RES / "X" / "d_lodo_t2_overall_seed42.json")) if (RES / "X" / "d_lodo_t2_overall_seed42.json").exists() else None
if dres and dov:
    dseeds = sorted({r["seed"] for rs_ in dres.values() for r in rs_})
    for e, rs_ in dres.items():
        for kind in ("lodo", "indist"):
            rows.append({"part": "X", "task": "T2", "config": f"d_t2_{kind}_{e}", "n_test": rs_[0]["n_test"],
                         **agg([{"test_acc": r[kind]["acc"], "test_macro_f1": r[kind]["macro_f1"], "test_weighted_f1": r[kind]["weighted_f1"]} for r in rs_], METRICS)})
    for kind in ("lodo", "indist"):
        per_seed = [{k: np.mean([[r[kind][m] for r in rs_ if r["seed"] == sd][0] for rs_ in dres.values()])
                     for k, m in (("test_acc", "acc"), ("test_macro_f1", "macro_f1"), ("test_weighted_f1", "weighted_f1"))} for sd in dseeds]
        rows.append({"part": "X", "task": "T2", "config": f"d_t2_{kind}_macro_avg", "n_test": sum(rs_[0]["n_test"] for rs_ in dres.values()), **agg(per_seed, METRICS)})
        rows.append({"part": "X", "task": "T2", "config": f"d_t2_{kind}_pooled", "n_test": dov["n_test"],
                     **agg([{"test_acc": v[kind]["acc"], "test_macro_f1": v[kind]["macro_f1"], "test_weighted_f1": v[kind]["weighted_f1"]}
                            for v in dov["pooled_by_seed"].values()], METRICS)})
# B (results/ablation/X/b_*): CLIP ViT-L/14 extension -- T2 image-only (3 seeds), T3 rows and zero-shot (deterministic)
bt2 = {c: [json.load(open(f)) for f in sorted(glob.glob(str(RES / "X" / f"b_t2_imageonly_{c}_seed*.json")))] for c in ("B32_raw", "L14_norm", "L14_raw")}
for c, rs_ in bt2.items():
    if rs_:
        rows.append({"part": "X", "task": "T2", "config": f"b_t2_imageonly_{c}", "n_test": rs_[0]["n_test"],
                     **agg([{"dev_macro_f1": r["dev_macro_f1"], "test_acc": r["test_acc"], "test_macro_f1": r["test_macro_f1"], "test_weighted_f1": r["test_weighted_f1"]} for r in rs_], METRICS)})
for enc in ("B32", "L14"):
    for emb in ("fused", "image"):
        for clf_ in ("knn5", "probe"):
            f = RES / "X" / f"b_t3_{enc}_{emb}_{clf_}_seed42.json"
            if f.exists():
                r = json.load(open(f))
                rows.append({"part": "X", "task": "T3", "config": f"b_t3_{enc}_{emb}_{clf_}", "n_test": r["n_test"], "index_rows": r["index_rows_test"],
                             **agg([{k: r[k] for k in METRICS}], METRICS)})
bzs = json.load(open(RES / "X" / "b_zeroshot_l14_seed42.json")) if (RES / "X" / "b_zeroshot_l14_seed42.json").exists() else None
if bzs:
    for tag, tst, dev_ in (("l14", bzs["test_l14_chosen"], bzs["dev_candidates_l14"][bzs["chosen_l14"]]["macro_f1"]),
                           ("b32", bzs["test_b32_zero_shot"], bzs["dev_candidates_b32"][bzs["chosen_b32"]])):
        rows.append({"part": "X", "task": "T3", "config": f"b_zeroshot_{tag}_{bzs['chosen_l14'] if tag == 'l14' else bzs['chosen_b32']}", "n_test": bzs["n_test"],
                     **agg([{"dev_macro_f1": dev_, "test_acc": tst["acc"], "test_macro_f1": tst["macro_f1"], "test_weighted_f1": tst["weighted_f1"]}], METRICS)})
summary = pd.DataFrame(rows)
cols =["part", "task", "config", "n"] + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")] + ["n_params", "n_test", "index_rows", "tier"]
summary = summary[[c for c in cols if c in summary.columns]]
summary.to_csv(RES / "summary.csv", index=False, float_format="%.4f")
S = summary.set_index(["part", "config"])


# ── helpers ───────────────────────────────────────────────────────────────────
def pm(cfg_key, metric, part=None, digits=3):
    """'0.587 $\\pm$ 0.004' (or just the mean when n = 1); '--' when missing."""
    try:
        r = S.loc[cfg_key]
    except KeyError:
        return "--"
    m, s = r[f"{metric}_mean"], r[f"{metric}_std"]
    if np.isnan(m):
        return "--"
    return f"{m:.{digits}f}" if np.isnan(s) else f"{m:.{digits}f} $\\pm$ {s:.{digits}f}"


def n_of(key):
    try:
        return int(S.loc[key]["n"])
    except KeyError:
        return 0


def esc(s):
    return s.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def table(caption, label, header, body, spec, fit=False):
    """fit=True (wide tables): \\small, \\tabcolsep 3pt, and the tabular is shrunk to the column width only if it is wider (graphicx \\resizebox)."""
    size = "\\small\n\\setlength{\\tabcolsep}{3pt}\n" if fit else "\\footnotesize\n"
    open_ = "\\resizebox{\\ifdim\\width>\\columnwidth\\columnwidth\\else\\width\\fi}{!}{%\n" if fit else ""
    close = "}" if fit else ""
    return ("\\begin{table}[t]\n\\centering\n" + size + "\\caption{" + caption + "}\n\\label{" + label + "}\n" + open_ +
            "\\begin{tabular}{" + spec + "}\n\\toprule\n" + " & ".join(header) + " \\\\\n\\midrule\n" +
            "\n".join(" & ".join(r) + " \\\\" for r in body) + "\n\\bottomrule\n\\end{tabular}" + close + "\n\\end{table}\n")


tex = ["% Generated by ablation/summarize.py. n = number of runs aggregated; mean $\\pm$ sample std over seeds.\n"
       "% Requires \\usepackage{booktabs}.\n"]

# ── Part A, T2 ladder ─────────────────────────────────────────────────────────
LADDER = [("L0", "v1 buggy train/dev, GELU CLIP, random crop, 30 ep"), ("L1", "+ full corrected train/dev"),
          ("L2", "+ QuickGELU CLIP"), ("L3", "+ epoch budget 30 $\\to$ 10"),
          ("L3b", "+ deterministic transform, cached features"), ("L4a", "+ AMP (pretrained DistilBERT)"),
          ("L4", "v2: random-init DistilBERT")]
body = [[r, desc, str(n_of(("A", f"t2_{r}"))), pm(("A", f"t2_{r}"), "dev_macro_f1"), pm(("A", f"t2_{r}"), "test_acc"),
         pm(("A", f"t2_{r}"), "test_macro_f1"), pm(("A", f"t2_{r}"), "test_weighted_f1")] for r, desc in LADDER]
tex.append(table("Task 2 ladder from the original notebook (L0) to v2 (L4). All rungs are scored on the full corrected test set (2237 rows).",
                 "tab:t2_ladder", ["Rung", "Change", "$n$", "Dev macro-F1", "Acc.", "Macro-F1", "Weighted-F1"], body, "llcllll"))

# ── Part A, T3 ladder ─────────────────────────────────────────────────────────
T3L = [("L0", "v1 index (incl. test, ``Don't know'' pollution)"), ("L1", "test rows removed"),
       ("L2", "pollution removed (train+dev)"), ("L3", "dev removed (train only)"), ("L4", "tweet siblings removed (= v2)")]
body = [[r, desc, str(int(S.loc[("A", f"t3_{r}")]["index_rows"])) if ("A", f"t3_{r}") in S.index else "--",
         pm(("A", f"t3_{r}"), "test_acc", digits=4), pm(("A", f"t3_{r}"), "test_macro_f1", digits=4),
         pm(("A", f"t3_{r}"), "test_weighted_f1", digits=4)] for r, desc in T3L]
tex.append(table("Task 3 ladder (deterministic, $n=1$ per rung). Every rung is scored on the same 529 test queries (3 real classes).",
                 "tab:t3_ladder", ["Rung", "Index", "Rows", "Acc.", "Macro-F1", "Weighted-F1"], body, "llrrrr"))

# ── Part B, T2 ────────────────────────────────────────────────────────────────
names = [("L4a", "full model (L4a)", ("A", "t2_L4a")), ("text_only", "text only", ("B", "t2_text_only")),
         ("image_only", "image only", ("B", "t2_image_only")), ("concat", "concat, no attention", ("B", "t2_concat")),
         ("no_t2i", "no text$\\to$image attention", ("B", "t2_no_t2i"))]
body = []
for _, desc, key in names:
    p = S.loc[key]["n_params"] if key in S.index else None
    body.append([desc, str(n_of(key)), f"{int(p):,}".replace(",", "\\,") if p is not None and not pd.isna(p) else "--",
                 pm(key, "dev_macro_f1"), pm(key, "test_acc"), pm(key, "test_macro_f1"), pm(key, "test_weighted_f1")])
tex.append(table("Task 2 component ablation on L4a's configuration (10 epochs, cached CLIP, AMP, pretrained DistilBERT).",
                 "tab:t2_partb", ["Variant", "$n$", "Trainable params", "Dev macro-F1", "Acc.", "Macro-F1", "Weighted-F1"], body, "lcrllll"))

# ── Part B, T3 ────────────────────────────────────────────────────────────────
def t3(emb, k, how):
    return ("B", f"t3_{emb}_k{k}_{how}")


body = [[str(k), pm(t3("fused", k, "majority"), "dev_macro_f1", digits=4), pm(t3("fused", k, "majority"), "test_acc", digits=4),
         pm(t3("fused", k, "majority"), "test_macro_f1", digits=4)] for k in (1, 3, 5, 10, 20)]
tex.append(table("Task 3: effect of $k$ (fused embedding, majority vote, tweet-disjoint index). Headline is $k=5$. Deterministic, $n=1$.",
                 "tab:t3_k", ["$k$", "Dev macro-F1", "Test acc.", "Test macro-F1"], body, "rrrr"))
VN = {"majority": "majority (v2)", "sim_weighted": "similarity-weighted", "class_balanced": "class-balanced"}
body = [[VN[h], pm(t3("fused", 5, h), "dev_macro_f1", digits=4), pm(t3("fused", 5, h), "test_acc", digits=4),
         pm(t3("fused", 5, h), "test_macro_f1", digits=4)] for h in VN]
tex.append(table("Task 3: voting rule ($k=5$, fused embedding). Deterministic, $n=1$.", "tab:t3_vote",
                 ["Vote", "Dev macro-F1", "Test acc.", "Test macro-F1"], body, "lrrr"))
body = []
for emb in ("image", "text", "fused"):
    body.append([emb, "5-NN majority", pm(t3(emb, 5, "majority"), "dev_macro_f1", digits=4), pm(t3(emb, 5, "majority"), "test_acc", digits=4),
                 pm(t3(emb, 5, "majority"), "test_macro_f1", digits=4)])
    body.append([emb, "logistic regression", pm(("B", f"t3_lr_{emb}"), "dev_macro_f1", digits=4), pm(("B", f"t3_lr_{emb}"), "test_acc", digits=4),
                 pm(("B", f"t3_lr_{emb}"), "test_macro_f1", digits=4)])
tex.append(table("Task 3: embedding and classifier (index/training rows identical; logistic-regression $C$ tuned on dev). Deterministic, $n=1$.",
                 "tab:t3_emb", ["Embedding", "Classifier", "Dev macro-F1", "Test acc.", "Test macro-F1"], body, "llrrr"))

# ── Part C1 ───────────────────────────────────────────────────────────────────
body = []
for lab, base in (("T2 L4a", "c1_t2_L4a"), ("T2 L4 (v2)", "c1_t2_L4"), ("T3 5-NN", "c1_t3")):
    task = "T3" if "t3" in base else "T2"
    f = ("C", f"{base}_full")
    for cond, cname in (("full", "full input"), ("image_grey", "grey image"), ("text_empty", "empty text")):
        key = ("C", f"{base}_{cond}")
        if key not in S.index:
            continue
        drop = "--" if cond == "full" else f"{S.loc[key]['test_macro_f1_mean'] - S.loc[f]['test_macro_f1_mean']:+.3f}"
        body.append([lab if cond == "full" else "", cname, str(n_of(key)), pm(key, "test_acc"), pm(key, "test_macro_f1"), drop])
tex.append(table("Missing-modality robustness at test time (no retraining). $\\Delta$ is the macro-F1 change from the full-input score.",
                 "tab:c1", ["Model", "Condition", "$n$", "Acc.", "Macro-F1", "$\\Delta$ macro-F1"], body, "llcrrr"))

# ── Part C2 ───────────────────────────────────────────────────────────────────
EV = ["california_wildfires", "hurricane_harvey", "hurricane_irma", "hurricane_maria", "iraq_iran_earthquake", "mexico_earthquake", "srilanka_floods"]
for task, t in (("T1", "t1"), ("T3", "t3")):
    body = []
    for e in EV:
        f = RES / "C" / f"c2_{t}_{e}_seed42.json"
        if f.exists():
            d = json.load(open(f))
            body.append([esc(e), str(d["n"]), f"{d['test_acc']:.3f}", f"{d['test_macro_f1']:.3f}", f"{d['in_dist_macro_f1']:.3f}"])
    f = RES / "C" / f"c2_{t}_macro_avg_seed42.json"
    if f.exists():
        d = json.load(open(f))
        body.append(["\\textbf{macro average}", str(d["n"]), f"{d['test_acc']:.3f}", f"{d['test_macro_f1']:.3f}", f"{d['in_dist_macro_f1']:.3f}"])
    tex.append(table(f"Leave-one-disaster-out, {task} (train/index on the other six events' train split; deterministic, $n=1$ per fold). "
                     "Per-event macro-F1 averages the classes present in that event. In-dist.\\ = standard model on the same rows.",
                     f"tab:c2_{t}", ["Held-out event", "Test rows", "Acc.", "LODO macro-F1", "In-dist. macro-F1"], body, "lrrrr"))
# ── D: Task 2 leave-one-disaster-out (image-only proxy) ───────────────────────
if dres and dov:
    pmx = lambda a: f"{np.mean(a):.3f} $\\pm$ {np.std(a, ddof=1):.3f}"
    body = []
    for e in EV:
        rs_ = dres[e]
        body.append([esc(e), str(rs_[0]["n_test"]), str(len(rs_)), pmx([r["lodo"]["acc"] for r in rs_]), pmx([r["lodo"]["macro_f1"] for r in rs_]),
                     pmx([r["indist"]["macro_f1"] for r in rs_]), f"{rs_[0]['majority_event']['macro_f1']:.3f}"])
    for kind_row, lab in (("macro_avg", "\\textbf{macro average}"), ("pooled", "pooled")):
        g = lambda k, m: [json.load(open(RES / "X" / "d_lodo_t2_overall_seed42.json"))["pooled_by_seed"][str(sd)][k][m] for sd in dseeds] if kind_row == "pooled" else \
            [np.mean([[r[k][m] for r in rs_ if r["seed"] == sd][0] for rs_ in dres.values()]) for sd in dseeds]
        body.append([lab, str(dov["n_test"]), str(len(dseeds)), pmx(g("lodo", "acc")), pmx(g("lodo", "macro_f1")), pmx(g("indist", "macro_f1")),
                     f"{np.mean([dres[e][0]['majority_event']['macro_f1'] for e in EV]):.3f}" if kind_row == "macro_avg" else "--"])
    tex.append(table("Leave-one-disaster-out, Task 2, with the image-only model on cached CLIP features (a proxy for L4a): train and select the checkpoint on the other six events' "
                     "train and dev rows, test on the held-out event; mean $\\pm$ std over three seeds. Per-event macro-F1 averages the classes present in that event. "
                     "In-dist.\\ = the existing image-only models (all events seen) on the same rows; Majority = always predicting the event's most frequent label. "
                     "The pooled row scores all 2237 test rows at once (macro-F1 over the 8 classes).",
                     "tab:d_t2", ["Held-out event", "Test rows", "$n$", "LODO acc.", "LODO macro-F1", "In-dist. macro-F1", "Majority macro-F1"], body, "lrcrrrr"))
# ── B: image encoder scale (CLIP ViT-B/32 vs ViT-L/14) ────────────────────────
if all(bt2.values()) and bzs and (RES / "X" / "b_t3_L14_fused_probe_seed42.json").exists():
    pmb = lambda a: f"{np.mean(a):.3f} $\\pm$ {np.std(a, ddof=1):.3f}"
    e1j = lambda n: json.load(open(RES / "X" / f"e1_{n}_seed42.json"))
    ci = lambda r: f"[{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]"
    t2m = lambda c: [r["test_macro_f1"] for r in bt2[c]]
    t3m = lambda enc, emb, c_: json.load(open(RES / "X" / f"b_t3_{enc}_{emb}_{c_}_seed42.json"))["test_macro_f1"]
    l4a_m = [json.load(open(f))["test_macro_f1"] for f in sorted(glob.glob(str(RES / "A" / "t2_L4a_seed*.json")))]
    body = []
    for lab, ref, new, nm in (("T2 image-only (L/14 L2-normalised)", t2m("B32_raw"), t2m("L14_norm"), "t2_imageonlyL14_vs_B32"),
                              ("T2 image-only (L/14 raw input, control)", t2m("B32_raw"), t2m("L14_raw"), "t2_imageonlyL14raw_vs_B32"),
                              ("T2 image-only L/14 against L4a (full model, B/32)", l4a_m, t2m("L14_norm"), "t2_imageonlyL14_vs_L4a")):
        r = e1j(nm)
        body.append([lab, str(len(new)), pmb(ref), pmb(new), f"{r['diff']:+.3f}", ci(r)])
    for lab, emb, c_, nm in (("T3 fused, logistic probe", "fused", "probe", "t3_probeL14_vs_B32_fused"), ("T3 fused, 5-NN", "fused", "knn5", "t3_knnL14_vs_B32_fused"),
                             ("T3 image-only, logistic probe", "image", "probe", None), ("T3 image-only, 5-NN", "image", "knn5", None)):
        r = e1j(nm) if nm else None
        body.append([lab, "1", f"{t3m('B32', emb, c_):.3f}", f"{t3m('L14', emb, c_):.3f}", f"{r['diff']:+.3f}" if r else "--", ci(r) if r else "--"])
    body.append([f"T3 zero-shot (prompt set {bzs['chosen_l14']})", "1", f"{bzs['test_b32_zero_shot']['macro_f1']:.3f}", f"{bzs['test_l14_chosen']['macro_f1']:.3f}", "--", "--"])
    tex.append(table("Image encoder scale: CLIP ViT-B/32 against ViT-L/14, test macro-F1. Task 2 image-only model: mean $\\pm$ std over three seeds; Task 3 and zero-shot: deterministic, "
                     "$n=1$, 529 test queries. Difference and 95\\% interval: paired bootstrap over the test rows at seed 42 (Task 2) or over the 529 queries (Task 3), L/14 minus the "
                     "first column. L/14 differs from B/32 in both size and patch size (14 against 32).",
                     "tab:b_scale", ["Setting", "$n$", "B/32 (or L4a)", "L/14", "Difference", "95\\% interval"], body, "lcrrrr"))
# ── A1: agreed vs disagreed subsets ───────────────────────────────────────────
if all(x1.values()):
    A1N = {"L4a": "L4a (full)", "image_only": "image only", "text_only": "text only"}
    mb = x1["L4a"][0]["majority_baseline"]
    body = []
    for sub, lab in (("agreed", "agreed"), ("disagreed", "disagreed"), ("all", "all rows")):
        body.append([lab, str(x1["L4a"][0]["subsets"][sub]["n"]), "majority class", "--", f"{mb[sub]['acc']:.3f}", f"{mb[sub]['macro_f1']:.3f}"])
        for m in ("L4a", "image_only", "text_only"):
            a_ = np.array([r["subsets"][sub]["acc"] for r in x1[m]])
            f_ = np.array([r["subsets"][sub]["macro_f1"] for r in x1[m]])
            body.append(["", "", A1N[m], str(len(a_)), f"{a_.mean():.3f} $\\pm$ {a_.std(ddof=1):.3f}", f"{f_.mean():.3f} $\\pm$ {f_.std(ddof=1):.3f}"])
    tex.append(table("Task 2 on test rows where the text and image labels agree or disagree (existing checkpoints, mean $\\pm$ std over seeds). "
                     "Macro-F1 averages the classes present in the subset; the majority-class row always predicts ``Not humanitarian''.",
                     "tab:a1", ["Subset", "Rows", "Model", "$n$", "Acc.", "Macro-F1"], body, "lrlcrr"))
# ── G: class-imbalance treatments (image-only head on cached CLIP features, B/32 and L/14) ───────────────────────────
if (RES / "G" / "g_imbalance_summary_seed42.json").exists():
    gF = json.load(open(RES / "G" / "g_imbalance_summary_seed42.json"))["feats"]
    G_ARMS = [("none", "none"), ("inv", "inv (current)"), ("sqrt_inv", "sqrt\\_inv"), ("eff_num", "eff\\_num"), ("focal1", "focal1"), ("focal2", "focal2"),
              ("focal2_inv", "focal2\\_inv"), ("sampler", "sampler"), ("logit_adj", "logit\\_adj"), ("none_posthoc", "none\\_posthoc")]
    body = []
    for a, lab in G_ARMS:
        row = [lab]
        for feat in ("b32", "l14"):
            x = gF[feat]["arms"][a]
            row += [f"{x['dev_f1']:.3f}", f"{x['test_f1']:.3f} $\\pm$ {x['test_f1_seed_std']:.3f}", f"{x['test_bal_acc']:.3f}"]
        body.append(row)
    tex.append(table("Task 2 class-imbalance treatments, image-only head on cached CLIP features, seeds 42, 43, 44. Dev macro-F1 and balanced accuracy are seed means; "
                     "test macro-F1 is the seed mean $\\pm$ the seed std. The first group of columns is CLIP B/32, the second CLIP L/14.",
                     "tab:g_arms", ["Arm", "Dev F1", "Test F1", "Bal. acc.", "Dev F1", "Test F1", "Bal. acc."], body, "lrrrrrr", fit=True))
(RES / "tables.tex").write_text("\n".join(tex), encoding="utf8")

# ── figure: T2 ladder ─────────────────────────────────────────────────────────
rungs = [r for r, _ in LADDER if ("A", f"t2_{r}") in S.index]
mean = [S.loc[("A", f"t2_{r}")]["test_macro_f1_mean"] for r in rungs]
std = [np.nan_to_num(S.loc[("A", f"t2_{r}")]["test_macro_f1_std"]) for r in rungs]
ns = [n_of(("A", f"t2_{r}")) for r in rungs]
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.bar(rungs, mean, yerr=std, capsize=4, color="#3b6ea5", ecolor="#222222", width=0.62)
for i, (m, n) in enumerate(zip(mean, ns)):
    ax.text(i, m + 0.012 + std[i], f"{m:.3f}\nn={n}", ha="center", va="bottom", fontsize=7)
ax.set_ylabel("Test macro-F1"); ax.set_ylim(0, max(mean) * 1.25 if mean else 1)
ax.set_title("Task 2: v1 $\\to$ v2 ladder (mean $\\pm$ std over seeds)", fontsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(RES / "ladder_t2.pdf")
if os.environ.get("LADDER_PNG"):   # optional preview for checking the figure
    fig.savefig(os.environ["LADDER_PNG"], dpi=130)
plt.close(fig)
print(summary.round(4).to_string(index=False))
print("wrote summary.csv, tables.tex, ladder_t2.pdf")
