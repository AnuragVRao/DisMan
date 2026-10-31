"""Day 4 step 6: error analysis, 12 misclassified TEST examples per task, chosen by a rule fixed before looking at any example.
Models: Task 1 = the deployed BERTweet (run bertweet_deploy seed 42, calibrated with the deployed dev temperature); Task 2 = L4a seed 42 (re-inferred run t2_L4a seed 42, the demo's model).
Population: every test row the model gets wrong. Category of each error, in this order of precedence (rule-based, NOT human-verified):
  1. text-image mismatch : the row's text label differs from its image label (normalised `label_text != label_image`), i.e. the two sources of annotation disagree about the tweet;
  2. label ambiguity     : Task 1: the calibrated top probability is below 0.65 (the model itself is unsure);
                           Task 2: (true, predicted) is one of the pre-declared confusable pairs  {Other relevant information <-> Rescue/volunteering/donation, Other relevant information <-> Not humanitarian,
                           Other relevant information <-> Infrastructure/utility damage, Affected individuals <-> Injured or dead people};
  3. model error         : everything else (labels agree, not a confusable pair / confident).
Sample: 4 errors per category, the first 4 by test-row index within each category; if a category has fewer than 4, the remaining slots go to the next categories in the order above, again first by index.
Writes docs/day4_error_analysis.md and docs/day4_error_analysis_table.tex. Does not use the guarded task loaders (it needs both tasks in one process)."""
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import metrics_all as M
from common import DATA, ROOT, SPLIT_DIR, cc

CATS = ("text-image mismatch", "label ambiguity", "model error")
PAIRS = {frozenset(p) for p in [("Other relevant information", "Rescue, volunteering or donation effort"), ("Other relevant information", "Not humanitarian"),
                                ("Other relevant information", "Infrastructure and utility damage"), ("Affected individuals", "Injured or dead people")]}
T1_AMBIG_P, PER_CAT = 0.65, 4


def tex(s):
    s = str(s).replace("\\", " ")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\~{}"), ("^", r"\^{}")):
        s = s.replace(a, b)
    return s


def pick(rows):
    """rows: list of dicts with 'cat' and 'i' (test index). 4 per category by index; shortfalls filled from the later categories, then the earlier ones."""
    by = {c: sorted([r for r in rows if r["cat"] == c], key=lambda r: r["i"]) for c in CATS}
    chosen = []
    for c in CATS:
        chosen += by[c][:PER_CAT]
    n = PER_CAT * len(CATS)
    if len(chosen) < n:
        rest = sorted([r for c in CATS for r in by[c][PER_CAT:]], key=lambda r: (CATS.index(r["cat"]), r["i"]))
        chosen += rest[:n - len(chosen)]
    return sorted(chosen, key=lambda r: (CATS.index(r["cat"]), r["i"]))


def load(task, split="test"):
    df = cc.load_split(task, split, DATA, SPLIT_DIR, verbose=False)[0]
    raw = pd.read_csv(SPLIT_DIR / f"task_{task}_text_img_{split}.tsv", sep="\t", dtype=str).drop_duplicates(subset=["image_id"]).set_index("image_id")
    norm = (lambda s: s.strip()) if task == "informative" else cc.t2_normalise
    df["text_label"] = raw.loc[df.image_id, "label_text"].map(norm).values
    df["image_label"] = raw.loc[df.image_id, "label_image"].map(norm).values
    return df


def analyse(task):
    df = load(task)
    if task == "informative":
        run = M.load_run("T1", "bertweet_deploy", 42); labels = cc.T1_LABELS
        T = json.load(open(ROOT / "artifacts" / "task1_bertweet" / "t1_meta.json"))["temperature_dev"]
        P = M.apply_temperature(np.log(np.clip(run["probs_test"], 1e-9, 1)), T); tag, model = "T1", "BERTweet (deployed; bertweet_deploy seed 42, dev temperature %.3f)" % T
    else:
        run = M.load_run("T2", "t2_L4a", 42); labels = sorted(cc.load_split("humanitarian", "train", DATA, SPLIT_DIR, verbose=False)[0].label.unique())
        P = np.array(run["probs_test"]); tag, model = "T2", "L4a seed 42 (the demo's Task 2 model)"
    y = np.array(run["y_test"]); pred = P.argmax(1)
    assert [labels[i] for i in y] == df["label"].tolist(), "stored labels do not follow the loader order"
    errs = []
    for i in np.where(pred != y)[0]:
        t, p = labels[y[i]], labels[pred[i]]
        if df.text_label.iloc[i] != df.image_label.iloc[i]:
            cat = CATS[0]
        elif (task == "informative" and P[i].max() < T1_AMBIG_P) or (task != "informative" and frozenset((t, p)) in PAIRS):
            cat = CATS[1]
        else:
            cat = CATS[2]
        errs.append(dict(i=int(i), cat=cat, true=t, pred=p, conf=float(P[i].max()), text_label=df.text_label.iloc[i], image_label=df.image_label.iloc[i], tweet=df.tweet_text.iloc[i],
                         image=str(df.rel_image.iloc[i]), tweet_id=df.tweet_id.iloc[i], event=df.event_name.iloc[i]))
    return tag, model, errs, len(y)


if __name__ == "__main__":
    md = ["# Day 4 error analysis (12 misclassified test examples per task)\n",
          "Selection is rule-based and fixed in advance (see the docstring of `ablation/d4_error_analysis.py`): the population is every test row the deployed model gets wrong; each error is assigned to the first matching category "
          "(text-image mismatch, then label ambiguity, then model error); the sample is the first 4 errors by test-row index per category (shortfalls filled in category order). "
          "**Categories are assigned by rule, not by human inspection**: *text-image mismatch* means the row's text label and image label disagree; *label ambiguity* means the calibrated Task 1 top probability is below 0.65, or, for Task 2, the (true, predicted) pair is one of four pre-declared confusable pairs; everything else is *model error*. "
          "A reviewer should read the examples before quoting any category as a cause.\n"]
    tex_rows, summary = [], {}
    for task in ("informative", "humanitarian"):
        tag, model, errs, n = analyse(task)
        cnt = Counter(e["cat"] for e in errs); sample = pick(errs)
        summary[tag] = dict(model=model, n_test=n, n_errors=len(errs), by_category=dict(cnt), sample_test_rows=[e["i"] for e in sample])
        md.append(f"\n## {tag}: {model}\n\n{len(errs)} errors on {n} test rows ({len(errs) / n:.1%}); by category: " + ", ".join(f"{c} {cnt.get(c, 0)}" for c in CATS) + ".\n")
        md.append("| # | test row | category | image (path under data/CrisisMMD_v2.0) | tweet | true | predicted (p) | text label / image label |\n|---|---|---|---|---|---|---|---|\n")
        for k, e in enumerate(sample, 1):
            tw = re.sub(r"\s+", " ", html.unescape(e["tweet"])).strip(); tw = tw if len(tw) <= 140 else tw[:137] + "..."
            md.append(f"| {tag}.{k} | {e['i']} | {e['cat']} | `{e['image']}` | {tw.replace('|', '/')} | {e['true']} | {e['pred']} ({e['conf']:.2f}) | {e['text_label']} / {e['image_label']} |\n")
            tex_rows.append(f"{tag}.{k} & {e['i']} & {tex(e['cat'])} & \\texttt{{{tex(Path(e['image']).name)}}} & {tex(tw[:90] + ('...' if len(tw) > 90 else ''))} & {tex(e['true'])} & {tex(e['pred'])} & {tex(e['text_label'])} / {tex(e['image_label'])} \\\\\n")
    Path(ROOT / "docs" / "day4_error_analysis.md").write_text("".join(md), encoding="utf-8")
    Path(ROOT / "docs" / "day4_error_analysis_table.tex").write_text(
        "% LaTeX-ready table, generated by ablation/d4_error_analysis.py (rule-based categories; see docs/day4_error_analysis.md)\n\\begin{table*}[t]\n\\centering\n\\scriptsize\n\\caption{Twelve misclassified test examples per task (Task 1: deployed BERTweet; Task 2: L4a seed 42), "
        "the first four by test-row index in each rule-assigned category. Categories are assigned by rule, not by human inspection.}\n\\label{tab:errors}\n\\begin{tabular}{llllp{4.6cm}p{2.2cm}p{2.2cm}p{2.4cm}}\n\\toprule\n"
        "\\# & Row & Category & Image & Tweet & True & Predicted & Text / image label \\\\\n\\midrule\n" + "".join(tex_rows) + "\\bottomrule\n\\end{tabular}\n\\end{table*}\n", encoding="utf-8")
    (ROOT / "results" / "ablation" / "D4_error_analysis.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
