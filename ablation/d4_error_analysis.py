"""Day 4 final round B1: error analysis, 12 misclassified TEST examples per task, chosen by the rule fixed in SELECTION_RULE_DAY4.md rule 22 BEFORE any example was looked at (this replaces the earlier
first-four-per-category sample). Models: Task 1 = bertweet_deploy (confidence = max probability after the dev temperature, 1.825); Task 2 = L4a seed 42 (temperature 1.185, demo/calibration.json).
Sample: rank the (true, predicted) pairs by their error count over the whole test set; walk the pairs and take the first rows by test-row index, at most 3 per pair (6 for Task 1, which has only two pairs), until 12.
Category (in this order): text-image mismatch (text label != image label) / label ambiguity (predicted class = the true class's most-confused partner AND (near-duplicate-like with a conflicting train label OR generic text of <= 5 words)) / model error.
Category counts are given over ALL test errors. Writes docs/day4_error_analysis.{md,_table.tex}, results/ablation/D4_error_analysis.json and contact sheets report_v2/figures/contact_T{1,2}.{pdf,png}.
Does not use the guarded task loaders (it needs both tasks in one process)."""
import html
import json
import re
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import d4_dup_utils as U
import metrics_all as M
from common import ART, DATA, ROOT, SPLIT_DIR, cc

CATS = ("text-image mismatch", "label ambiguity", "model error")
N_SAMPLE, CAP_DEFAULT, GENERIC_WORDS = 12, 3, 5
FIG = ROOT / "report_v2" / "figures"


def tex(s):
    s = str(s).replace("\\", " ")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\~{}"), ("^", r"\^{}")):
        s = s.replace(a, b)
    return s


def load(task, split):
    df = cc.load_split(task, split, DATA, SPLIT_DIR, verbose=False)[0]
    raw = pd.read_csv(SPLIT_DIR / f"task_{task}_text_img_{split}.tsv", sep="\t", dtype=str).drop_duplicates(subset=["image_id"]).set_index("image_id")
    norm = (lambda s: s.strip()) if task == "informative" else cc.t2_normalise
    df["text_label"] = raw.loc[df.image_id, "label_text"].map(norm).values; df["image_label"] = raw.loc[df.image_id, "label_image"].map(norm).values
    return df


def select(errs, labels_n):
    pairs = Counter((e["true"], e["pred"]) for e in errs)
    ranked = sorted(pairs, key=lambda p: (-pairs[p], p)); cap = CAP_DEFAULT if len(ranked) * CAP_DEFAULT >= N_SAMPLE else int(np.ceil(N_SAMPLE / len(ranked)))
    chosen = []
    for p in ranked:
        rows = sorted((e for e in errs if (e["true"], e["pred"]) == p), key=lambda e: e["i"])[:cap]
        chosen += rows
        if len(chosen) >= N_SAMPLE:
            break
    return chosen[:N_SAMPLE], ranked, pairs, cap


def analyse(task):
    tr, te = load(task, "train"), load(task, "test")
    if task == "informative":
        run = M.load_run("T1", "bertweet_deploy", 42); labels = cc.T1_LABELS; T = json.load(open(ROOT / "artifacts" / "task1_bertweet" / "t1_meta.json"))["temperature_dev"]
        P = M.apply_temperature(np.log(np.clip(run["probs_test"], 1e-9, 1)), T); tag, model = "T1", f"BERTweet (deployed checkpoint, bertweet_deploy seed 42; dev temperature {T:.3f})"
        ntr, nte = [U.norm_text(t) for t in tr.tweet_text.fillna("")], [U.norm_text(t) for t in te.tweet_text.fillna("")]
        sim = U.tfidf_sims(ntr, nte); thr = U.TEXT_THR; kind = "text"
    else:
        run = M.load_run("T2", "t2_L4a", 42); labels = sorted(tr.label.unique()); P = np.array(run["probs_test"]); T = json.load(open(ROOT / "demo" / "calibration.json"))["t2"]["temperature"]
        P = M.apply_temperature(np.log(np.clip(P, 1e-9, 1)), T); tag, model = "T2", f"L4a seed 42 (the demo's Task 2 model; temperature {T:.3f})"
        E = {s: cc.l2(cc.cached(ART / "emb" / f"clip_humanitarian_{s}.npy", d.image_id.tolist(), lambda: (_ for _ in ()).throw(RuntimeError("cache miss")))).astype("float32") for s, d in (("train", tr), ("test", te))}
        sim = E["test"] @ E["train"].T; thr = U.IMG_THR; kind = "image"
    y = np.array(run["y_test"]); pred = P.argmax(1); assert [labels[i] for i in y] == te["label"].tolist(), "stored labels do not follow the loader order"
    ytr = np.array([labels.index(l) for l in tr["label"]])
    err_idx = np.where(pred != y)[0]
    conf = Counter((int(y[i]), int(pred[i])) for i in err_idx)
    partner = {c: max((p for p in range(len(labels)) if p != c), key=lambda p: (conf.get((c, p), 0), -p)) for c in range(len(labels))}      # most-confused partner of class c
    errs = []
    for i in err_idx:
        t, p = int(y[i]), int(pred[i]); generic = U.n_words(te.tweet_text.iloc[i]) <= GENERIC_WORDS
        conflicting_dup = bool(((sim[i] >= thr) & (ytr != t)).any())
        mismatch = te.text_label.iloc[i] != te.image_label.iloc[i]
        ambiguity = (p == partner[t]) and (conflicting_dup or generic)
        cat = CATS[0] if mismatch else CATS[1] if ambiguity else CATS[2]
        errs.append(dict(i=int(i), cat=cat, true=labels[t], pred=labels[p], conf=float(P[i].max()), text_label=te.text_label.iloc[i], image_label=te.image_label.iloc[i], tweet=te.tweet_text.iloc[i],
                         image=str(te.rel_image.iloc[i]), img_path=te.img_path.iloc[i], tweet_id=te.tweet_id.iloc[i], event=te.event_name.iloc[i], is_partner=bool(p == partner[t]), near_dup_conflicting=conflicting_dup, generic=bool(generic)))
    return tag, model, errs, len(y), labels, partner, kind


def contact_sheet(tag, model, sample, path_stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    col = {CATS[0]: "#0072B2", CATS[1]: "#E69F00", CATS[2]: "#CC79A7"}                  # colourblind-safe (Okabe-Ito)
    fig, axes = plt.subplots(3, 4, figsize=(13.3, 11.2)); fig.patch.set_facecolor("white")
    for ax, e in zip(axes.ravel(), sample):
        try:
            im = Image.open(e["img_path"]).convert("RGBA"); bg = Image.new("RGBA", im.size, (255, 255, 255, 255)); im = Image.alpha_composite(bg, im).convert("RGB"); im.thumbnail((420, 300))
            canvas = Image.new("RGB", (420, 300), (255, 255, 255)); canvas.paste(im, ((420 - im.width) // 2, (300 - im.height) // 2)); ax.imshow(canvas)      # letterboxed: every cell has the same size
        except Exception:
            ax.text(0.5, 0.5, "image unreadable", ha="center", va="center")
        for sp in ax.spines.values():
            sp.set_edgecolor(col[e["cat"]]); sp.set_linewidth(3.2)
        ax.set_xticks([]); ax.set_yticks([])
        tw = textwrap.shorten(re.sub(r"\s+", " ", html.unescape(e["tweet"])), 100, placeholder="...")
        ax.set_title(f"{tag}.{sample.index(e) + 1}  row {e['i']}  [{e['cat']}]", fontsize=7.5, loc="left", color="#222222")
        cap = "\n".join(textwrap.wrap(f"true: {e['true']}", 38) + textwrap.wrap(f"pred: {e['pred']} ({e['conf']:.2f})", 38) + textwrap.wrap(tw, 38)[:3])
        ax.text(0.0, -0.04, cap, transform=ax.transAxes, va="top", ha="left", fontsize=6.8, color="#222222")
    for ax in axes.ravel()[len(sample):]:
        ax.axis("off")
    handles = [plt.Line2D([0], [0], color=col[c], lw=4, label=c) for c in CATS]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(f"{tag}: misclassified test examples ({model})", fontsize=9, y=0.965); fig.subplots_adjust(top=0.9, bottom=0.04, hspace=0.80, wspace=0.10)
    FIG.mkdir(parents=True, exist_ok=True); fig.savefig(str(path_stem) + ".png", dpi=300, bbox_inches="tight"); fig.savefig(str(path_stem) + ".pdf", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    md = ["# Day 4 error analysis (12 misclassified test examples per task)\n",
          "Rule fixed before any example was looked at (SELECTION_RULE_DAY4.md rule 22): the population is every test row the model gets wrong; the (true, predicted) pairs are ranked by error count, and from each pair in rank order the first rows by test-row index are taken "
          "(at most 3 per pair; 6 for Task 1, which has only two pairs) until 12 are collected. Each error gets the first matching category: *text-image mismatch* (the row's text label differs from its image label); *label ambiguity* "
          "(the predicted class is the true class's most-confused partner AND the example is near-duplicate-like, i.e. has a near-identical train example with a conflicting label, or generic, i.e. at most 5 words after normalisation); otherwise *model error*. "
          "**The categories come from these written criteria, not from reading each example**; the counts over all errors are given so that the 12 can be judged for representativeness.\n"]
    tex_rows, summary = [], {}
    for task in ("informative", "humanitarian"):
        tag, model, errs, n, labels, partner, kind = analyse(task)
        sample, ranked, pairs, cap = select(errs, len(labels)); cnt = Counter(e["cat"] for e in errs)
        summary[tag] = dict(model=model, n_test=n, n_errors=len(errs), by_category=dict(cnt), cap_per_pair=cap, sample_test_rows=[e["i"] for e in sample], sample_categories=dict(Counter(e["cat"] for e in sample)),
                            ranked_pairs=[[p[0], p[1], pairs[p]] for p in ranked[:10]], near_dup_kind=kind, partners={labels[c]: labels[p] for c, p in partner.items()})
        md.append(f"\n## {tag}: {model}\n\n{len(errs)} errors on {n} test rows ({len(errs) / n:.1%}); category counts over ALL errors: " + ", ".join(f"{c} {cnt.get(c, 0)}" for c in CATS)
                  + f". Sample categories: " + ", ".join(f"{c} {summary[tag]['sample_categories'].get(c, 0)}" for c in CATS) + f". Cap per pair: {cap}. Near-duplicate criterion: {kind}.\n\n")
        md.append("| # | test row | category | image (path under data/CrisisMMD_v2.0) | tweet | true | predicted (calibrated p) | text label / image label |\n|---|---|---|---|---|---|---|---|\n")
        for k, e in enumerate(sample, 1):
            tw = re.sub(r"\s+", " ", html.unescape(e["tweet"])).strip(); tw = tw if len(tw) <= 140 else tw[:137] + "..."
            md.append(f"| {tag}.{k} | {e['i']} | {e['cat']} | `{e['image']}` | {tw.replace('|', '/')} | {e['true']} | {e['pred']} ({e['conf']:.2f}) | {e['text_label']} / {e['image_label']} |\n")
            tex_rows.append(f"{tag}.{k} & {e['i']} & {tex(e['cat'])} & \\texttt{{{tex(Path(e['image']).name)}}} & {tex(tw[:90] + ('...' if len(tw) > 90 else ''))} & {tex(e['true'])} & {tex(e['pred'])} ({e['conf']:.2f}) & {tex(e['text_label'])} / {tex(e['image_label'])} \\\\\n")
        contact_sheet(tag, model, sample, FIG / f"contact_{tag}")
    (ROOT / "docs" / "day4_error_analysis.md").write_text("".join(md), encoding="utf-8")
    (ROOT / "docs" / "day4_error_analysis_table.tex").write_text(
        "% generated by ablation/d4_error_analysis.py (rule-based categories; see docs/day4_error_analysis.md)\n\\begin{table*}[t]\n\\centering\n\\scriptsize\n\\caption{Twelve misclassified test examples per task (Task 1: deployed BERTweet; Task 2: L4a seed 42), chosen by a rule fixed in advance "
        "(most frequent confusion pairs first, first rows by index, at most three per pair). Categories come from written criteria, not from reading each example.}\n\\label{tab:errors}\n\\begin{tabular}{llllp{4.4cm}p{2.1cm}p{2.5cm}p{2.3cm}}\n\\toprule\n"
        "\\# & Row & Category & Image & Tweet & True & Predicted ($p$) & Text / image label \\\\\n\\midrule\n" + "".join(tex_rows) + "\\bottomrule\n\\end{tabular}\n\\end{table*}\n", encoding="utf-8")
    (ROOT / "results" / "ablation" / "D4_error_analysis.json").write_text(json.dumps(summary, indent=1))
    for t in summary:
        print(t, summary[t]["n_errors"], summary[t]["by_category"], "sample:", summary[t]["sample_categories"], "cap", summary[t]["cap_per_pair"])
