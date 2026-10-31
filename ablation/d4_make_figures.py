"""Day 4 final round, Part C: figures for report_v2 and the slides (PDF for LaTeX + PNG at 300 dpi) into report_v2/figures/.
C1 T1 confusion matrix (deployed BERTweet), C2 T2 row-normalised 8x8 (L4a seed 42), C3 agreed/disagreed bars for BOTH tasks (the headline finding), C4 T1 model comparison, C5 T2 model comparison,
C8 fusion gain vs own components (both tasks). C6 (pipeline diagram) is d4_make_pipeline_fig.py; C7 (contact sheets) is d4_error_analysis.py.
Style: one colourblind-safe palette (Okabe-Ito; validated with the dataviz skill's validator: text blue, image orange, fusion green, deployed vermillion), thin marks, direct value labels, no dual axes.
All numbers are read from the stored run JSONs / D4_*.json; error bars are the sample std over seeds."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import verify_day4 as V
from common import DATA, ROOT, SPLIT_DIR, cc

RES, FIG = ROOT / "results" / "ablation", ROOT / "report_v2" / "figures"
BLUE, ORANGE, GREEN, VERM, GREY, PURPLE = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#6b6b6b", "#CC79A7"
FAM = {"text": BLUE, "image": ORANGE, "fusion": GREEN, "base": "#9a9a9a"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
                     "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5, "figure.dpi": 100})
J = lambda n: json.load(open(RES / n))


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight"); fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def heat(ax, M, fmt, labels, title, vmax=None, rows_n=None):
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=vmax if vmax else M.max())
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels))); ax.set_xticklabels(labels, rotation=40, ha="right"); ax.set_yticklabels(labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]; ax.text(j, i, fmt(v), ha="center", va="center", fontsize=6.5 if M.shape[0] > 2 else 9, color="white" if v > 0.55 * im.norm.vmax else "#222222")
    for s in ax.spines.values():
        s.set_visible(False)


def c1():
    r = V.run("T1", "bertweet_deploy", 42)["raw"]; cm = np.array(r["test"]["confusion"]); rn = np.array(r["test"]["confusion_row_norm"]); lab = ["informative", "not informative"]
    fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.6))
    heat(ax[0], cm.astype(float), lambda v: f"{int(v)}", lab, "counts"); heat(ax[1], rn, lambda v: f"{100 * v:.1f}%", lab, "row-normalised", vmax=1.0)
    fig.suptitle("Task 1: deployed BERTweet, test set (n = 2,237)", fontsize=8.5, y=1.02); fig.tight_layout(); save(fig, "fig_c1_t1_confusion")


def c2():
    r = V.run("T2", "t2_L4a", 42)["raw"]; rn = np.array(r["test"]["confusion_row_norm"]); cm = np.array(r["test"]["confusion"]); labs = r["labels"]
    short = {"Affected individuals": "Affected", "Infrastructure and utility damage": "Infrastructure", "Injured or dead people": "Injured/dead", "Missing or found people": "Missing/found", "Not humanitarian": "Not humanit.",
             "Other relevant information": "Other relevant", "Rescue, volunteering or donation effort": "Rescue/donation", "Vehicle damage": "Vehicle"}
    names = [f"{short[l]} ({cm[i].sum()})" for i, l in enumerate(labs)]
    fig, ax = plt.subplots(figsize=(4.4, 3.9)); heat(ax, rn, lambda v: f"{100 * v:.0f}", names, "Task 2: L4a seed 42, row-normalised (% of each true class; n in brackets)", vmax=1.0)
    ax.set_xticklabels([short[l] for l in labs], rotation=40, ha="right"); fig.tight_layout(); save(fig, "fig_c2_t2_confusion")


def agree_masks():
    out = {}
    t1 = cc.load_split("informative", "test", DATA, SPLIT_DIR, verbose=False)[0]; r1 = pd.read_csv(SPLIT_DIR / "task_informative_text_img_test.tsv", sep="\t", dtype=str).drop_duplicates(subset=["image_id"]).set_index("image_id").loc[t1.image_id]
    out["T1"] = (r1.label_text.str.strip() == r1.label_image.str.strip()).values
    t2 = cc.load_split("humanitarian", "test", DATA, SPLIT_DIR, verbose=False)[0]; r2 = pd.read_csv(SPLIT_DIR / "task_humanitarian_text_img_test.tsv", sep="\t", dtype=str).drop_duplicates(subset=["image_id"]).set_index("image_id").loc[t2.image_id]
    out["T2"] = (r2.label_text.map(cc.t2_normalise).values == r2.label_image.map(cc.t2_normalise).values)
    return out


def c3():
    mk = agree_masks()
    spec = {"T1": dict(metric="wf1", ylabel="weighted F1 (Task 1 headline)", bars=[("text-only (BERTweet)", "bertweet", BLUE), ("image-only (CLIP L/14 head)", "img_clip_l14", ORANGE),
                                                                         ("best fusion on dev (BERTweet + CLIP L/14 concat)", "early_bertweet_clip_l14", GREEN), ("deployed (BERTweet checkpoint)", "bertweet_deploy", VERM)]),
            "T2": dict(metric="mf1", ylabel="macro-F1 over classes present (Task 2 headline)", bars=[("text-only (BERTweet)", "bertweet_text_rule8", BLUE), ("image-only (CLIP L/14 head)", "t2_img_clip_l14_inv", ORANGE),
                                                                                        ("best fusion on dev (late fusion)", "late_bertweet_text_rule8+t2_img_clip_l14_inv", GREEN), ("deployed (L4a)", "t2_L4a", VERM)])}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), gridspec_kw={"wspace": 0.32}); data = {}
    for ax, t in zip(axes, ("T1", "T2")):
        sp = spec[t]; yv = V.run(t, sp["bars"][0][1], 42)["y"]["test"]; w = 0.19
        for k, (lab, run, col) in enumerate(sp["bars"]):
            seeds = V.seeds_of(t, run)
            for g, (gname, m) in enumerate((("agreed", mk[t]), ("disagreed", ~mk[t]))):
                v = np.array([V.subset(yv, V.run(t, run, s)["p"]["test"], m)[sp["metric"]] for s in seeds]); mean, sd = v.mean(), (v.std(ddof=1) if len(v) > 1 else 0)
                x = g + (k - 1.5) * w; ax.bar(x, mean, w * 0.92, color=col, edgecolor="#222222", linewidth=0.5, label=lab if g == 0 else None)
                if sd:
                    ax.errorbar(x, mean, yerr=sd, color="#222222", capsize=1.8, lw=0.7)
                ax.text(x, mean + sd + 0.012, f"{mean:.2f}", ha="center", va="bottom", fontsize=6.3, rotation=90)
                data[(t, run, gname)] = (float(mean), float(sd))
        ax.set_xticks([0, 1]); ax.set_xticklabels([f"agreed (n={int(mk[t].sum()):,})", f"disagreed (n={int((~mk[t]).sum()):,})"]); ax.set_ylim(0, 1.12 if t == "T1" else 1.0); ax.set_ylabel(sp["ylabel"], fontsize=7.5)
        ax.set_title({"T1": "Task 1 (scored against the text label)", "T2": "Task 2 (scored against the image label)"}[t]); ax.grid(axis="y", lw=0.3, color="#dddddd"); ax.set_axisbelow(True)
    for ax in axes:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=1, frameon=False, fontsize=6.6, handlelength=1.2)
    fig.suptitle("Each modality helps where it describes the label being scored (test set)", fontsize=8.6, y=1.0)
    fig.subplots_adjust(bottom=0.27, top=0.9, left=0.08, right=0.99); save(fig, "fig_c3_agreed_disagreed"); json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in data.items()}, open(RES / "D4_fig_c3_data.json", "w"), indent=1)


def family(name):
    n = name.lower()
    if any(k in n for k in ("early", "late", "l4a", "fusion", "core", "post-hoc", "+ clip", "bertweet + ")) and "text-only" not in n:
        return "fusion"
    if any(k in n for k in ("head", "resnet50 head", "dinov2", "clip b/32 head", "clip l/14 head", "image-only")):
        return "image"
    if "majority" in n:
        return "base"
    return "text"


def bars_h(ax, rows, key_dev, key_test, xlabel, base_line, deployed):
    """Dot plot (a truncated axis would make bars misleading): per model a hollow marker = dev and a filled marker = test, joined by a thin line; whiskers = std over seeds; family colour."""
    n = len(rows); y = np.arange(n)[::-1]
    for yi, (lab, fam, dev, test) in zip(y, rows):
        col = FAM[fam]; dep = lab == deployed
        ax.plot([dev[0], test[0]], [yi, yi], color="#9a9a9a", lw=0.8, zorder=1)
        for v, filled in ((dev, False), (test, True)):
            if v[1]:
                ax.errorbar(v[0], yi, xerr=v[1], color="#222222", capsize=1.6, lw=0.7, zorder=2)
            ax.scatter([v[0]], [yi], s=46 if dep else 30, marker="o", facecolor=col if filled else "white", edgecolor=VERM if dep else col, linewidth=1.8 if dep else 1.3, zorder=3)
        hi = max(dev[0] + (dev[1] or 0), test[0] + (test[1] or 0))
        ax.text(hi + 0.008, yi, f"dev {dev[0]:.3f} | test {test[0]:.3f}" if abs(dev[0] - test[0]) > 1e-9 else f"{test[0]:.3f}", va="center", fontsize=6.2, color="#222222")
    ax.set_yticks(y); ax.set_yticklabels([("★ " if r[0] == deployed else "") + r[0] for r in rows], fontsize=7); ax.set_xlabel(xlabel)
    ax.axvline(base_line, color="#222222", ls="--", lw=0.8); ax.text(base_line, n - 0.35, " majority-class baseline (test)", fontsize=6.5, va="bottom")
    ax.grid(axis="x", lw=0.3, color="#dddddd"); ax.set_axisbelow(True); ax.set_ylim(-0.7, n - 0.1)


def c4():
    rows = J("D4_T1_summary.json"); lab = {"majority": "majority class", "tfidf_lr": "TF-IDF + LR (v2)", "tfidf_nb": "NB, unweighted (ref.)", "tfidf_nb_bal": "NB, balanced", "tfidf_cnb": "ComplementNB", "tfidf_rf": "Random Forest",
                                           "tfidf_svm": "Linear SVM", "bilstm_glove": "BiLSTM + GloVe", "bertweet": "BERTweet (weighted CE)", "bertweet_noweight": "BERTweet (unweighted CE)", "img_resnet50": "ResNet50 head (image)",
                                           "img_dinov2": "DINOv2 head (image)", "img_clip_l14": "CLIP L/14 head (image)", "late_bertweet+img_clip_l14": "late fusion BERTweet + CLIP", "early_bertweet_resnet50": "core: BERTweet + ResNet50",
                                           "early_bertweet_clip_l14": "BERTweet + CLIP L/14"}
    fam = {"majority": "base", "img_resnet50": "image", "img_dinov2": "image", "img_clip_l14": "image", "late_bertweet+img_clip_l14": "fusion", "early_bertweet_resnet50": "fusion", "early_bertweet_clip_l14": "fusion"}
    data = [(lab[r["model"]], fam.get(r["model"], "text"), r["dev_weighted_f1"], r["test_weighted_f1"]) for r in sorted(rows, key=lambda r: r["dev_weighted_f1"][0])]
    maj = [r for r in rows if r["model"] == "majority"][0]["test_weighted_f1"][0]
    fig, ax = plt.subplots(figsize=(5.2, 5.0)); bars_h(ax, data, "dev", "test", "weighted F1 (hollow = dev, filled = test; mean ± std over seeds)", maj, "BERTweet (weighted CE)"); ax.set_xlim(0.55, 0.97)
    ax.set_title("Task 1 models, sorted by dev weighted F1 (★ = deployed model family)"); handles = [plt.Rectangle((0, 0), 1, 1, color=FAM[k], ec="#222222", lw=0.4) for k in ("text", "image", "fusion", "base")]
    ax.legend(handles, ["text model", "image head", "fusion", "baseline"], loc="upper right", frameon=False, ncol=2, fontsize=6.8); fig.tight_layout(); save(fig, "fig_c4_t1_models")


def c5():
    t2 = J("D4_T2_fusion.json")["models"]; tt = J("D4_T2_text_summary.json"); cl = tt["models"]
    items = [("majority class", "base", cl["t2_majority"]), ("TF-IDF + LR", "text", cl["t2_tfidf_lr"]), ("TF-IDF + LinearSVC", "text", cl["t2_tfidf_svm"]), ("ComplementNB (unweighted)", "text", cl["t2_tfidf_cnb"]),
             ("DistilBERT text-only", "text", t2["t2_distilbert_text"]), ("BERTweet text-only (rule-8)", "text", t2["bertweet_text_rule8"]), ("BERTweet text-only (matched)", "text", t2["bertweet_text_matched"]),
             ("image head, CLIP B/32", "image", t2["t2_img_clip_b32"]), ("image head, CLIP L/14", "image", t2["t2_img_clip_l14_inv"]), ("L4a (deployed)", "fusion", t2["t2_L4a"]),
             ("late fusion BERTweet + CLIP L/14", "fusion", t2["late_bertweet_text_rule8+t2_img_clip_l14_inv"]), ("core: BERTweet + ResNet50", "fusion", t2["t2_early_bertweet_resnet50"]),
             ("post-hoc: BERTweet + CLIP B/32", "fusion", t2["t2_early_bertweet_clip_b32"])]
    data = sorted([(a, b, c["dev_macro_f1"], c["test_macro_f1"]) for a, b, c in items], key=lambda r: r[2][0])
    fig, ax = plt.subplots(figsize=(5.2, 4.6)); bars_h(ax, data, "dev", "test", "macro-F1 (hollow = dev, filled = test; mean ± std over seeds)", cl["t2_majority"]["test_macro_f1"][0], "L4a (deployed)"); ax.set_xlim(0, 0.95)
    ax.set_title("Task 2 models, sorted by dev macro-F1 (★ = deployed model)"); handles = [plt.Rectangle((0, 0), 1, 1, color=FAM[k], ec="#222222", lw=0.4) for k in ("text", "image", "fusion", "base")]
    ax.legend(handles, ["text model", "image head", "fusion", "baseline"], loc="upper right", frameon=False, ncol=2, fontsize=6.8); fig.tight_layout(); save(fig, "fig_c5_t2_models")


def c8():
    g = J("D4_fusion_gain_both.json"); fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, t, ttl, h in zip(axes, ("T1", "T2"), ("Task 1 (headline: weighted F1)", "Task 2 (headline: macro-F1)"), ("weighted_f1", "macro_f1")):
        rows = g[t]["rows"]; x = np.arange(len(rows)); w = 0.36
        for k, (sp, col, hatch) in enumerate((("dev", "#bdbdbd", "////"), ("test", "#3a3a3a", None))):
            m = np.array([r["vs_own"][f"{sp}_{h}"][0] for r in rows]); sd = np.array([r["vs_own"][f"{sp}_{h}"][1] for r in rows])
            ax.bar(x + (k - 0.5) * w, m, w * 0.92, color=col, edgecolor="#222222", linewidth=0.5, hatch=hatch, label=sp); ax.errorbar(x + (k - 0.5) * w, m, yerr=sd, fmt="none", color="#222222", capsize=1.8, lw=0.7)
            for xi, mi, si in zip(x + (k - 0.5) * w, m, sd):
                ax.text(xi, mi + (si + 0.002 if mi >= 0 else -si - 0.002), f"{mi:+.3f}", ha="center", va="bottom" if mi >= 0 else "top", fontsize=6, rotation=90)
        short = {"late fusion: BERTweet + CLIP L/14 head": "late fusion", "core model: BERTweet (fine-tuned) + ResNet50 (frozen)": "core model\n(ResNet50)", "BERTweet (fine-tuned) + CLIP L/14 (frozen)": "BERTweet +\nCLIP L/14",
                 "post-hoc: BERTweet (fine-tuned) + CLIP B/32 (frozen)": "post-hoc\nBERTweet + B/32", "L4a: CLIP B/32 + DistilBERT cross-attention": "L4a"}
        ax.set_xticks(x); ax.set_xticklabels([short.get(r["label"], r["label"]) for r in rows], fontsize=6.8); ax.axhline(0, color="#222222", lw=0.7); ax.set_title(ttl); ax.set_ylabel("gain over the better component model", fontsize=7.5)
        lo = min(0, min(r["vs_own"][f"{sp}_{h}"][0] - r["vs_own"][f"{sp}_{h}"][1] for r in rows for sp in ("dev", "test"))); hi = max(r["vs_own"][f"{sp}_{h}"][0] + r["vs_own"][f"{sp}_{h}"][1] for r in rows for sp in ("dev", "test"))
        ax.set_ylim(lo - 0.022, hi + 0.014); ax.grid(axis="y", lw=0.3, color="#dddddd"); ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="upper left", title="split (dev decides)", title_fontsize=7); fig.suptitle("Fusion gain over own components (fused minus the better of its two single-modality models; mean ± std over 3 seeds)", fontsize=8, y=1.02)
    fig.tight_layout(); save(fig, "fig_c8_fusion_gain")


if __name__ == "__main__":
    for f in (c1, c2, c3, c4, c5, c8):
        f(); print("done", f.__name__, flush=True)
