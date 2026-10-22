"""Figures for RESULTS.md -> results/figures/*.png. Reads results/preds_task*_test.csv (no retraining).

Style follows the dataviz skill: single-hue sequential ramp (blue, light->dark) for confusion matrices, one series colour for
single-series bars, text in ink tokens (never the series colour), recessive grid, direct value labels.
"""
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

RES, OUT = Path("results"), Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)
SURFACE, INK, INK2, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de", "#2a78d6"
RAMP = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "font.size": 10, "axes.titleweight": "bold", "axes.titlesize": 12})

TASKS = {
    "task1": ("Task 1: Informative vs not informative", ["informative", "not_informative"]),
    "task2": ("Task 2: Humanitarian categories", None),
    "task3": ("Task 3: Damage severity (tweet-disjoint index)", ["Severe damage", "Mild damage", "Little or no damage"]),
}


def load(key):
    d = pd.read_csv(RES / f"preds_{key}_test.csv")
    true, pred = ("label", "pred") if key == "task1" else (("true", "pred") if "true" in d else ("label", "pred"))
    return d[true].astype(str), d[pred].astype(str)


def confusion_fig(key, title, labels):
    y, p = load(key)
    labels = labels or sorted(set(y) | set(p))
    cm = confusion_matrix(y, p, labels=labels)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(5.6, 1.15 * n + 3), max(4.6, 0.85 * n + 2.4)))
    ax.imshow(norm, cmap=RAMP, vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8.5,
                    color="#fcfcfb" if norm[i, j] > 0.5 else INK)
    ax.set_xticks(range(n), [textwrap.fill(l, 13) for l in labels], fontsize=8.5)
    ax.set_yticks(range(n), [textwrap.fill(l, 24) for l in labels], fontsize=8.5)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True (row-normalised; counts above %)")
    ax.set_title(f"{title}\nTest set, n = {len(y)}", loc="left")
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout(); fig.savefig(OUT / f"confusion_{key}.png", dpi=160); plt.close(fig)


def f1_fig(key, title, labels):
    y, p = load(key)
    labels = labels or sorted(set(y) | set(p))
    _, _, f1, sup = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    order = np.argsort(f1)
    fig, ax = plt.subplots(figsize=(8, max(2.6, 0.55 * len(labels) + 1.4)))
    ypos = np.arange(len(labels))
    ax.barh(ypos, f1[order], height=0.55, color=SERIES)
    ax.set_yticks(ypos, [f"{labels[i]}  (n={sup[i]})" for i in order])
    for yp, v in zip(ypos, f1[order]):
        ax.text(v + 0.012, yp, f"{v:.2f}", va="center", color=INK, fontsize=9)
    ax.set_xlim(0, 1.08); ax.set_xlabel("F1 (test)")
    ax.axvline(f1.mean(), color=INK2, lw=1, ls=(0, (4, 3)))
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"{title}\nPer-class F1 on test; dashed line = macro-F1 {f1.mean():.3f}", loc="left")
    fig.tight_layout(); fig.savefig(OUT / f"f1_{key}.png", dpi=160); plt.close(fig)


def architecture():
    fig, ax = plt.subplots(figsize=(13, 7.2)); ax.set_xlim(0, 13); ax.set_ylim(0, 7.2); ax.axis("off")

    def box(x, y, w, h, text, fc="#f0efec", ec=GRID, bold=False, tc=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.2, color=tc,
                fontweight="bold" if bold else "normal", linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13, color=INK2, lw=1.2))

    ax.text(0.1, 6.95, "Disaster-triage cascade: three independent stages over one tweet (image + text)", fontsize=13,
            fontweight="bold", va="center")
    box(0.1, 3.05, 1.9, 1.1, "Tweet text\n(cleaned)", bold=True)
    box(0.1, 1.15, 1.9, 1.1, "Tweet image", bold=True)

    # row 1: Task 1
    box(3.0, 5.55, 2.6, 0.95, "TF-IDF (1-2 grams)\n+ Logistic Regression")
    box(6.6, 5.55, 2.4, 0.95, "Informative?\nbadge (not a gate)", fc="#cde2fb", ec="#86b6ef")
    ax.text(3.0, 6.62, "Task 1: text only", fontsize=9.5, color=INK2, fontweight="bold")
    # row 2: Task 2
    ax.text(3.0, 4.75, "Task 2: humanitarian (8 classes)", fontsize=9.5, color=INK2, fontweight="bold")
    box(3.0, 3.55, 2.6, 0.95, "DistilBERT\n(fine-tuned)")
    box(3.0, 2.35, 2.6, 0.95, "CLIP ViT-B/32 image encoder\n(frozen, 512-d, QuickGELU)")
    box(6.3, 2.95, 1.7, 1.25, "Cross-attention\n+ MLP\n(1 image key)")
    box(8.5, 2.95, 1.9, 1.25, "Humanitarian\ncategory\n+ probabilities", fc="#cde2fb", ec="#86b6ef")
    # row 3: Task 3
    ax.text(3.0, 1.95, "Task 3: damage severity (3 classes)", fontsize=9.5, color=INK2, fontweight="bold")
    box(3.0, 0.65, 2.6, 0.95, "SBERT MiniLM (384-d)\n+ same CLIP vector (512-d)")
    box(6.3, 0.65, 1.7, 0.95, "concat + L2\nFAISS inner product")
    box(8.5, 0.65, 1.9, 0.95, "5-NN vote\n(train-only index)", fc="#cde2fb", ec="#86b6ef")
    box(10.9, 0.65, 2.0, 0.95, "Severity + top-5\nsimilar past cases", fc="#cde2fb", ec="#86b6ef")
    ax.text(10.9, 1.7, "vote share, not a calibrated\nprobability", fontsize=8.5, color=INK2, va="bottom")

    arrow(2.0, 3.75, 3.0, 6.0); arrow(5.6, 6.02, 6.6, 6.02)
    arrow(2.0, 3.6, 3.0, 4.0); arrow(2.0, 1.7, 3.0, 2.8)
    arrow(5.6, 4.0, 6.3, 3.7); arrow(5.6, 2.85, 6.3, 3.3); arrow(8.0, 3.57, 8.5, 3.57)
    ax.plot([0.5, 0.5, 2.55], [3.05, 0.9, 0.9], color=INK2, lw=1.2); arrow(2.55, 0.9, 3.0, 0.9)
    arrow(2.0, 1.55, 3.0, 1.3)
    arrow(5.6, 1.12, 6.3, 1.12); arrow(8.0, 1.12, 8.5, 1.12); arrow(10.4, 1.12, 10.9, 1.12)
    ax.text(0.1, 0.2, "If Task 1 and Task 2 disagree both are shown: Task 1 is trained on tweet-text labels, Task 2 on image labels.",
            fontsize=9, color=INK2)
    fig.savefig(OUT / "architecture.png", dpi=160, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    for k, (t, labs) in TASKS.items():
        confusion_fig(k, t, labs); f1_fig(k, t, labs)
    architecture()
    print(sorted(p.name for p in OUT.iterdir()))
