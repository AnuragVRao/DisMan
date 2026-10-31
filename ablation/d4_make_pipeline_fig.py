"""Day 4 Part C6: updated pipeline figure (PNG at 300 dpi + PDF) with Task 1 = BERTweet (fine-tuned, temperature-calibrated). The TikZ version used in report_v2/main.tex mirrors this layout
(it cannot be compiled in this environment, see the final summary); this matplotlib drawing is the file for the slides."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG = Path(__file__).resolve().parent.parent / "report_v2" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
BLUE, GREEN, ORANGE, GREY = "#0072B2", "#009E73", "#E69F00", "#666666"


def box(ax, x, y, w, h, text, fc="#f2f2f2", ec="#444444", fs=7.4, lw=0.9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color="#111111", linespacing=1.25)


def arrow(ax, p, q, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=1.0, color="#333333", connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=0, shrinkB=0))


fig, ax = plt.subplots(figsize=(7.4, 3.5)); ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
box(ax, 0.1, 1.9, 1.35, 1.2, "tweet text\n+ image", fs=8, bold=True)
# Task 1
box(ax, 2.2, 3.75, 2.9, 1.0, "BERTweet (fine-tuned)\nraw text, weighted CE", fc="#dcebf6", ec=BLUE)
box(ax, 5.7, 3.75, 2.1, 1.0, "temperature\nT = 1.825 (dev)", fc="#ffffff", ec=GREY, fs=7)
box(ax, 8.3, 3.75, 1.6, 1.0, "T1: informative?\n(badge, not a gate)", fc="#cfe8f3", ec=BLUE, fs=6.5)
# Task 2
box(ax, 2.2, 1.95, 2.9, 1.1, "DistilBERT (fine-tuned)\n+ frozen CLIP B/32\ncross-attention fusion", fc="#d7efe6", ec=GREEN)
box(ax, 5.7, 1.95, 2.1, 1.1, "temperature\nT = 1.185 (dev)", fc="#ffffff", ec=GREY, fs=7)
box(ax, 8.3, 1.95, 1.6, 1.1, "T2: 8 humanitarian\nclasses", fc="#cfe8f3", ec=BLUE, fs=6.5)
# Task 3
box(ax, 2.2, 0.1, 2.9, 1.1, "CLIP B/32 image +\nMiniLM text\nfused embedding", fc="#fbeccb", ec=ORANGE)
box(ax, 5.7, 0.65, 2.1, 0.55, "logistic probe", fc="#ffffff", ec=GREY, fs=7)
box(ax, 5.7, 0.05, 2.1, 0.55, "FAISS 5-NN (evidence)", fc="#ffffff", ec=GREY, fs=7)
box(ax, 8.3, 0.1, 1.6, 1.1, "T3: severity\n+ evidence\n+ review flag", fc="#cfe8f3", ec=BLUE, fs=6.5)
for y in (4.25, 2.5, 0.65):
    arrow(ax, (1.45, 2.5), (2.2, y), rad=0.0)
arrow(ax, (5.1, 4.25), (5.7, 4.25)); arrow(ax, (7.8, 4.25), (8.3, 4.25))
arrow(ax, (5.1, 2.5), (5.7, 2.5)); arrow(ax, (7.8, 2.5), (8.3, 2.5))
arrow(ax, (5.1, 0.65), (5.7, 0.92)); arrow(ax, (5.1, 0.65), (5.7, 0.32)); arrow(ax, (7.8, 0.92), (8.3, 0.65)); arrow(ax, (7.8, 0.32), (8.3, 0.65))
ax.add_patch(FancyArrowPatch((9.1, 3.75), (9.1, 3.05), arrowstyle="<|-|>", mutation_scale=8, lw=0.9, color=VERM if (VERM := "#D55E00") else "#333", linestyle=(0, (3, 2))))
ax.text(9.2, 3.4, "disagreement\nflag", fontsize=6.3, color="#D55E00", va="center")
ax.text(5.0, -0.28, "All three tasks run on every request. Task 3's probe predicts; the five retrieved neighbours are evidence, and their agreement with the probe sets the confidence.", ha="center", fontsize=6.6, color="#333333")
fig.savefig(FIG / "fig_c6_pipeline.png", dpi=300, bbox_inches="tight"); fig.savefig(FIG / "fig_c6_pipeline.pdf", bbox_inches="tight"); print("pipeline figure written")
