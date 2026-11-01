"""Day 5: summary of the preprocessing A/B (Task 1 BERTweet). Reads the per-run JSONs, writes results/ablation/D5_prep/summary.{json,md}.
Baseline = D4 raw-text runs (seeds 42-44) + D5 raw-text runs (seeds 45-49, same recipe). A and B have seeds 42-44, C has 42-49."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

R = Path(__file__).resolve().parent.parent / "results" / "ablation"
def load(d, n, s): return json.loads((R / d / f"T1_{n}_seed{s}.json").read_text())
S3, S8 = (42, 43, 44), tuple(range(42, 50))
ARMS = {"baseline": [load("D4", "bertweet", s) for s in S3] + [load("D5_prep", "bertweet_raw", s) for s in range(45, 50)],
        "A": [load("D5_prep", "bertweet_shek_norm", s) for s in S3], "B": [load("D5_prep", "bertweet_shek_nonorm", s) for s in S3],
        "C": [load("D5_prep", "bertweet_prepC", s) for s in S8]}
NAMES = {"baseline": "Baseline: raw tweet_text, BERTweet normalization on", "A": "A: preprocessing/ branch text, normalization on", "B": "B: preprocessing/ branch text, normalization off", "C": "C: minimal cleaning (ftfy + html.unescape + truncated-tail drop), normalization on"}
M = ("weighted_f1", "macro_f1", "acc")
def ms(rs, m, split="test"): v = np.array([r[split][m] for r in rs]); return float(v.mean()), float(v.std(ddof=1))
def boot(rs, base, n=1000):
    y = np.array(base[0]["y_dev"]); rng = np.random.default_rng(0)
    pa = np.mean([np.array(r["probs_dev"]) for r in rs], 0).argmax(1); pb = np.mean([np.array(r["probs_dev"]) for r in base], 0).argmax(1)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y)); d.append(f1_score(y[i], pa[i], average="weighted") - f1_score(y[i], pb[i], average="weighted"))
    return [float(np.mean(d)), *map(float, np.percentile(d, [2.5, 97.5]))]
out = {}
for k, rs in ARMS.items():
    o = dict(name=NAMES[k], seeds=[r["seed"] for r in rs], n_seeds=len(rs), test={m: ms(rs, m) for m in M}, dev_weighted_f1=ms(rs, "weighted_f1", "dev"),
             test_weighted_f1_per_seed=[r["test"]["weighted_f1"] for r in rs])
    if k != "baseline":
        base = ARMS["baseline"]; sh = [r["seed"] for r in rs]; bsame = [r for r in base if r["seed"] in sh]
        o["vs_baseline_same_seeds"] = dict(test_diff={m: float(np.mean([a["test"][m] - b["test"][m] for a, b in zip(rs, bsame)])) for m in M}, dev_bootstrap_weighted_f1_mean_ci95=boot(rs, bsame))
        if k == "C":
            d = np.array([a["test"]["weighted_f1"] - b["test"]["weighted_f1"] for a, b in zip(rs, bsame)])
            o["vs_baseline_same_seeds"]["paired_test_weighted_f1"] = dict(mean_diff=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))), seeds_better=int((d > 0).sum()), seeds_worse=int((d < 0).sum()))
    out[k] = o
(R / "D5_prep" / "summary.json").write_text(json.dumps(out, indent=1))
f = lambda t: f"{t[0]:.4f} ± {t[1]:.4f}"
L = ["# Day 5: preprocessing A/B on Task 1 BERTweet (test split, mean ± std over seeds)", "",
     "Recipe for every arm: vinai/bertweet-base, max_len 128, lr 2e-5, 4 epochs, batch 32, weighted CE, epoch chosen on dev weighted F1, test scored once. Only the input text differs.", "",
     "| Arm | Seeds | Weighted F1 | Macro-F1 | Accuracy | Dev weighted F1 |", "|---|---|---|---|---|---|"]
for k, o in out.items(): L.append(f"| {o['name']} | {o['n_seeds']} | {f(o['test']['weighted_f1'])} | {f(o['test']['macro_f1'])} | {f(o['test']['acc'])} | {f(o['dev_weighted_f1'])} |")
L += ["", "## Difference from the baseline on the same seeds", "", "| Arm | Test Δ weighted F1 | Test Δ macro-F1 | Test Δ accuracy | Dev bootstrap Δ weighted F1 (mean [95% CI]) |", "|---|---|---|---|---|"]
for k in "ABC":
    v = out[k]["vs_baseline_same_seeds"]; b = v["dev_bootstrap_weighted_f1_mean_ci95"]
    L.append(f"| {k} | {v['test_diff']['weighted_f1']:+.4f} | {v['test_diff']['macro_f1']:+.4f} | {v['test_diff']['acc']:+.4f} | {b[0]:+.4f} [{b[1]:+.4f}, {b[2]:+.4f}] |")
p = out["C"]["vs_baseline_same_seeds"]["paired_test_weighted_f1"]
L += ["", f"Arm C paired over 8 seeds: Δ weighted F1 {p['mean_diff']:+.4f}, SE {p['se']:.4f}, better on {p['seeds_better']} seeds, worse on {p['seeds_worse']}.", "",
      "Pre-registered rule for 'helps': mean test weighted-F1 gain > 0.004 AND the dev bootstrap 95% CI excludes 0. No arm meets it.", "", "## Test weighted F1 per seed", "", "| Arm | " + " | ".join(f"s{s}" for s in S8) + " |", "|---|" + "---|" * len(S8)]
for k, o in out.items(): L.append(f"| {k} | " + " | ".join(f"{x:.4f}" for x in o["test_weighted_f1_per_seed"]) + " |" + (" |" * (len(S8) - len(o["seeds"]))))
(R / "D5_prep" / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
