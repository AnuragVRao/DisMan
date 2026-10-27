"""G: paired bootstraps of balanced accuracy and minority-class recall for none vs inv vs sqrt_inv (T2, image-only head, B/32 and L/14).

Pairs: inv-none, sqrt_inv-none, inv-sqrt_inv. Metrics: balanced accuracy (fixed 8-class label set, recall 0 for a class absent from a resample),
recall of 'Injured or dead people' and of 'Vehicle damage' (resamples with no true row of the class are dropped; the count is reported).
'Missing or found people' has recall 0 in every arm on both backbones (asserted below), so it is not bootstrapped.
DEV versions (2000 resamples; dev has 8 injured / 38 vehicle / 2 missing rows) are the ones the selection rule may use.
TEST versions (10,000 resamples; test has 10 injured / 36 vehicle / 3 missing rows) are REPORTING ONLY and are test-sampling CIs; the seed std
(3 seeds) is reported as a separate quantity. Point estimates are seed means; the CI is for the seed-averaged quantity under row resampling.
Writes results/ablation/G/g_balacc_recall_boot_seed42.json.
"""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, recall_score

from bootstrap_utils import balanced_acc_vec, mean_ci, paired_diff_ci, recall_vec, resample_idx
from common import DATA, SPLIT_DIR, cc, write_row
from g_analyze_imbalance import MISS, N_DEV_BOOT, N_TEST_BOOT, SEEDS, load, verdict

ARMS = ("none", "inv", "sqrt_inv")
PAIRS = (("inv", "none"), ("sqrt_inv", "none"), ("inv", "sqrt_inv"))
INJ, VEH = "Injured or dead people", "Vehicle damage"


def main():
    S = {s: cc.load_split("humanitarian", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "dev", "test")}
    labels = sorted(S["train"].label.unique()); l2i = {l: i for i, l in enumerate(labels)}; K = len(labels)
    y = {s: np.array([l2i[l] for l in S[s].label]) for s in ("dev", "test")}
    sup = {s: {c: int((y[s] == l2i[c]).sum()) for c in (INJ, VEH, MISS)} for s in y}
    print("true rows  dev:", sup["dev"], " test:", sup["test"])
    idx = {"dev": resample_idx(len(y["dev"]), N_DEV_BOOT), "test": resample_idx(len(y["test"]), N_TEST_BOOT)}
    out = dict(task="T2", support=sup, n_boot=dict(dev=N_DEV_BOOT, test=N_TEST_BOOT), feats={})
    for feat in ("b32", "l14"):
        R = {a: {s: load(feat, a, s) for s in SEEDS} for a in ARMS}
        out["feats"][feat] = {}
        for split, col, note in (("dev", 1, "DEV (selection-eligible)"), ("test", 2, "TEST - REPORTING ONLY, test-sampling CIs")):
            P = {a: {s: R[a][s][col].argmax(1) for s in SEEDS} for a in ARMS}
            if split == "test":
                assert max(recall_score(y["test"], P[a][s], labels=[l2i[MISS]], average="macro", zero_division=0) for a in ARMS for s in SEEDS) == 0.0
            metrics = {}
            for name, cls in (("balanced_acc", None), ("recall_injured", INJ), ("recall_vehicle", VEH)):
                if cls is None:
                    point = {a: {s: float(balanced_accuracy_score(y[split], P[a][s])) for s in SEEDS} for a in ARMS}
                    vec = {a: {s: balanced_acc_vec(y[split], P[a][s], K, idx[split]) for s in SEEDS} for a in ARMS}; keep = slice(None); dropped = 0
                else:
                    c = l2i[cls]
                    point = {a: {s: float(recall_score(y[split], P[a][s], labels=[c], average="macro", zero_division=0)) for s in SEEDS} for a in ARMS}
                    rv = {a: {s: recall_vec(y[split], P[a][s], K, idx[split]) for s in SEEDS} for a in ARMS}
                    keep = rv["none"][SEEDS[0]][1][:, c] > 0; dropped = int((~keep).sum())   # support depends only on the resample, so one mask serves all arms
                    vec = {a: {s: rv[a][s][0][:, c][keep] for s in SEEDS} for a in ARMS}
                arms = {a: dict(mean=float(np.mean(list(point[a].values()))), seed_std=float(np.std(list(point[a].values()), ddof=1)),
                                resample_ci=mean_ci(vec[a], point=float(np.mean(list(point[a].values()))))) for a in ARMS}
                pairs = {}
                for a, b in PAIRS:
                    pd_ = float(np.mean([point[a][s] - point[b][s] for s in SEEDS]))
                    ci = paired_diff_ci(vec[a], vec[b], point_diff=pd_); ci["verdict"] = verdict(ci); pairs[f"{a}-{b}"] = ci
                metrics[name] = dict(arms=arms, pairs=pairs, resamples_dropped_no_true_rows=dropped)
                print(f"\n[{feat}] {note}: {name}  (resamples dropped for zero true rows: {dropped})")
                for a in ARMS:
                    m = arms[a]
                    print(f"   {a:9s} {m['mean']:.3f}  seed std {m['seed_std']:.3f}  resample 95% CI [{m['resample_ci']['ci95_low']:.3f}, {m['resample_ci']['ci95_high']:.3f}]")
                for k, ci in pairs.items():
                    print(f"   {k:14s} diff {ci['diff']:+.3f} [{ci['ci95_low']:+.3f}, {ci['ci95_high']:+.3f}]  {ci['verdict']}")
            out["feats"][feat][split] = dict(label=note, metrics=metrics)
    write_row("G", "g_balacc_recall_boot", 42, out)


if __name__ == "__main__":
    main()
