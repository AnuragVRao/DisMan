"""Day 2: apply the pre-registered selection rule (ablation/SELECTION_RULE.md, Day 2 addendum) to one decision, mechanically.

  python d2_select.py d1|d2|d3|d4

For the decision's arms (3 seeds each, results/ablation/H/): mean dev macro-F1; the best arm; paired dev bootstrap (2000 resamples of the dev rows, seed 20241025, the same rows
for every arm and seed, per-seed differences averaged over seeds, 95% percentile CI) of best - arm; an arm is tied if the CI includes 0; among the tied arms the SIMPLEST is
picked: fewest trainable parameters, then the smaller frozen backbone(s) it consumes (D4: then smaller d_model is already covered by the parameter count, then the configuration
closest to the v2 defaults lr 1e-4 / dropout 0.2 in grid steps, then the lower lr). Highest balanced accuracy is NOT used. Test numbers are not read here.
Writes results/ablation/H/selection_<d>_seed42.json.
"""
import itertools
import json
import sys

import numpy as np

import bootstrap_utils as bu
from common import write_row
from d2_common import RES, SEEDS, data

LR, DO, DM = (3e-5, 1e-4, 3e-4), (0.1, 0.2, 0.4), (128, 256, 512)
ARMS = {"d1": [f"d1_{a}" for a in ("clip_b32", "clip_l14", "dinov2", "resnet50")], "d2": [f"d2_{a}" for a in ("minilm", "mpnet")],
        "d3": [f"d3_{k}" for k in ("image_only", "text_only", "concat", "gated_late", "cross_attention")],
        "d4": [f"d4_lr{lr:g}_do{do:g}_dm{dm}" for lr, do, dm in itertools.product(LR, DO, DM)]}


def main(d):
    D = data(); ydev = D["y"]["dev"].numpy(); K = D["nc"]
    idx = bu.resample_idx(len(ydev), 2000)
    R = {a: [json.load(open(RES / "H" / f"{a}_seed{s}.json")) for s in SEEDS] for a in ARMS[d]}
    dev = {a: np.array([r["dev_macro_f1"] for r in R[a]]) for a in R}
    mean = {a: float(dev[a].mean()) for a in R}
    best = max(R, key=lambda a: mean[a])
    vec = {a: {s: bu.macro_f1_vec(ydev, np.array(r["pred_dev"]), K, idx) for s, r in zip(SEEDS, R[a])} for a in R}
    for a in R:      # the stored dev F1 is the dev-selected checkpoint's; check the stored dev predictions reproduce it
        for r in R[a]:
            from sklearn.metrics import f1_score
            assert abs(f1_score(ydev, r["pred_dev"], labels=list(range(K)), average="macro", zero_division=0) - r["dev_macro_f1"]) < 1e-12
    diff = {a: bu.paired_diff_ci(vec[best], vec[a], point_diff=mean[best] - mean[a]) for a in R if a != best}
    tied = [best] + [a for a in R if a != best and not (diff[a]["ci95_low"] > 0 or diff[a]["ci95_high"] < 0)]

    def key(a):
        r = R[a][0]
        if d == "d4":
            hp = r["hp"]; steps = abs(LR.index(hp["lr"]) - 1) + abs(DO.index(hp["dropout"]) - 1)
            return (r["n_trainable"], steps, hp["lr"])
        return (r["n_trainable"], r["backbone_params_m"])
    pick = min(tied, key=key)
    print(f"\n=== {d.upper()}: {len(R)} arms; best by mean dev macro-F1: {best} ({mean[best]:.4f}); {len(tied)} tied (the best and {len(tied) - 1} others) ===")
    print(f"{'arm':36s} {'dev F1 (mean ± std)':>20s} {'params':>10s} {'backbone M':>10s} {'best - arm [95% CI]':>34s}  tied")
    for a in sorted(R, key=lambda a: -mean[a]):
        r = R[a][0]
        c = "best" if a == best else f"{diff[a]['diff']:+.4f} [{diff[a]['ci95_low']:+.4f}, {diff[a]['ci95_high']:+.4f}]"
        print(f"{a:36s} {mean[a]:.4f} ± {dev[a].std(ddof=1):.4f} {r['n_trainable']:10,d} {r['backbone_params_m']:10.1f} {c:>34s}  {'yes' if a in tied else 'no'}{'   <- PICK' if a == pick else ''}")
    print(f"tied set by complexity key: {[ (a, key(a)) for a in sorted(tied, key=key)]}")
    print(f"RULE'S PICK for {d.upper()}: {pick}")
    write_row("H", f"selection_{d}", 42, dict(decision=d.upper(), best=best, tied=tied, pick=pick, mean_dev_macro_f1=mean, seed_std_dev=dict((a, float(dev[a].std(ddof=1))) for a in R),
                                               best_minus_arm=diff, key={a: list(key(a)) for a in R}, n_boot=2000))


if __name__ == "__main__":
    main(sys.argv[1])
