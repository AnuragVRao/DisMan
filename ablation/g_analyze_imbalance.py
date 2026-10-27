"""G: analysis of the T2 imbalance study (image-only head on cached CLIP features; B/32 and L/14).

Step 1  full table per backbone from the result JSONs / saved logits (all arms)
Step 3  test-set bootstraps (REPORTING only): CI on seed-averaged test macro-F1 and paired differences vs `inv` and `none`
Step 4  the selection rule in ablation/SELECTION_RULE.md applied mechanically on DEV (2000 paired resamples, seed-averaged difference)
Also checks the specific claims listed in the study brief against the data. Writes results/ablation/G/g_imbalance_summary_seed42.json.
Wording rule: "better"/"worse" only where the paired 95% CI excludes 0, otherwise "not distinguishable".
"""
import json

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

from bootstrap_utils import macro_f1_vec, mean_ci, paired_diff_ci, resample_idx
from common import ABL_ART, DATA, RES, SPLIT_DIR, cc, write_row

ARMS = ("none", "inv", "sqrt_inv", "eff_num", "focal1", "focal2", "focal2_inv", "sampler", "logit_adj", "none_posthoc")
SEEDS = (42, 43, 44)
MISS, INJ, VEH, AFF = "Missing or found people", "Injured or dead people", "Vehicle damage", "Affected individuals"
N_DEV_BOOT, N_TEST_BOOT = 2000, 10000
OUT = ABL_ART / "imbalance"


def verdict(ci):
    return "better" if ci["ci95_low"] > 0 else "worse" if ci["ci95_high"] < 0 else "not distinguishable"


def load(feat, arm, seed):
    j = json.load(open(RES / "G" / f"g_t2_{feat}_{arm}_seed{seed}.json"))
    z = np.load(OUT / f"{feat}_{arm}_seed{seed}.npz")
    return j, z["dev"], z["test"]


def main():
    S = {s: cc.load_split("humanitarian", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "dev", "test")}
    labels = sorted(S["train"].label.unique()); l2i = {l: i for i, l in enumerate(labels)}; K = len(labels); ALL = list(range(K))
    ydv, yte = (np.array([l2i[l] for l in S[s].label]) for s in ("dev", "test"))
    n_miss = {s: int((np.array([l2i[l] for l in S[s].label]) == l2i[MISS]).sum()) for s in ("train", "dev", "test")}
    print(f"'{MISS}' rows: train {n_miss['train']}, dev {n_miss['dev']}, test {n_miss['test']}  (dev n={len(ydv)}, test n={len(yte)})")
    didx, tidx = resample_idx(len(ydv), N_DEV_BOOT), resample_idx(len(yte), N_TEST_BOOT)
    summary, chosen = {}, {}
    for feat in ("b32", "l14"):
        R = {a: {s: load(feat, a, s) for s in SEEDS} for a in ARMS}
        st = {}
        for a in ARMS:
            devf1 = [float(f1_score(ydv, R[a][s][1].argmax(1), labels=ALL, average="macro", zero_division=0)) for s in SEEDS]
            assert np.allclose(devf1, [R[a][s][0]["dev_macro_f1"] for s in SEEDS], atol=1e-6), f"{feat} {a}: dev F1 from logits != stored"
            T = [R[a][s][0]["test"] for s in SEEDS]
            pc = lambda c, k: float(np.mean([t["per_class"][c][k] for t in T]))
            st[a] = dict(dev_f1=float(np.mean(devf1)),
                         dev_bal_acc=float(np.mean([balanced_accuracy_score(ydv, R[a][s][1].argmax(1)) for s in SEEDS])),
                         test_f1=float(np.mean([t["macro_f1"] for t in T])), test_f1_seed_std=float(np.std([t["macro_f1"] for t in T], ddof=1)),
                         test_bal_acc=float(np.mean([t["balanced_acc"] for t in T])),
                         test_f1_excl_missing=float(np.mean([np.mean([v["f1"] for c, v in t["per_class"].items() if c != MISS]) for t in T])),
                         recall={c: pc(c, "recall") for c in (MISS, INJ, VEH, AFF)}, f1={c: pc(c, "f1") for c in (MISS, INJ, VEH, AFF)})
        order = sorted(ARMS, key=lambda a: -st[a]["dev_f1"])
        print(f"\n===== {feat.upper()} | Step 1: all arms, sorted by dev macro-F1 (means over seeds; test std is across seeds) =====")
        print(f"{'arm':13s} {'dev F1':>7s} {'test F1':>8s} {'seed std':>8s} {'bal acc':>8s} {'F1 excl.miss':>12s} | recall: {'Missing':>7s} {'Injured':>7s} {'Vehicle':>7s}")
        for a in order:
            s = st[a]
            print(f"{a:13s} {s['dev_f1']:7.3f} {s['test_f1']:8.3f} {s['test_f1_seed_std']:8.3f} {s['test_bal_acc']:8.3f} {s['test_f1_excl_missing']:12.3f} | "
                  f"        {s['recall'][MISS]:7.3f} {s['recall'][INJ]:7.3f} {s['recall'][VEH]:7.3f}")
        print(f"F1 of '{MISS}': " + ", ".join(f"{a} {st[a]['f1'][MISS]:.3f}" for a in order))
        # ---- Step 3: bootstraps ----
        dvec = {a: {s: macro_f1_vec(ydv, R[a][s][1].argmax(1), K, didx) for s in SEEDS} for a in ARMS}
        tvec = {a: {s: macro_f1_vec(yte, R[a][s][2].argmax(1), K, tidx) for s in SEEDS} for a in ARMS}
        print(f"\n--- {feat.upper()} Step 3 (REPORTING ONLY): test-sampling 95% CI of seed-averaged test macro-F1, and paired difference vs inv / vs none ---")
        print(f"{'arm':13s} {'test F1':>8s} {'test-sampling CI':>18s} | {'vs inv: diff [CI]':>30s} {'verdict':>20s} | {'vs none: diff [CI]':>30s} {'verdict':>20s}")
        for a in order:
            c = mean_ci(tvec[a], point=st[a]["test_f1"])      # centre = point estimate (seed mean); the percentile CI is from the resamples
            di = paired_diff_ci(tvec[a], tvec["inv"], point_diff=st[a]["test_f1"] - st["inv"]["test_f1"])
            dn = paired_diff_ci(tvec[a], tvec["none"], point_diff=st[a]["test_f1"] - st["none"]["test_f1"])
            st[a].update(test_ci=c, test_vs_inv=dict(di, verdict=verdict(di)), test_vs_none=dict(dn, verdict=verdict(dn)))
            print(f"{a:13s} {c['mean']:8.3f} [{c['ci95_low']:.3f}, {c['ci95_high']:.3f}] | {di['diff']:+8.3f} [{di['ci95_low']:+.3f}, {di['ci95_high']:+.3f}] {verdict(di):>20s} | "
                  f"{dn['diff']:+8.3f} [{dn['ci95_low']:+.3f}, {dn['ci95_high']:+.3f}] {verdict(dn):>20s}")
        # ---- Step 4: the pre-registered rule, on DEV only ----
        best = max(ARMS, key=lambda a: st[a]["dev_f1"]); tied = [best]
        print(f"\n--- {feat.upper()} Step 4: selection rule on DEV (best arm = {best}, mean dev macro-F1 {st[best]['dev_f1']:.3f}) ---")
        for a in order:
            if a == best:
                continue
            ci = paired_diff_ci(dvec[best], dvec[a], point_diff=st[best]["dev_f1"] - st[a]["dev_f1"]); st[a]["dev_best_minus_arm"] = ci
            t = not (ci["ci95_low"] > 0 or ci["ci95_high"] < 0)
            if t:
                tied.append(a)
            print(f"  {best} - {a:13s} {ci['diff']:+.4f} [{ci['ci95_low']:+.4f}, {ci['ci95_high']:+.4f}]  {'TIED' if t else 'best is better'}")
        pick = max(tied, key=lambda a: st[a]["dev_bal_acc"])
        print(f"  tied set: {tied}\n  dev balanced accuracy in tied set: " + ", ".join(f"{a} {st[a]['dev_bal_acc']:.3f}" for a in tied) + f"\n  -> SELECTED on {feat.upper()}: {pick}")
        if "none_posthoc" in tied or pick == "none_posthoc":
            print("  note: none_posthoc's tau was chosen on dev, so its dev score is optimistically biased (see SELECTION_RULE.md)")
        chosen[feat] = pick
        summary[feat] = dict(arms=st, order=order, dev_best=best, tied=tied, selected=pick)
        # ---- claims check ----
        print(f"\n--- {feat.upper()} claim check ---")
        rank = order.index("inv") + 1; r9 = [a for a in order if a != "none_posthoc"].index("inv") + 1
        print(f"1. 'inv ranks fifth on dev': inv is {rank} of {len(order)} arms on dev ({r9} of 9 excluding none_posthoc)  -> {'holds' if rank == 5 or r9 == 5 else 'CONTRADICTED'}")
        bt = max(ARMS, key=lambda a: st[a]["test_f1"])
        print(f"2. 'eff_num has the best test macro-F1': best test arm is {bt} ({st[bt]['test_f1']:.3f}); eff_num {st['eff_num']['test_f1']:.3f}  -> {'holds' if bt == 'eff_num' else 'CONTRADICTED'}")
        for a in ("sampler", "focal2_inv", "logit_adj"):
            print(f"3. {a} vs none: test diff {st[a]['test_vs_none']['diff']:+.3f} [{st[a]['test_vs_none']['ci95_low']:+.3f}, {st[a]['test_vs_none']['ci95_high']:+.3f}] "
                  f"-> {st[a]['test_vs_none']['verdict']}")
        print(f"4. Affected individuals none -> inv: recall {st['none']['recall'][AFF]:.2f} -> {st['inv']['recall'][AFF]:.2f}, F1 {st['none']['f1'][AFF]:.2f} -> {st['inv']['f1'][AFF]:.2f}")
        print(f"5. none_posthoc vs none test macro-F1: {st['none_posthoc']['test_f1']:.3f} vs {st['none']['test_f1']:.3f} "
              f"(paired diff {st['none_posthoc']['test_vs_none']['diff']:+.3f}, {st['none_posthoc']['test_vs_none']['verdict']})")
    same = len(set(chosen.values())) == 1
    adopt = chosen["b32"] if same else None
    print(f"\n===== Adoption condition: B/32 selects {chosen['b32']}, L/14 selects {chosen['l14']} -> "
          f"{'same arm; ' + ('it is already the headline loss, so no change' if adopt == 'inv' else 'ADOPT ' + str(adopt)) if same else 'different arms: keep inverse-frequency and document the trade-off'} =====")
    write_row("G", "g_imbalance_summary", 42, dict(task="T2", n_boot_dev=N_DEV_BOOT, n_boot_test=N_TEST_BOOT, missing_rows=n_miss, selected=chosen, adopt=adopt, feats=summary))
    print("minority class 'Missing or found people' (9 train / 3 test): F1 = 0 in every arm on both backbones" if all(
        st["f1"][MISS] == 0 for f in summary.values() for st in f["arms"].values()) else "WARNING: some arm has non-zero F1 on the missing class")


if __name__ == "__main__":
    main()
