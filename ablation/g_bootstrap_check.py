"""G: Step-2 audit of the imbalance-study bootstraps (no new training; reads saved logits and g_imbalance_summary_seed42.json).

Questions: (1) is macro-F1 always computed over the fixed full 8-class label set with zero_division=0, also in resamples where a class has no true
rows? (2) how many resamples have zero 'Missing or found people' rows? (3) where does the gap between the bootstrap mean and the point estimate
come from? (4) do the paired differences / verdicts change when everything is recomputed?
The audit compares the production convention (fixed 8 classes) with a deliberately wrong one ("present classes only": classes with no true rows and
no predictions in a resample are dropped from the average); the wrong convention is used for contrast only and is never reported as a result.
Writes results/ablation/G/g_bootstrap_check_seed42.json.
"""
import json

import numpy as np
from sklearn.metrics import f1_score

from bootstrap_utils import class_stats_vec, macro_f1_vec, mean_ci, paired_diff_ci, resample_idx
from common import RES, DATA, SPLIT_DIR, cc, write_row
from g_analyze_imbalance import ARMS, MISS, N_DEV_BOOT, N_TEST_BOOT, SEEDS, load, verdict

CHECK_N = 300   # resamples cross-checked against sklearn for every arm / seed / backbone


def present_only_vec(f1, sup, npred):
    keep = (sup + npred) > 0
    return (f1 * keep).sum(1) / keep.sum(1)


def main():
    S = {s: cc.load_split("humanitarian", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "dev", "test")}
    labels = sorted(S["train"].label.unique()); l2i = {l: i for i, l in enumerate(labels)}; K = len(labels); ALL = list(range(K)); m = l2i[MISS]
    ydv, yte = (np.array([l2i[l] for l in S[s].label]) for s in ("dev", "test"))
    didx, tidx = resample_idx(len(ydv), N_DEV_BOOT), resample_idx(len(yte), N_TEST_BOOT)
    prev = json.load(open(RES / "G" / "g_imbalance_summary_seed42.json"))
    out = dict(task="T2", n_test=int(len(yte)), n_dev=int(len(ydv)), n_boot_test=N_TEST_BOOT, n_boot_dev=N_DEV_BOOT)

    # (2) resamples without any true row of the missing class (the same index matrices as every reported CI)
    z_t = int((np.sum(yte[tidx] == m, axis=1) == 0).sum()); z_d = int((np.sum(ydv[didx] == m, axis=1) == 0).sum())
    out["zero_missing_resamples"] = dict(test=z_t, test_frac=z_t / N_TEST_BOOT, dev=z_d, dev_frac=z_d / N_DEV_BOOT,
                                         expected_frac_test=float((1 - 3 / 2237) ** 2237), expected_frac_dev=float((1 - 2 / 2237) ** 2237))
    print(f"resamples with zero '{MISS}' true rows: test {z_t}/{N_TEST_BOOT} ({z_t / N_TEST_BOOT:.3%}; expected {out['zero_missing_resamples']['expected_frac_test']:.3%}), "
          f"dev {z_d}/{N_DEV_BOOT} ({z_d / N_DEV_BOOT:.3%}; expected {out['zero_missing_resamples']['expected_frac_dev']:.3%})")

    out["feats"] = {}; changed = []; whatif_changed = []
    for feat in ("b32", "l14"):
        R = {a: {s: load(feat, a, s) for s in SEEDS} for a in ARMS}
        P = {a: {s: R[a][s][2].argmax(1) for s in SEEDS} for a in ARMS}
        # (1) cross-check macro_f1_vec against sklearn's definition on the first CHECK_N resamples
        worst = 0.0
        for a in ARMS:
            for s in SEEDS:
                v = macro_f1_vec(yte, P[a][s], K, tidx[:CHECK_N])
                ref = np.array([f1_score(yte[i], P[a][s][i], labels=ALL, average="macro", zero_division=0) for i in tidx[:CHECK_N]])
                worst = max(worst, float(np.abs(v - ref).max()))
        print(f"\n[{feat}] macro_f1_vec vs sklearn f1_score(labels=all 8, zero_division=0) on {CHECK_N} resamples x {len(ARMS) * len(SEEDS)} runs: max |diff| = {worst:.1e}")
        n_pred_missing = int(sum((P[a][s] == m).sum() for a in ARMS for s in SEEDS))
        print(f"[{feat}] total test predictions of '{MISS}' over all arms/seeds: {n_pred_missing}")
        # (3) gap between bootstrap mean and point estimate, plus what the wrong convention would do
        rows = {}; gaps = []
        for a in ARMS:
            f1s = {s: class_stats_vec(yte, P[a][s], K, tidx) for s in SEEDS}
            point = float(np.mean([f1_score(yte, P[a][s], labels=ALL, average="macro", zero_division=0) for s in SEEDS]))
            boot = np.mean([f1s[s][0].mean(1) for s in SEEDS], axis=0)
            wrong = np.mean([present_only_vec(*f1s[s]) for s in SEEDS], axis=0)
            has_m = np.mean([f1s[s][1][:, m] for s in SEEDS], axis=0) > 0
            gaps.append(float(boot.mean() - point))
            rows[a] = dict(point=point, boot_mean=float(boot.mean()), gap=float(boot.mean() - point),
                           boot_mean_resamples_with_missing=float(boot[has_m].mean()), boot_mean_resamples_without_missing=float(boot[~has_m].mean()),
                           wrong_present_only_boot_mean=float(wrong.mean()), wrong_minus_correct=float(wrong.mean() - boot.mean()))
        pc_point = class_stats_vec(yte, P["none"][42], K, np.arange(len(yte))[None, :])[0][0]
        pc_boot = class_stats_vec(yte, P["none"][42], K, tidx)[0].mean(0)
        print(f"[{feat}] bootstrap mean - point estimate (seed-averaged macro-F1): min {min(gaps):+.4f} max {max(gaps):+.4f}  "
              f"(resamples without missing rows change the mean by {np.mean([r['boot_mean_resamples_without_missing'] - r['boot_mean_resamples_with_missing'] for r in rows.values()]):+.4f} on average; "
              f"the missing class has F1 0 in every arm, so dropping or keeping it cannot change the fixed-8 value)")
        print(f"[{feat}] per-class bootstrap-mean minus point F1 for arm 'none' seed 42: " + ", ".join(f"{labels[i][:12]} {pc_boot[i] - pc_point[i]:+.4f}" for i in ALL))
        print(f"[{feat}] contrast: the WRONG 'present classes only' convention would raise the bootstrap-mean macro-F1 by {np.mean([r['wrong_minus_correct'] for r in rows.values()]):+.4f} on average "
              f"(max {max(r['wrong_minus_correct'] for r in rows.values()):+.4f}); the production code does not do this")
        # (4) paired differences: before (reported earlier) vs recomputed now, and the wrong-convention verdicts
        tv = {a: {s: macro_f1_vec(yte, P[a][s], K, tidx) for s in SEEDS} for a in ARMS}
        wv = {a: {s: present_only_vec(*class_stats_vec(yte, P[a][s], K, tidx)) for s in SEEDS} for a in ARMS}
        print(f"[{feat}] paired test differences, BEFORE (as reported) vs AFTER (recomputed)   [reporting only]")
        for ref in ("inv", "none"):
            for a in ARMS:
                if a == ref:
                    continue
                before = prev["feats"][feat]["arms"][a][f"test_vs_{ref}"]
                point = float(np.mean([f1_score(yte, P[a][s], labels=ALL, average="macro", zero_division=0) - f1_score(yte, P[ref][s], labels=ALL, average="macro", zero_division=0) for s in SEEDS]))
                after = paired_diff_ci(tv[a], tv[ref], point_diff=point); after["verdict"] = verdict(after)
                wrong = paired_diff_ci(wv[a], wv[ref]); wrong["verdict"] = verdict(wrong)
                dlo, dhi = abs(before["ci95_low"] - after["ci95_low"]), abs(before["ci95_high"] - after["ci95_high"])
                if before["verdict"] != after["verdict"]:
                    changed.append((feat, a, ref, before["verdict"], after["verdict"]))
                if before["verdict"] != wrong["verdict"]:
                    whatif_changed.append((feat, a, ref, before["verdict"], wrong["verdict"]))
                print(f"   {a:13s} vs {ref:4s} before {before['diff']:+.4f} [{before['ci95_low']:+.4f}, {before['ci95_high']:+.4f}] {before['verdict']:>19s} | "
                      f"after {after['diff']:+.4f} [{after['ci95_low']:+.4f}, {after['ci95_high']:+.4f}] {after['verdict']:>19s} | max CI-endpoint change {max(dlo, dhi):.1e}")
                rows[a][f"vs_{ref}"] = dict(before=before, after=after, wrong_convention=wrong)
        out["feats"][feat] = dict(sklearn_crosscheck_max_abs_diff=worst, missing_predicted_total=n_pred_missing, arms=rows)
    out["verdict_changes_after_recompute"] = changed; out["verdict_changes_under_wrong_convention"] = whatif_changed
    print(f"\nverdict changes after recomputing with the verified convention: {len(changed)} {changed}")
    print(f"(contrast) verdict changes the wrong convention would have caused: {len(whatif_changed)} {whatif_changed}")
    write_row("G", "g_bootstrap_check", 42, out)


if __name__ == "__main__":
    main()
