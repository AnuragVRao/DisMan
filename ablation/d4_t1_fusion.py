"""Day 4, calibrated late fusion for either task from saved runs (SELECTION_RULE_DAY4.md): each model's dev log-probabilities are temperature-scaled on dev, then the
two probability vectors are averaged. Seed k of the neural model is paired with seed k of the partner; a deterministic partner (TF-IDF, seed 42) is reused for every seed.
Usage: python d4_t1_fusion.py <task T1|T2> <text_run> <image_run> [<text_run> <image_run> ...]"""
import sys

import numpy as np

import metrics_all as M


def load(task, name, seed):
    try:
        return M.load_run(task, name, seed)
    except FileNotFoundError:
        return M.load_run(task, name, 42)


def calibrated(run):
    zd = np.log(np.clip(run["probs_dev"], 1e-9, 1)); zt = np.log(np.clip(run["probs_test"], 1e-9, 1))
    t = M.fit_temperature(zd, run["y_dev"])
    return M.apply_temperature(zd, t), M.apply_temperature(zt, t), t


if __name__ == "__main__":
    task, args = sys.argv[1], sys.argv[2:]
    for a, b in zip(args[::2], args[1::2]):
        for seed in (42, 43, 44):
            ra, rb = load(task, a, seed), load(task, b, seed)
            assert (ra["y_dev"] == rb["y_dev"]).all() and (ra["y_test"] == rb["y_test"]).all()
            da, ta, tempa = calibrated(ra); db, tb, tempb = calibrated(rb)
            row = M.save_run(task, f"late_{a}+{b}", seed, ra["labels"], ra["y_dev"], (da + db) / 2, ra["y_test"], (ta + tb) / 2, extra=dict(temps=[tempa, tempb]))
            t, d = row["test"], row["dev"]
            print(f"late {a}+{b} seed {seed}: T=({tempa:.2f},{tempb:.2f}) dev wF1 {d['weighted_f1']:.4f} macroF1 {d['macro_f1']:.4f} | test wF1 {t['weighted_f1']:.4f} macroF1 {t['macro_f1']:.4f} acc {t['acc']:.4f}", flush=True)
