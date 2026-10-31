"""Day 4: the pre-registered Task 1 switch rule (SELECTION_RULE_DAY4.md rule 3), evaluated on DEV only.
Gain = dev weighted F1 of BERTweet (weighted-CE headline arm) minus dev weighted F1 of TF-IDF+LR. One resample matrix (10,000 dev rows resamples, seed 20241025) is used for
every model and seed (paired); the gain is computed per seed and averaged over the 3 seeds per resample. Switch iff the 95% percentile CI of the seed-averaged gain excludes 0
AND the point gain is positive at every seed. Test is not read."""
import json

import numpy as np

import d4_prov
import metrics_all as M
from bootstrap_utils import N_BOOT, RNG_SEED, resample_idx

K = 2


def wf1(y, p, idx):
    """Weighted F1 for every resample at once (idx: n_boot x n)."""
    yt, yp = y[idx], p[idx]; out = 0.0
    n = idx.shape[1]
    for c in range(K):
        tp = ((yt == c) & (yp == c)).sum(1); sup = (yt == c).sum(1); npred = (yp == c).sum(1)
        f = np.where(sup + npred > 0, 2 * tp / np.maximum(sup + npred, 1), 0.0)
        out = out + f * sup / n
    return out


if __name__ == "__main__":
    base = M.load_run("T1", "tfidf_lr", 42); d4_prov.assert_run(base, "informative", "tfidf_lr"); y = base["y_dev"]
    idx = resample_idx(len(y), N_BOOT)
    f_base = wf1(y, base["probs_dev"].argmax(1), idx)
    point_base = float(M.evaluate(y, base["probs_dev"], base["labels"])["weighted_f1"])
    gains, points = [], []
    for seed in (42, 43, 44):
        r = M.load_run("T1", "bertweet", seed); d4_prov.assert_run(r, "informative", f"bertweet seed{seed}"); assert (r["y_dev"] == y).all()
        gains.append(wf1(y, r["probs_dev"].argmax(1), idx) - f_base)
        points.append(float(M.evaluate(y, r["probs_dev"], r["labels"])["weighted_f1"]) - point_base)
    avg = np.mean(gains, 0); lo, hi = np.percentile(avg, [2.5, 97.5])
    positive_all = all(p > 0 for p in points)
    switch = bool(lo > 0 and positive_all)
    res = dict(n_boot=N_BOOT, rng_seed=RNG_SEED, dev_weighted_f1_tfidf_lr=point_base, per_seed_point_gain=points, seed_avg_gain=float(np.mean(points)),
               ci95=[float(lo), float(hi)], ci_excludes_zero=bool(lo > 0 or hi < 0), positive_at_every_seed=positive_all, switch=switch)
    for s, g in zip((42, 43, 44), gains):
        a, b = np.percentile(g, [2.5, 97.5]); res[f"seed{s}_ci95"] = [float(a), float(b)]
    print(json.dumps(res, indent=1))
    out = M.RUN_DIR.parent / "D4_switch_rule_T1.json"; out.write_text(json.dumps(res, indent=1))
    print("DECISION:", "SWITCH the deployed Task 1 model to BERTweet" if switch else "KEEP TF-IDF+LR (justify on latency)")
