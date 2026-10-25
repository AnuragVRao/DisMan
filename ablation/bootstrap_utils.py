"""Shared helpers for paired bootstrap comparisons (same definitions as e1_bootstrap.py; that script is left unchanged)."""
import numpy as np

from common import write_row

N_BOOT, RNG_SEED = 10_000, 20241025


def confusion(yt, yp, k):
    return np.bincount(yt * k + yp, minlength=k * k).reshape(k, k)


def macro_f1(cm, classes):
    f = []
    for c in classes:
        tp = cm[c, c]; fn = cm[c].sum() - tp; fp = cm[:, c].sum() - tp
        f.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0)
    return float(np.mean(f))


def boot(name, yt, pa, pb, k, classes, metric, mask=None, extra=None, write=True):
    """Paired percentile bootstrap of metric(pa) - metric(pb) over test rows (restricted to `mask` if given); 10,000 resamples."""
    if mask is not None:
        yt, pa, pb = yt[mask], pa[mask], pb[mask]
    n = len(yt)
    fn = (lambda y, p: macro_f1(confusion(y, p, k), classes)) if metric == "macro_f1" else (lambda y, p: float((y == p).mean()))
    point = fn(yt, pa) - fn(yt, pb)
    rng = np.random.default_rng(RNG_SEED)
    d = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, n, n)
        d[b] = fn(yt[i], pa[i]) - fn(yt[i], pb[i])
    lo, hi = np.percentile(d, [2.5, 97.5])
    row = dict(task=name.split("_")[0].upper(), tier="cpu", comparison=name, metric=metric, n_rows=int(n), n_boot=N_BOOT, rng_seed=RNG_SEED,
               score_a=fn(yt, pa), score_b=fn(yt, pb), diff=point, ci95_low=float(lo), ci95_high=float(hi),
               ci_excludes_zero=bool(lo > 0 or hi < 0), boot_mean=float(d.mean()), frac_boot_diff_positive=float((d > 0).mean()), **(extra or {}))
    if write:
        write_row("X", f"e1_{name}", 42, row)
    print(f"{name:34s} n={n:4d} {metric:8s} a={row['score_a']:.4f} b={row['score_b']:.4f} diff {point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"excludes 0: {row['ci_excludes_zero']}", flush=True)
    return row
