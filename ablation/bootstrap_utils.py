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


def resample_idx(n, n_boot, seed=RNG_SEED):
    """n_boot x n matrix of resampled row indices; draw it once and reuse it for every arm and seed so comparisons are paired."""
    return np.random.default_rng(seed).integers(0, n, (n_boot, n))


def _cm_chunks(yt, yp, k, idx, chunk):
    for s in range(0, len(idx), chunk):
        ii = idx[s:s + chunk]; b = len(ii)
        flat = (yt[ii] * k + yp[ii]) + np.arange(b)[:, None] * k * k
        yield s, b, np.bincount(flat.ravel(), minlength=b * k * k).reshape(b, k, k)


def class_stats_vec(yt, yp, k, idx, chunk=500):
    """Per-resample, per-class F1, true-row count and predicted-row count: three (n_boot, k) arrays.
    F1 of a class with no true rows and no predictions in a resample is 0 (zero_division=0), never skipped."""
    f1 = np.empty((len(idx), k)); sup = np.empty((len(idx), k), dtype=np.int64); npred = np.empty((len(idx), k), dtype=np.int64)
    for s, b, cm in _cm_chunks(yt, yp, k, idx, chunk):
        tp = np.einsum("bii->bi", cm).astype(float)
        sup[s:s + b], npred[s:s + b] = cm.sum(2), cm.sum(1)
        den = 2 * tp + (sup[s:s + b] - tp) + (npred[s:s + b] - tp)
        f1[s:s + b] = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)
    return f1, sup, npred


def macro_f1_vec(yt, yp, k, idx, chunk=500):
    """Macro-F1 over ALL k classes (0 for a class with no support and no predictions) for every resample in idx; same definition as macro_f1()."""
    return class_stats_vec(yt, yp, k, idx, chunk)[0].mean(1)


def recall_vec(yt, yp, k, idx, chunk=500):
    """Per-resample, per-class recall tp/support (NaN where the class has no true rows in the resample) and the support: two (n_boot, k) arrays."""
    rec = np.empty((len(idx), k)); sup = np.empty((len(idx), k), dtype=np.int64)
    for s, b, cm in _cm_chunks(yt, yp, k, idx, chunk):
        tp = np.einsum("bii->bi", cm).astype(float); sup[s:s + b] = cm.sum(2)
        rec[s:s + b] = np.where(sup[s:s + b] > 0, tp / np.maximum(sup[s:s + b], 1), np.nan)
    return rec, sup


def balanced_acc_vec(yt, yp, k, idx, chunk=500):
    """Balanced accuracy over ALL k classes; a class with no true rows in a resample contributes recall 0 (the same fixed-label-set convention as
    macro-F1). On the full test set every class is present, so the point estimate equals sklearn's balanced_accuracy_score."""
    return np.nan_to_num(recall_vec(yt, yp, k, idx, chunk)[0], nan=0.0).mean(1)


def paired_diff_ci(vec_a, vec_b, point_diff=None):
    """vec_a / vec_b: {seed: per-resample metric vector}. Percentile CI of the seed-averaged difference a - b (per resample, per seed, then averaged).
    `diff` is the point estimate when point_diff is given (otherwise the bootstrap mean); `boot_mean_diff` is always the bootstrap mean."""
    d = np.mean([vec_a[s] - vec_b[s] for s in vec_a], axis=0)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(diff=float(d.mean() if point_diff is None else point_diff), boot_mean_diff=float(d.mean()), ci95_low=float(lo), ci95_high=float(hi),
                excludes_zero=bool(lo > 0 or hi < 0), n_boot=int(len(d)))


def mean_ci(vec, point=None):
    """vec: {seed: per-resample vector}. Percentile CI of the seed-averaged metric under row resampling; `mean` is the point estimate when given."""
    m = np.mean([vec[s] for s in vec], axis=0)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return dict(mean=float(m.mean() if point is None else point), boot_mean=float(m.mean()), ci95_low=float(lo), ci95_high=float(hi))


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
