"""Extended classification metrics beyond macro-F1 (guide feedback after the presentation).

full_metrics(y, pred, labels, score=None) returns, for one model on one test set:
  accuracy, macro/weighted F1, balanced accuracy (= macro recall), MCC, per-class precision/recall/F1/support,
  and, when `score` (an n x K matrix of probabilities or logits) is given, one-vs-rest ROC-AUC and PR-AUC (average precision)
  per class plus their macro means over the classes that have both positives and negatives in `y`.
macro_f1_ci() is a percentile bootstrap CI for one model's macro-F1 (same resampling scheme and seed as bootstrap_utils.boot).
Macro averages are over all K classes (zero_division=0), as in the headline numbers.
"""
import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, matthews_corrcoef, precision_recall_fscore_support,
                             roc_auc_score)

from bootstrap_utils import N_BOOT, RNG_SEED, confusion, macro_f1


def full_metrics(y, pred, labels, score=None):
    k = len(labels); ids = list(range(k))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=ids, zero_division=0)
    out = dict(n=int(len(y)), acc=float((y == pred).mean()), macro_f1=float(f.mean()),
               weighted_f1=float((f * s).sum() / s.sum()), macro_precision=float(p.mean()), macro_recall=float(r.mean()),
               balanced_acc=float(balanced_accuracy_score(y, pred)), mcc=float(matthews_corrcoef(y, pred)),
               per_class={labels[i]: dict(precision=float(p[i]), recall=float(r[i]), f1=float(f[i]), support=int(s[i])) for i in ids})
    if score is not None:
        score = np.asarray(score, dtype=float)
        if (score < 0).any() or not np.allclose(score.sum(1), 1.0, atol=1e-3):   # logits -> softmax
            e = np.exp(score - score.max(1, keepdims=True)); score = e / e.sum(1, keepdims=True)
        roc, ap = [], []
        for i in ids:
            pos = (y == i)
            if pos.sum() == 0 or pos.sum() == len(y):
                roc.append(np.nan); ap.append(np.nan); continue
            roc.append(float(roc_auc_score(pos, score[:, i]))); ap.append(float(average_precision_score(pos, score[:, i])))
            out["per_class"][labels[i]].update(roc_auc=roc[-1], pr_auc=ap[-1])
        out["macro_roc_auc"] = float(np.nanmean(roc)); out["macro_pr_auc"] = float(np.nanmean(ap))
    return out


def macro_f1_ci(y, pred, k, n_boot=N_BOOT, seed=RNG_SEED):
    """Percentile bootstrap 95% CI for macro-F1 over test rows."""
    rng = np.random.default_rng(seed); classes = list(range(k)); n = len(y)
    d = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        d[b] = macro_f1(confusion(y[i], pred[i], k), classes)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(point=macro_f1(confusion(y, pred, k), classes), ci95_low=float(lo), ci95_high=float(hi), n_boot=n_boot)
