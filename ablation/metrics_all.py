"""Day 4 (SELECTION_RULE_DAY4.md): one evaluation entry point for every model, both tasks.

evaluate(y, probs, labels) -> dict with accuracy, weighted/macro F1, per-class P/R/F1, balanced accuracy, MCC, ROC-AUC, PR-AUC, ECE,
confusion matrix (counts and row-normalised) and, for more than 2 classes only, top-2 accuracy.
`probs` may be probabilities or logits (softmax is applied to anything that is not a probability matrix).
Also: fit_temperature / apply_temperature (dev-fitted, for calibrated late fusion), save_run / load_run (dev AND test probabilities per run),
and fusion_gain.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import confusion_matrix, roc_auc_score

from extra_metrics import full_metrics

RUN_DIR = Path(__file__).resolve().parent.parent / "results" / "ablation" / "D4"


def to_probs(s):
    s = np.asarray(s, dtype=float)
    if s.ndim == 1:                                           # binary positive-class score -> two columns
        s = np.stack([1 - s, s], 1)
    if (s < 0).any() or not np.allclose(s.sum(1), 1.0, atol=1e-3):
        e = np.exp(s - s.max(1, keepdims=True)); s = e / e.sum(1, keepdims=True)
    return s


def ece(p, y, n_bins=15):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1); tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(tot)


def fit_temperature(logits_dev, y_dev):
    """Single temperature minimising dev NLL. Pass log-probabilities or logits."""
    z = np.asarray(logits_dev, dtype=float)

    def nll(t):
        s = z / t; s = s - s.max(1, keepdims=True)
        return float(-(s[np.arange(len(y_dev)), y_dev] - np.log(np.exp(s).sum(1))).mean())
    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


def apply_temperature(logits, t):
    s = np.asarray(logits, dtype=float) / t; s = s - s.max(1, keepdims=True)
    e = np.exp(s); return e / e.sum(1, keepdims=True)


def evaluate(y, probs, labels):
    y = np.asarray(y); p = to_probs(probs); k = len(labels)
    assert p.shape[1] == k, (p.shape, k)
    pred = p.argmax(1)
    out = full_metrics(y, pred, labels, score=p)
    cm = confusion_matrix(y, pred, labels=list(range(k)))
    out.update(confusion=cm.tolist(), confusion_row_norm=(cm / np.maximum(cm.sum(1, keepdims=True), 1)).round(4).tolist(), ece=ece(p, y))
    if k == 2:
        out["roc_auc"] = float(roc_auc_score(y, p[:, 1]))         # headline binary ROC-AUC; top-2 would be trivially 1.0, so it is omitted
    else:
        out["top2_acc"] = float((np.argsort(-p, 1)[:, :2] == y[:, None]).any(1).mean())
    return out


def fusion_gain(fused, *unimodal, key="macro_f1"):
    """Multimodal metric minus the best single-modality metric (call once with key='macro_f1' and once with the task headline)."""
    return float(fused[key] - max(u[key] for u in unimodal))


def save_run(task, name, seed, labels, y_dev, p_dev, y_test, p_test, extra=None):
    """One JSON per run holding dev and test probabilities and the full metric set on both (test is scored once; dev is for selection)."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    row = dict(task=task, name=name, seed=seed, labels=list(labels), dev=evaluate(y_dev, p_dev, labels), test=evaluate(y_test, p_test, labels),
               y_dev=np.asarray(y_dev).tolist(), y_test=np.asarray(y_test).tolist(),
               probs_dev=np.round(to_probs(p_dev), 6).tolist(), probs_test=np.round(to_probs(p_test), 6).tolist(), **(extra or {}))
    (RUN_DIR / f"{task}_{name}_seed{seed}.json").write_text(json.dumps(row))
    return row


def load_run(task, name, seed):
    r = json.loads((RUN_DIR / f"{task}_{name}_seed{seed}.json").read_text())
    for k in ("y_dev", "y_test", "probs_dev", "probs_test"):
        r[k] = np.asarray(r[k])
    return r
