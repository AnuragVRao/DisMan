"""C: calibration (inference only; no retraining). ECE with 15 equal-width confidence bins (top-label, left-closed bins).

T3: the hybrid's logistic PROBE probabilities (fused embedding, C = 1, fitted on v2's index vectors), not the 5-NN vote fractions.
   Besides ECE, test accuracy by (a) probe-neighbour agreement and (b) probe-confidence bins (equal-count quintiles), with AUROCs of both
   signals for predicting correctness, so the report can show that they track correctness.
T2: L4a seeds 42, 43, 44 (existing checkpoints), test split.
If a model's test ECE > 0.05, a single temperature T is fitted on the DEV split (minimising dev NLL) and applied to the test logits; ECE / NLL /
Brier are reported before and after. The T3 probe used for the dev fit is trained on damage-train minus the tweets shared with dev (the test
probe would see dev siblings); T is then applied to the test probe. Writes results/ablation/X/c_calibration_*.json and results/ablation/reliability.pdf.
"""
import json
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")

import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from transformers import DistilBertConfig, DistilBertTokenizerFast

from common import ABL_ART, ART, DATA, RES, SPLIT_DIR, cc, write_row

N_BINS, THRESH, N_BOOT = 15, 0.05, 10_000
faiss.omp_set_num_threads(4)


def no_miss():
    raise RuntimeError("cache miss")


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def bins(conf, correct):
    b = np.minimum((conf * N_BINS).astype(int), N_BINS - 1)
    out = []
    for i in range(N_BINS):
        m = b == i
        out.append(dict(lo=i / N_BINS, hi=(i + 1) / N_BINS, n=int(m.sum()), conf=float(conf[m].mean()) if m.any() else None,
                        acc=float(correct[m].mean()) if m.any() else None))
    return out


def ece(conf, correct):
    n = len(conf)
    return float(sum(b["n"] / n * abs(b["acc"] - b["conf"]) for b in bins(conf, correct) if b["n"]))


def nll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def brier(p, y):
    oh = np.eye(p.shape[1])[y]
    return float(((p - oh) ** 2).sum(1).mean())


def fit_temperature(z_dev, y_dev):
    r = minimize_scalar(lambda T: nll(softmax(z_dev / T), y_dev), bounds=(0.05, 20.0), method="bounded")
    return float(r.x)


def auroc(score, correct):
    pos, neg = correct.astype(bool), ~correct.astype(bool)
    r = rankdata(score)
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def boot_auroc(score, correct, seed=20241025):
    rng = np.random.default_rng(seed); n = len(score); v = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if 0 < correct[i].sum() < n:
            v.append(auroc(score[i], correct[i]))
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]


def report(name, conf, correct, p, y, z=None, z_dev=None, y_dev=None, force=False):
    """ECE / NLL / Brier before, and after temperature scaling (fitted on dev) when the test ECE exceeds THRESH (or force=True, used for the
    three T2 seeds so that the before/after aggregate covers every seed; `exceeds_threshold` records which seeds the 0.05 rule selects)."""
    out = dict(n=int(len(y)), accuracy=float(correct.mean()), mean_confidence=float(conf.mean()), ece_before=ece(conf, correct),
               nll_before=nll(p, y), brier_before=brier(p, y), bins_before=bins(conf, correct), temperature_scaled=False)
    out["exceeds_threshold"] = bool(out["ece_before"] > THRESH)
    if out["exceeds_threshold"] or force:
        T = fit_temperature(z_dev, y_dev)
        p2 = softmax(z / T); c2 = p2.max(1); ok2 = (p2.argmax(1) == y)
        out.update(temperature_scaled=True, temperature=T, ece_after=ece(c2, ok2), nll_after=nll(p2, y), brier_after=brier(p2, y),
                   mean_confidence_after=float(c2.mean()), bins_after=bins(c2, ok2), accuracy_after=float(ok2.mean()),
                   dev_ece_before=ece(softmax(z_dev).max(1), softmax(z_dev).argmax(1) == y_dev),
                   dev_ece_after=ece(softmax(z_dev / T).max(1), softmax(z_dev / T).argmax(1) == y_dev))
    line = f"{name}: acc {out['accuracy']:.4f} mean conf {out['mean_confidence']:.4f} ECE {out['ece_before']:.4f} NLL {out['nll_before']:.4f}"
    if out["temperature_scaled"]:
        line += f" -> T={out['temperature']:.3f}: ECE {out['ece_after']:.4f} NLL {out['nll_after']:.4f} Brier {out['brier_before']:.4f}->{out['brier_after']:.4f}"
    else:
        line += "  (ECE <= 0.05: no temperature scaling)"
    if force and not out["exceeds_threshold"]:
        line += "  [scaled anyway for the three-seed aggregate]"
    print(line, flush=True)
    return out


def main():
    # ── Task 3: probe probabilities ──────────────────────────────────────────────────────────────────────────
    S = {s: cc.load_split("damage", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "dev", "test")}
    V = {s: cc.t3_vectors(cc.cached(ART / "emb" / f"clip_damage_{s}.npy", S[s].image_id.tolist(), no_miss),
                          cc.cached(ART / "emb" / f"sbert_damage_{s}.npy", S[s].image_id.tolist(), no_miss)) for s in S}
    keep = {s: ~S["train"].tweet_id.isin(set(S[s].tweet_id)).to_numpy() for s in ("dev", "test")}
    ylab = S["train"].label.to_numpy()
    probe = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(V["train"][keep["test"]], ylab[keep["test"]])
    cls = [str(c) for c in probe.classes_]; ci = {c: i for i, c in enumerate(cls)}
    y = np.array([ci[l] for l in S["test"].label])
    z = probe.decision_function(V["test"]); P = probe.predict_proba(V["test"]); assert np.allclose(P, softmax(z), atol=1e-9)
    pred = P.argmax(1); conf = P.max(1); correct = pred == y
    stored = json.load(open(RES / "B" / "t3_lr_fused_seed42.json"))
    assert abs(cc.metrics(S["test"].label, [cls[i] for i in pred], cc.T3_LABELS)["macro_f1"] - stored["test_macro_f1"]) < 1e-12
    ix = faiss.IndexFlatIP(V["train"].shape[1]); ix.add(V["train"][keep["test"]])
    sims, nn_ = ix.search(V["test"], 5)
    knn = np.array([ci[cc.knn_vote(list(ylab[keep["test"]][r]), list(s))[0]] for r, s in zip(nn_, sims)])
    agree = knn == pred
    # dev side for temperature scaling (probe trained without the tweets shared with dev)
    probe_d = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(V["train"][keep["dev"]], ylab[keep["dev"]])
    assert [str(c) for c in probe_d.classes_] == cls
    y_dev = np.array([ci[l] for l in S["dev"].label]); z_dev = probe_d.decision_function(V["dev"])
    t3 = report("T3 probe", conf, correct, P, y, z=z, z_dev=z_dev, y_dev=y_dev)

    # (a) agreement, (b) confidence bins, AUROC of both signals for predicting correctness
    t3["by_agreement"] = {k: dict(n=int(m.sum()), accuracy=float(correct[m].mean())) for k, m in (("agree", agree), ("disagree", ~agree))}
    order = np.argsort(conf, kind="stable"); groups = np.array_split(order, 5)
    t3["by_confidence_quintile"] = [dict(quintile=i + 1, n=int(len(g)), conf_min=float(conf[g].min()), conf_max=float(conf[g].max()),
                                         mean_confidence=float(conf[g].mean()), accuracy=float(correct[g].mean())) for i, g in enumerate(groups)]
    t3["spearman_quintile_vs_accuracy"] = float(spearmanr([q["quintile"] for q in t3["by_confidence_quintile"]], [q["accuracy"] for q in t3["by_confidence_quintile"]])[0])
    med = float(np.median(conf)); hi = conf >= med
    t3["agreement_x_confidence"] = {f"{a}_{c}": dict(n=int((m1 & m2).sum()), accuracy=float(correct[m1 & m2].mean()))
                                    for a, m1 in (("agree", agree), ("disagree", ~agree)) for c, m2 in (("high_conf", hi), ("low_conf", ~hi))}
    t3["median_confidence"] = med
    t3["auroc_confidence"] = auroc(conf, correct); t3["auroc_confidence_ci95"] = boot_auroc(conf, correct)
    t3["auroc_agreement"] = auroc(agree.astype(float), correct); t3["auroc_agreement_ci95"] = boot_auroc(agree.astype(float), correct)
    t3["n_boot"] = N_BOOT
    write_row("X", "c_calibration_t3_probe", 42, dict(task="T3", tier="cpu", model="logistic probe, fused, C=1", **t3))
    print(f"  accuracy by agreement: {t3['by_agreement']}")
    print("  accuracy by confidence quintile: " + " | ".join(f"Q{q['quintile']} n={q['n']} conf {q['mean_confidence']:.3f} acc {q['accuracy']:.3f}" for q in t3["by_confidence_quintile"]))
    print(f"  2x2 (agreement x confidence >= median {med:.3f}): " + " | ".join(f"{k} n={v['n']} acc {v['accuracy']:.3f}" for k, v in t3["agreement_x_confidence"].items()))
    print(f"  AUROC (predicting correctness): confidence {t3['auroc_confidence']:.3f} CI {np.round(t3['auroc_confidence_ci95'], 3).tolist()} | "
          f"agreement {t3['auroc_agreement']:.3f} CI {np.round(t3['auroc_agreement_ci95'], 3).tolist()}", flush=True)

    # ── Task 2: L4a, three seeds ─────────────────────────────────────────────────────────────────────────────
    from t2_models import Variant  # noqa: F401  (kept for symmetry with the other scripts)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr2 = cc.load_split("humanitarian", "train", DATA, SPLIT_DIR, verbose=False)[0]
    labels2 = sorted(tr2["label"].unique()); l2i = {l: i for i, l in enumerate(labels2)}
    tok = DistilBertTokenizerFast.from_pretrained(ART / "tokenizer"); bcfg = DistilBertConfig.from_pretrained(ART / "distilbert_config")
    D = {}
    for s in ("dev", "test"):
        d = cc.load_split("humanitarian", s, DATA, SPLIT_DIR, verbose=False)[0]
        ids, mask = cc.tokenize(tok, d.text.tolist(), 128)
        D[s] = (cc.cached(ART / "emb" / f"clip_humanitarian_{s}.npy", d.image_id.tolist(), no_miss), ids, mask, np.array([l2i[l] for l in d.label]))

    @torch.no_grad()
    def logits(model, s):
        vec, ids, mask, _ = D[s]; out = []
        for i in range(0, len(ids), 128):
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                out.append(model(torch.as_tensor(vec[i:i + 128]).float().to(DEVICE), ids[i:i + 128].to(DEVICE), mask[i:i + 128].to(DEVICE)).float().cpu())
        return torch.cat(out).numpy().astype(np.float64)

    panels = {"t3": (conf, correct, t3), }
    for seed in (42, 43, 44):
        model = cc.DualStreamClassifier(len(labels2), bert_config=bcfg).to(DEVICE)
        model.load_state_dict(torch.load(ABL_ART / "ckpt" / f"t2_L4a_seed{seed}.pt", map_location=DEVICE)); model.eval()
        zt, zd = logits(model, "test"), logits(model, "dev")
        yt2, yd2 = D["test"][3], D["dev"][3]
        p = softmax(zt)
        stored2 = json.load(open(RES / "A" / f"t2_L4a_seed{seed}.json"))
        assert abs((p.argmax(1) == yt2).mean() - stored2["test_acc"]) < 1e-4, "T2 accuracy != stored"
        r = report(f"T2 L4a seed {seed}", p.max(1), p.argmax(1) == yt2, p, yt2, z=zt, z_dev=zd, y_dev=yd2, force=True)
        write_row("X", "c_calibration_t2_L4a", seed, dict(task="T2", tier="gpu-small", model="L4a", **r))
        if seed == 42:
            panels["t2"] = (p.max(1), p.argmax(1) == yt2, r)

    # ── reliability diagram ──────────────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    for ax, (key, title) in zip(axes, (("t3", "Task 3 probe (test, n=529)"), ("t2", "Task 2 L4a, seed 42 (test, n=2237)"))):
        _, _, r = panels[key]
        for lab, bl, col, off in (("before", r["bins_before"], "#3b6ea5", -0.5), ("after temperature scaling", r.get("bins_after"), "#d98c2b", 0.5)):
            if bl is None:
                continue
            xs = [(b["lo"] + b["hi"]) / 2 + off * 0.027 for b in bl if b["n"]]
            ax.bar(xs, [b["acc"] for b in bl if b["n"]], width=0.026, color=col, alpha=0.85, label=f"{lab} (ECE {r['ece_before' if lab == 'before' else 'ece_after']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("confidence"); ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("accuracy in bin")
    fig.tight_layout(); fig.savefig(RES / "reliability.pdf")
    if os.environ.get("RELIABILITY_PNG"):
        fig.savefig(os.environ["RELIABILITY_PNG"], dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
