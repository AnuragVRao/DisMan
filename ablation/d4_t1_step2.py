"""Day 4 step 2 (Task 1 comparison fixes): fair Naive Bayes arms, majority-class baseline, weighted-vs-unweighted BERTweet (post-hoc), GloVe file check, consolidated table.
Writes results/ablation/D4_T1_summary.{json,md}, D4_T1_weighting_posthoc.json, D4_glove_check.json. CPU only."""
import hashlib
import json
import zipfile

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

import d4_t1_data as d4
import metrics_all as M
from bootstrap_utils import N_BOOT, resample_idx
from common import ART, RES, cc

SEEDS = (42, 43, 44)
GLOVE_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.twitter.27B.zip"
POSTHOC = {"tfidf_nb_bal", "tfidf_cnb", "majority", "bertweet_noweight"}      # added after the first Day 4 results were seen (SELECTION_RULE_DAY4.md rules 9-10)


def vec():
    return TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)


def fair_nb(D):
    S, y = D["S"], D["y"]; out = {}
    for name, mk, weighted in (("tfidf_nb_bal", MultinomialNB, True), ("tfidf_cnb", ComplementNB, False)):
        pipe = Pipeline([("tfidf", vec()), ("clf", mk())])
        kw = {"clf__sample_weight": compute_sample_weight("balanced", y["train"])} if weighted else {}
        pipe.fit(S["train"].text, y["train"], **kw)
        out[name] = M.save_run("T1", name, 42, d4.LABELS, y["dev"], pipe.predict_proba(S["dev"].text), y["test"], pipe.predict_proba(S["test"].text), extra=dict(posthoc=True))
    return out


def majority(D):
    y = D["y"]; maj = int(np.bincount(y["train"]).argmax())                        # "always informative" (the train majority)
    p = lambda n: np.tile(np.eye(2)[maj], (n, 1))
    return M.save_run("T1", "majority", 42, d4.LABELS, y["dev"], p(len(y["dev"])), y["test"], p(len(y["test"])), extra=dict(posthoc=True, class_predicted=d4.LABELS[maj]))


def bal_acc_boot(y, pred, idx):
    yt, yp = y[idx], pred[idx]
    r = [((yt == c) & (yp == c)).sum(1) / np.maximum((yt == c).sum(1), 1) for c in range(2)]
    return (r[0] + r[1]) / 2


def weighting_posthoc():
    NI = d4.LABELS.index("not_informative"); rows = {}
    for arm in ("bertweet", "bertweet_noweight"):
        for s in SEEDS:
            r = M.load_run("T1", arm, s)
            rows[f"{arm}_seed{s}"] = {sp: dict(balanced_acc=r[sp]["balanced_acc"], ni_recall=r[sp]["per_class"]["not_informative"]["recall"],
                                              ni_precision=r[sp]["per_class"]["not_informative"]["precision"], macro_f1=r[sp]["macro_f1"], weighted_f1=r[sp]["weighted_f1"])
                                      for sp in ("dev", "test")}
    y = M.load_run("T1", "bertweet", 42)["y_dev"]; idx = resample_idx(len(y), N_BOOT)
    diffs = []
    for s in SEEDS:
        a, b = M.load_run("T1", "bertweet", s), M.load_run("T1", "bertweet_noweight", s)
        diffs.append(bal_acc_boot(y, a["probs_dev"].argmax(1), idx) - bal_acc_boot(y, b["probs_dev"].argmax(1), idx))      # weighted - unweighted
    avg = np.mean(diffs, 0); lo, hi = np.percentile(avg, [2.5, 97.5])
    point = float(np.mean([rows[f"bertweet_seed{s}"]["dev"]["balanced_acc"] - rows[f"bertweet_noweight_seed{s}"]["dev"]["balanced_acc"] for s in SEEDS]))
    shown = bool(lo > 0 or hi < 0)
    verdict = ("weighting retained by pre-registration; " + ("shown to change minority recall (dev balanced accuracy CI excludes 0)" if shown
                else "NOT shown to change minority recall (dev balanced-accuracy CI includes 0)"))
    res = dict(posthoc=True, comparison="weighted minus unweighted BERTweet, dev balanced accuracy, seed-averaged paired bootstrap", n_boot=N_BOOT, point_gain=point,
               ci95=[float(lo), float(hi)], verdict=verdict, per_run=rows)
    (RES / "D4_T1_weighting_posthoc.json").write_text(json.dumps(res, indent=1))
    return res


def glove_check():
    z = ART / "glove" / "glove.twitter.27B.zip"; h = hashlib.sha256()
    with open(z, "rb") as f:
        for ch in iter(lambda: f.read(1 << 24), b""):
            h.update(ch)
    n, dim = 0, None
    with zipfile.ZipFile(z) as zf, zf.open("glove.twitter.27B.200d.txt") as f:
        for line in f:
            if dim is None:
                dim = len(line.decode("utf8").split()) - 1
            n += 1
    res = dict(url=GLOVE_URL, file=z.name, bytes=z.stat().st_size, sha256=h.hexdigest(), member="glove.twitter.27B.200d.txt", n_vectors=n, dim=dim)
    (RES / "D4_glove_check.json").write_text(json.dumps(res, indent=1))
    for s in SEEDS:                                                                  # record it in the BiLSTM run JSONs
        p = M.RUN_DIR / f"T1_bilstm_glove_seed{s}.json"; r = json.loads(p.read_text()); r["glove"] = res; p.write_text(json.dumps(r))
    return res


MODELS = [("majority", "majority class (always informative)"), ("tfidf_lr", "TF-IDF + LR (balanced)"), ("tfidf_nb", "TF-IDF + NB, unweighted (reference)"),
          ("tfidf_nb_bal", "TF-IDF + NB, balanced sample weights"), ("tfidf_cnb", "TF-IDF + ComplementNB"), ("tfidf_rf", "TF-IDF + Random Forest (balanced)"),
          ("tfidf_svm", "TF-IDF + Linear SVM (CV-calibrated)"), ("bilstm_glove", "BiLSTM + GloVe Twitter 200d"), ("bertweet", "BERTweet fine-tuned (weighted CE)"),
          ("bertweet_noweight", "BERTweet fine-tuned, unweighted CE"), ("img_resnet50", "ResNet50 head (frozen) [image-only]"), ("img_dinov2", "DINOv2 head (frozen) [image-only]"),
          ("img_clip_l14", "CLIP L/14 head (frozen) [image-only]")]
KEYS = ("acc", "weighted_f1", "macro_f1", "balanced_acc", "mcc", "roc_auc")


def summary():
    import os
    rows = []
    extra = [(n, l) for n, l in (("late_bertweet+img_clip_l14", "late fusion"), ) if os.path.exists(M.RUN_DIR / f"T1_{n}_seed42.json")]
    for name, label in MODELS + extra:
        runs = []
        for s in SEEDS:
            try:
                runs.append(M.load_run("T1", name, s))
            except FileNotFoundError:
                if s == 42:
                    raise
        row = dict(model=name, label=label, n_seeds=len(runs), posthoc=name in POSTHOC)
        for sp in ("dev", "test"):
            for k in KEYS:
                v = np.array([r[sp][k] for r in runs]); row[f"{sp}_{k}"] = [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else None]
        rows.append(row)
    rows.sort(key=lambda r: -r["dev_weighted_f1"][0])
    (RES / "D4_T1_summary.json").write_text(json.dumps(rows, indent=1))
    f = lambda v: f"{v[0]:.4f}" + (f" ± {v[1]:.4f}" if v[1] is not None else "")
    hdr = "| Model | seeds | dev wF1 | dev bal-acc | test acc | test wF1 | test macro-F1 | test bal-acc | test MCC | test ROC-AUC |\n|---|---|---|---|---|---|---|---|---|---|\n"
    md = "# Task 1 summary (text label; sorted by dev weighted F1; mean ± std over seeds, deterministic models have 1 seed)\n\n" + hdr
    for r in rows:
        md += (f"| {r['label']}{' (post-hoc)' if r['posthoc'] else ''} | {r['n_seeds']} | {f(r['dev_weighted_f1'])} | {f(r['dev_balanced_acc'])} | {f(r['test_acc'])} | "
               f"{f(r['test_weighted_f1'])} | {f(r['test_macro_f1'])} | {f(r['test_balanced_acc'])} | {f(r['test_mcc'])} | {f(r['test_roc_auc'])} |\n")
    (RES / "D4_T1_summary.md").write_text(md, encoding="utf8")
    return md


if __name__ == "__main__":
    D = d4.data()
    for n, r in fair_nb(D).items():
        print(f"{n}: dev wF1 {r['dev']['weighted_f1']:.4f} bal-acc {r['dev']['balanced_acc']:.4f} | test wF1 {r['test']['weighted_f1']:.4f} macro-F1 {r['test']['macro_f1']:.4f}")
    mj = majority(D); print("majority: dev", {k: round(mj['dev'][k], 4) for k in ('acc', 'weighted_f1', 'macro_f1', 'mcc', 'balanced_acc')}, "test", {k: round(mj['test'][k], 4) for k in ('acc', 'weighted_f1', 'macro_f1', 'mcc', 'balanced_acc')})
    w = weighting_posthoc(); print("weighting:", w["verdict"], f"| dev bal-acc gain {w['point_gain']:+.4f} CI {w['ci95']}")
    print("glove:", {k: v for k, v in glove_check().items() if k != "url"})
    print(summary())
