"""Day 4 step 3e/3f, Task 1: fusion gain, paired dev bootstrap for the two pre-registered comparisons, agreed/disagreed split.
Fusion gain per seed = fused - best unimodal, where "best unimodal" is the better of {BERTweet text-only, CLIP L/14 image head} by DEV weighted F1 of that seed (never by test); reported for
weighted F1 (Task 1 headline) and macro-F1, on dev and on test.
Bootstrap (dev, paired, seed-averaged, 10,000 resamples, one resample matrix, seed 20241025): (i) best fused model (highest mean dev weighted F1) vs BERTweet text-only;
(ii) the guide's core model BERTweet(ft)+ResNet50(frozen) vs our deployed choice (BERTweet text-only). Test CIs are reported afterwards, for reporting only.
Agreed/disagreed split of the T1 test set by whether label_text == label_image (the scoring label stays label_text). Writes D4_T1_fusion.{json,md}."""
import json

import numpy as np
import pandas as pd

import d4_prov
import d4_t1_data as d4
import metrics_all as M
from bootstrap_utils import N_BOOT, resample_idx
from common import RES, SPLIT_DIR

SEEDS = (42, 43, 44)
TEXT, IMAGE = "bertweet", "img_clip_l14"
FUSED = [("late_bertweet+img_clip_l14", "late fusion (calibrated avg) BERTweet + CLIP L/14 head"),
         ("early_bertweet_resnet50", "early fusion BERTweet (fine-tuned) + ResNet50 (frozen)  [guide's core model]"),
         ("early_bertweet_clip_l14", "early fusion BERTweet (fine-tuned) + CLIP L/14 (frozen)  [optional]")]


def f1s(y, p, idx):
    """per-class F1 for every resample: list of K arrays (n_boot,), plus supports (n_boot,K)."""
    yt, yp = y[idx], p[idx]; fs, sup = [], []
    for c in range(2):
        tp = ((yt == c) & (yp == c)).sum(1); s = (yt == c).sum(1); npd = (yp == c).sum(1)
        fs.append(np.where(s + npd > 0, 2 * tp / np.maximum(s + npd, 1), 0.0)); sup.append(s)
    return fs, np.stack(sup, 1)


def boot(y, p, idx):
    fs, sup = f1s(y, p, idx)
    return dict(weighted_f1=(fs[0] * sup[:, 0] + fs[1] * sup[:, 1]) / idx.shape[1], macro_f1=(fs[0] + fs[1]) / 2)


def subset_metrics(y, p, mask):
    ev = M.evaluate(y[mask], p[mask], d4.LABELS)
    return dict(n=int(mask.sum()), acc=ev["acc"], weighted_f1=ev["weighted_f1"], macro_f1=ev["macro_f1"])


def load(name, seed):
    r = M.load_run("T1", name, seed); d4_prov.assert_run(r, "informative", f"{name} seed{seed}"); return r


if __name__ == "__main__":
    D = d4.data()
    import os
    fused = [(n, l) for n, l in FUSED if os.path.exists(M.RUN_DIR / f"T1_{n}_seed42.json")]
    R = {n: {s: load(n, s) for s in SEEDS} for n in [TEXT, IMAGE] + [n for n, _ in fused]}
    y = {sp: R[TEXT][42][f"y_{sp}"] for sp in ("dev", "test")}
    for n in R:
        for s in SEEDS:
            assert (R[n][s]["y_dev"] == y["dev"]).all() and (R[n][s]["y_test"] == y["test"]).all()
    out = dict(models={}, gains={}, bootstrap={}, subsets={})
    # --- fusion gain per seed
    for n, label in fused:
        per = []
        for s in SEEDS:
            tx, im, fu = R[TEXT][s], R[IMAGE][s], R[n][s]
            best = tx if tx["dev"]["weighted_f1"] >= im["dev"]["weighted_f1"] else im          # chosen on dev
            per.append({sp: {k: fu[sp][k] - best[sp][k] for k in ("weighted_f1", "macro_f1")} for sp in ("dev", "test")} | dict(best_unimodal=best["name"]))
        out["gains"][n] = dict(label=label, per_seed=per, mean={sp: {k: float(np.mean([p[sp][k] for p in per])) for k in ("weighted_f1", "macro_f1")} for sp in ("dev", "test")},
                               std={sp: {k: float(np.std([p[sp][k] for p in per], ddof=1)) for k in ("weighted_f1", "macro_f1")} for sp in ("dev", "test")},
                               positive_every_seed_weighted_f1_dev=all(p["dev"]["weighted_f1"] > 0 for p in per), positive_every_seed_weighted_f1_test=all(p["test"]["weighted_f1"] > 0 for p in per))
    # --- dev bootstrap: two comparisons (+ test, reporting only)
    best_fused = max(fused, key=lambda t: np.mean([R[t[0]][s]["dev"]["weighted_f1"] for s in SEEDS]))[0]
    comps = {"best_fused_vs_text_only": (best_fused, TEXT), "guides_core_vs_deployed": ("early_bertweet_resnet50", TEXT)}
    for key, (a, b) in comps.items():
        out["bootstrap"][key] = dict(a=a, b=b)
        for sp in ("dev", "test"):
            idx = resample_idx(len(y[sp]), N_BOOT)
            d = {k: np.mean([boot(y[sp], R[a][s][f"probs_{sp}"].argmax(1), idx)[k] - boot(y[sp], R[b][s][f"probs_{sp}"].argmax(1), idx)[k] for s in SEEDS], 0) for k in ("weighted_f1", "macro_f1")}
            pt = {k: float(np.mean([R[a][s][sp][k] - R[b][s][sp][k] for s in SEEDS])) for k in ("weighted_f1", "macro_f1")}
            out["bootstrap"][key][sp] = {k: dict(point=pt[k], ci95=[float(v) for v in np.percentile(d[k], [2.5, 97.5])],
                                                 verdict=("better" if np.percentile(d[k], 2.5) > 0 else "worse" if np.percentile(d[k], 97.5) < 0 else "not distinguishable")) for k in d}
    # --- agreed / disagreed split (label_text vs label_image of the same row)
    for sp in ("dev", "test"):
        raw = pd.read_csv(SPLIT_DIR / f"task_informative_text_img_{sp}.tsv", sep="\t", dtype=str).drop_duplicates(subset=["image_id"]).set_index("image_id")
        lt, li = raw.loc[D["ids"][sp], "label_text"].str.strip().values, raw.loc[D["ids"][sp], "label_image"].str.strip().values
        agree = lt == li
        out["subsets"][sp] = dict(n_agree=int(agree.sum()), n_disagree=int((~agree).sum()), informative_share_agree=float((y[sp][agree] == 0).mean()), informative_share_disagree=float((y[sp][~agree] == 0).mean()), models={})
        for n in [TEXT, IMAGE] + [f for f, _ in fused]:
            out["subsets"][sp]["models"][n] = {nm: {k: float(np.mean([subset_metrics(y[sp], R[n][s][f"probs_{sp}"], m)[k] for s in SEEDS])) for k in ("acc", "weighted_f1", "macro_f1")}
                                               for nm, m in (("agree", agree), ("disagree", ~agree))}
    (RES / "D4_T1_fusion.json").write_text(json.dumps(out, indent=1))
    # --- markdown
    f4 = lambda v: f"{v:+.4f}"
    md = "# Task 1 fusion (text label; mean over 3 seeds)\n\n## Fusion gain = fused - best unimodal (best unimodal chosen per seed on dev weighted F1)\n\n"
    md += "| Fused model | dev gain wF1 | dev gain macro-F1 | test gain wF1 | test gain macro-F1 | positive every seed (dev wF1 / test wF1) |\n|---|---|---|---|---|---|\n"
    for n, g in out["gains"].items():
        md += (f"| {g['label']} | {f4(g['mean']['dev']['weighted_f1'])} ± {g['std']['dev']['weighted_f1']:.4f} | {f4(g['mean']['dev']['macro_f1'])} ± {g['std']['dev']['macro_f1']:.4f} | "
               f"{f4(g['mean']['test']['weighted_f1'])} ± {g['std']['test']['weighted_f1']:.4f} | {f4(g['mean']['test']['macro_f1'])} ± {g['std']['test']['macro_f1']:.4f} | "
               f"{g['positive_every_seed_weighted_f1_dev']} / {g['positive_every_seed_weighted_f1_test']} |\n")
    md += "\n## Paired bootstrap, seed-averaged, 10,000 resamples (dev decides; test is reporting only)\n\n| Comparison | split | metric | point | 95% CI | verdict |\n|---|---|---|---|---|---|\n"
    for key, c in out["bootstrap"].items():
        for sp in ("dev", "test"):
            for k, v in c[sp].items():
                md += f"| {c['a']} − {c['b']} ({key}) | {sp} | {k} | {f4(v['point'])} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | {v['verdict']} |\n"
    for sp in ("test", "dev"):
        s = out["subsets"][sp]
        md += (f"\n## {sp.capitalize()} split by label agreement (label_text == label_image): agreed n={s['n_agree']}, disagreed n={s['n_disagree']}; informative share "
               f"{s['informative_share_agree']:.2f} (agreed) vs {s['informative_share_disagree']:.2f} (disagreed)\n\n| Model | agreed wF1 | agreed macro-F1 | agreed acc | disagreed wF1 | disagreed macro-F1 | disagreed acc |\n|---|---|---|---|---|---|---|\n")
        for n, m in s["models"].items():
            md += f"| {n} | " + " | ".join(f"{m[g][k]:.4f}" for g in ("agree", "disagree") for k in ("weighted_f1", "macro_f1", "acc")) + " |\n"
    (RES / "D4_T1_fusion.md").write_text(md, encoding="utf-8")
    print(md)
