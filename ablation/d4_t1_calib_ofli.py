"""Day 4 final round B4 + B6 (CPU, from stored probabilities).
B4: calibration of the Task 1 BERTweet (3 seeds + the deployed checkpoint) like the report's calibration table: test accuracy, mean confidence, ECE, dev-fitted temperature T, ECE after (and the dev ECE before/after).
B6: INDICATIVE comparison with Ofli et al.: BERTweet on the Task 1 test rows where the text and image labels agree (n = 1,534 of datasplit_all; NOT Ofli's agreed-label split), weighted F1 and accuracy.
Writes results/ablation/D4_T1_calibration.{json,md} and D4_T1_ofli.json."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import metrics_all as M
from common import DATA, RES, SPLIT_DIR, cc

RUNS = [("bertweet", 42, "BERTweet, seed 42 (original run)"), ("bertweet", 43, "BERTweet, seed 43"), ("bertweet", 44, "BERTweet, seed 44"), ("bertweet_deploy", 42, "BERTweet, deployed checkpoint (retrain of seed 42)")]


def calib():
    rows = []
    for name, seed, label in RUNS:
        r = M.load_run("T1", name, seed); out = dict(model=label, run=f"{name}_seed{seed}")
        zd, zt = np.log(np.clip(r["probs_dev"], 1e-9, 1)), np.log(np.clip(r["probs_test"], 1e-9, 1)); T = M.fit_temperature(zd, r["y_dev"])
        for sp, z, yy in (("dev", zd, r["y_dev"]), ("test", zt, r["y_test"])):
            p0, p1 = M.to_probs(z), M.apply_temperature(z, T)
            out[sp] = dict(acc=float((p0.argmax(1) == yy).mean()), mean_conf=float(p0.max(1).mean()), ece=M.ece(p0, yy), ece_after=M.ece(p1, yy), mean_conf_after=float(p1.max(1).mean()))
        out["T"] = float(T); rows.append(out)
    return rows


def ofli():
    D = {s: cc.load_split("informative", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("test",)}["test"]
    raw = pd.read_csv(SPLIT_DIR / "task_informative_text_img_test.tsv", sep="\t", dtype=str).drop_duplicates(subset=["image_id"]).set_index("image_id").loc[D.image_id]
    agree = (raw.label_text.str.strip() == raw.label_image.str.strip()).values; assert int(agree.sum()) == 1534
    res = {}
    for name, seeds in (("bertweet", (42, 43, 44)), ("bertweet_deploy", (42,)), ("tfidf_lr", (42,))):
        a, f = [], []
        for s in seeds:
            r = M.load_run("T1", name, s); y, p = r["y_test"][agree], np.array(r["probs_test"])[agree].argmax(1)
            a.append(float((y == p).mean())); f.append(float(f1_score(y, p, labels=[0, 1], average="weighted", zero_division=0)))
        res[name] = dict(n=int(agree.sum()), seeds=len(seeds), acc=[float(np.mean(a)), float(np.std(a, ddof=1)) if len(a) > 1 else None], weighted_f1=[float(np.mean(f)), float(np.std(f, ddof=1)) if len(f) > 1 else None])
    return res


if __name__ == "__main__":
    rows = calib(); of = ofli()
    (RES / "D4_T1_calibration.json").write_text(json.dumps(rows, indent=1)); (RES / "D4_T1_ofli.json").write_text(json.dumps(of, indent=1))
    md = "# Task 1 calibration (BERTweet; temperature fitted on dev; ECE with 15 equal-width bins)\n\n| Model | split | Acc. | Mean conf. | ECE | T | ECE after |\n|---|---|---|---|---|---|---|\n"
    for r in rows:
        for sp in ("dev", "test"):
            md += f"| {r['model']} | {sp} | {r[sp]['acc']:.4f} | {r[sp]['mean_conf']:.4f} | {r[sp]['ece']:.4f} | {r['T']:.3f} | {r[sp]['ece_after']:.4f} |\n"
    (RES / "D4_T1_calibration.md").write_text(md, encoding="utf-8"); print(md); print(json.dumps(of, indent=1))
