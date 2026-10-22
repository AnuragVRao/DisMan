"""Build demo_examples/: ~10 upload-ready examples (image + tweet text) with a manifest of truth vs model output.

Candidates: test-split rows only, with no tweet_id/text/image twin in the damage-train index, excluding the preloaded examples.
Selection is by true label first, then by model correctness for the 'good' showcase picks; two deliberate failure cases are
included and labelled. Correctness-based picking is disclosed in the manifest. Run from the repo root.
"""
import json, re, shutil, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "demo"); sys.path.insert(0, ".")
import crisis_common as cc
from pipeline import Pipeline

D = Path("data/CrisisMMD_v2.0"); S = D / "crisismmd_datasplit_all" / "crisismmd_datasplit_all"
load = lambda t, s: cc.load_split(t, s, D, S, verbose=False)[0]
tr3, te3, te2, te1 = load("damage", "train"), load("damage", "test"), load("humanitarian", "test"), load("informative", "test")
pre = {e["image_id"] for e in json.load(open("demo/examples.json"))}
t1 = te1.set_index("image_id")["label"]; t3 = te3.set_index("image_id")["label"]

pool = te2[~(te2.tweet_id.isin(set(tr3.tweet_id)) | te2.text.isin(set(tr3.text)) | te2.image_id.isin(set(tr3.image_id)))].copy()
extra = te3[~te3.image_id.isin(pool.image_id)]
extra = extra[~(extra.tweet_id.isin(set(tr3.tweet_id)) | extra.text.isin(set(tr3.text)))].copy()
extra["label"] = extra.image_id.map(te2.set_index("image_id")["label"])          # T2 truth where available
pool = __import__("pandas").concat([pool.rename(columns={"label": "t2_true"}), extra.rename(columns={"label": "t2_true"})])
pool = pool[~pool.image_id.isin(pre) & pool.text.str.len().gt(40) & ~pool.tweet_text.str.contains("â|Ã|�", regex=True)]
pool = pool.drop_duplicates("image_id")
pool["t1_true"] = pool.image_id.map(t1).fillna("n/a"); pool["t3_true"] = pool.image_id.map(t3).fillna("n/a")
pool["t2_true"] = pool.t2_true.fillna("n/a")
rng = np.random.RandomState(cc.SEED); pool = pool.iloc[rng.permutation(len(pool))].reset_index(drop=True)
pool = pool.iloc[:900]                                                              # plenty of candidates, bounded runtime
print("candidate pool:", len(pool))

P = Pipeline(); P.predict(D / pool.rel_image[0], pool.tweet_text[0])
rows = []
for r in pool.itertuples():
    o = P.predict(D / r.rel_image, r.tweet_text)
    rows.append(dict(r._asdict(), p1=o["t1"]["label"], p1p=o["t1"]["prob"], p2=o["t2"]["label"], p2p=max(o["t2"]["probs"].values()),
                     p3=o["t3"]["label"], p3s=max(o["t3"]["vote_share"].values())))
import pandas as pd
R = pd.DataFrame(rows)
ok2, ok3 = (R.p2 == R.t2_true), (R.p3 == R.t3_true)
used = set()
def pick(mask, name, why, kind="good"):
    c = R[mask & ~R.image_id.isin(used)]
    if not len(c): print("NO CANDIDATE for", name); return
    r = c.iloc[0]; used.add(r.image_id); chosen.append((name, why, kind, r))
chosen = []
disagree = ((R.p1 == "informative") & (R.p2 == "Not humanitarian")) | ((R.p1 == "not_informative") & (R.p2 != "Not humanitarian"))
pick(ok3 & (R.t3_true == "Severe damage") & ok2 & (R.p3s >= 0.8), "severe_damage", "Clear severe damage: infrastructure class plus a unanimous, similar-cases vote")
pick(ok3 & (R.t3_true == "Mild damage"), "mild_damage", "Mild damage is the hardest severity class (18% recall overall); a correct one is worth showing")
pick(ok3 & (R.t3_true == "Little or no damage"), "little_or_no_damage", "Little or no damage")
pick(ok2 & (R.t2_true == "Rescue, volunteering or donation effort"), "rescue_donation", "Rescue / donation tweet: the Task 2 class that is not about damage")
pick(ok2 & (R.t2_true == "Affected individuals"), "affected_individuals", "Affected individuals (a small class: 88 test samples)")
pick(ok2 & (R.t2_true == "Not humanitarian") & (R.p1 == "not_informative") & (R.t1_true == "not_informative"), "not_humanitarian", "Irrelevant tweet: both Task 1 and Task 2 say so")
pick(ok2 & (R.t2_true == "Vehicle damage"), "vehicle_damage", "Vehicle damage (a small class: 36 test samples)")
pick(disagree & ~(R.t2_true == "n/a"), "t1_t2_disagree", "Task 1 and Task 2 disagree: shows the disagreement note (text labels vs image labels)")
pick(~ok2 & (R.p2p > 0.7) & (R.t2_true != "n/a"), "FAILURE_task2", "Task 2 confidently wrong: shown on purpose", "failure")
pick(~ok3 & (R.t3_true != "n/a") & (R.p3s >= 0.8), "FAILURE_task3", "Task 3 unanimous vote but wrong: shows that vote share is not accuracy", "failure")

out = Path("demo_examples"); shutil.rmtree(out, ignore_errors=True); out.mkdir()
md = ["# Demo examples", "",
      "Upload the image and paste the text from the matching `.txt` file into the demo. All images are from the **test split** and have no",
      "tweet/text/image twin in the damage-train index (the neighbour panel cannot show the same tweet). The six built-in *Preloaded examples*",
      "are different images.", "",
      "**Selection:** picked by true label first, then (for the `good` picks) by the model being right, so these show the system at its best;",
      "the two `FAILURE_*` rows are deliberate failure cases, to show the limits honestly. Overall test accuracy is in `RESULTS.md`, not here.", "",
      "`n/a` = the image is not in that task's test split, so there is no ground truth for that column.", "",
      "| # | File | Kind | Event | T1 true -> pred | T2 true -> pred (p) | T3 true -> pred (vote share) | Why show it |", "|---|---|---|---|---|---|---|---|"]
for i, (name, why, kind, r) in enumerate(chosen, 1):
    stem = f"{i:02d}_{name}"
    shutil.copy(D / r.rel_image, out / f"{stem}{Path(r.rel_image).suffix}")
    (out / f"{stem}.txt").write_text(re.sub(r"\s*https?://\S+", "", r.tweet_text).strip(), encoding="utf-8")
    md.append(f"| {i} | `{stem}` | {kind} | {r.event_name} | {r.t1_true} -> {r.p1} | {r.t2_true} -> {r.p2} ({r.p2p:.2f}) | {r.t3_true} -> {r.p3} ({r.p3s:.1f}) | {why} |")
(out / "MANIFEST.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md[-(len(chosen)+0):]))
