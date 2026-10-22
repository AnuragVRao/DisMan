"""Pick ~6 preloaded demo examples from the damage TEST split (so all three tasks have ground truth where available).

Selection uses true labels only (never model predictions). Tweets with a tweet_id/text twin in ANY damage-train row are
excluded, so the neighbour panel can never show the same tweet. Run from the repo root.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import crisis_common as cc  # noqa: E402

D = Path("data/CrisisMMD_v2.0")
S = D / "crisismmd_datasplit_all" / "crisismmd_datasplit_all"
load = lambda t, s: cc.load_split(t, s, D, S, verbose=False)[0]

tr, te = load("damage", "train"), load("damage", "test")
twin = te.tweet_id.isin(set(tr.tweet_id)) | te.text.isin(set(tr.text))
print(f"damage test rows: {len(te)}; with a tweet_id/text twin in damage train (excluded): {int(twin.sum())}")
pool = te[~twin].copy()

# ground truth for the other tasks where the image is in their test split
t2 = load("humanitarian", "test"); t1 = load("informative", "test").set_index("image_id")["label"]
t3_true = te.set_index("image_id")["label"]
rng = np.random.RandomState(cc.SEED)
no_twin = lambda d: d[~(d.tweet_id.isin(set(tr.tweet_id)) | d.text.isin(set(tr.text)) | d.image_id.isin(set(tr.image_id)))
                      & d.text.str.len().gt(25)].copy()

# A) three damage-test examples, one per severity label, preferring distinct Task 2 labels
A = pool = no_twin(te)
A["t2_true"] = A.image_id.map(t2.set_index("image_id")["label"])
chosen, used = [], set()
for lab in cc.T3_LABELS:
    c = A[A.label == lab]; fresh = c[~c.t2_true.isin(used)]; c = fresh if len(fresh) else c
    r = c.iloc[rng.randint(len(c))]; chosen.append(r); used.add(r.t2_true)
# B) three humanitarian-test examples of other Task 2 classes (damage truth only where the image is in damage-test)
B = no_twin(t2).rename(columns={"label": "t2_true"})
B = B[~B.image_id.isin([r.image_id for r in chosen])]
for lab in ["Not humanitarian", "Rescue, volunteering or donation effort", "Affected individuals"]:
    c = B[B.t2_true == lab]; r = c.iloc[rng.randint(len(c))].copy()
    r["label"] = t3_true.get(r.image_id, "n/a"); chosen.append(r)
for r in chosen:
    r["t1_true"] = t1.get(r.image_id, "n/a")
    if "t2_true" not in r or not isinstance(r["t2_true"], str): r["t2_true"] = "n/a"
ex = [dict(image_id=r.image_id, tweet_id=r.tweet_id, event=r.event_name, rel_image=r.rel_image, tweet_text=r.tweet_text,
           t1_true=r.t1_true, t2_true=r.t2_true, t3_true=r.label) for r in chosen]

# ── the assert the user asked for: no twin of any example anywhere in the damage-train index source ──
for e in ex:
    assert e["tweet_id"] not in set(tr.tweet_id), f"twin tweet_id in train: {e['tweet_id']}"
    assert cc.clean_tweet(e["tweet_text"]) not in set(tr.text), f"twin text in train: {e['tweet_id']}"
    assert e["image_id"] not in set(tr.image_id)
print("OK: no preloaded example has a tweet_id/text/image twin in the damage-train split")

Path("demo").mkdir(exist_ok=True)
json.dump(ex, open("demo/examples.json", "w"), indent=1)
for e in ex:
    print(f"- {e['event']:20s} T3={e['t3_true']:20s} T2={e['t2_true']:42s} T1={e['t1_true']:16s} {e['tweet_text'][:70]!r}")
