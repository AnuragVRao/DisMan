"""Pick live-demo examples from the damage TEST split with a fixed rule (not by eye); writes demo/live_examples.json.

Rule (hybrid Task 3, temperature-scaled probe confidence, leakage-free index):
  * for each damage class: the test query where the probe label == 5-NN majority == true label with the HIGHEST calibrated probe confidence;
  * one disagreement: the query where the probe and the 5-NN majority differ whose calibrated confidence is CLOSEST TO THE MEDIAN over all
    disagreeing queries (a typical disagreement), reported with whichever side turns out to be right.
Each pick is then run through the real demo pipeline (default settings) so the listed outputs are what the demo shows.
"""
import json
import sys

import faiss
import numpy as np
import pandas as pd

from common import ART, DATA, ROOT, SPLIT_DIR, cc

sys.path.insert(0, str(ROOT / "demo"))
from hybrid import HybridT3  # noqa: E402
from pipeline import Pipeline  # noqa: E402


def no_miss():
    raise RuntimeError("cache miss")


te = cc.load_split("damage", "test", DATA, SPLIT_DIR, verbose=False)[0]
V = cc.t3_vectors(cc.cached(ART / "emb" / "clip_damage_test.npy", te.image_id.tolist(), no_miss),
                  cc.cached(ART / "emb" / "sbert_damage_test.npy", te.image_id.tolist(), no_miss))
index = faiss.read_index(str(ART / "task3_index.faiss")); meta = pd.read_csv(ART / "task3_meta.csv", dtype=str, keep_default_na=False)
T = json.load(open(ROOT / "demo" / "calibration.json"))["t3_probe"]["temperature"]
hyb = HybridT3(index, meta["label"], temperature=T)
scores, idx = index.search(V, 5)
out = [hyb.predict(V[i:i + 1], meta["label"].to_numpy()[idx[i]], scores[i]) for i in range(len(V))]
y = te["label"].to_numpy()
probe = np.array([o["label"] for o in out]); knn = np.array([o["knn_label"] for o in out]); conf = np.array([o["confidence"] for o in out])

picks = []
for cls in cc.T3_LABELS:
    ok = np.where((probe == cls) & (knn == cls) & (y == cls))[0]
    i = int(ok[np.argmax(conf[ok])])
    picks.append(("agree and correct", i))
dis = np.where(probe != knn)[0]
med = float(np.median(conf[dis]))
picks.append(("disagree", int(dis[np.argmin(np.abs(conf[dis] - med))])))

pipe = Pipeline()                                                     # default settings = what the live demo runs
humanitarian = cc.load_split("humanitarian", "test", DATA, SPLIT_DIR, verbose=False)[0].set_index("image_id")["label"].to_dict()
res = []
for kind, i in picks:
    r = te.iloc[i]
    d = pipe.predict(r.img_path, r.tweet_text)
    assert d["t3"]["label"] == probe[i] and d["t3"]["knn_label"] == knn[i], "demo output differs from the offline selection"
    res.append(dict(kind=kind, image_id=r.image_id, event=r.event_name, rel_image=r.rel_image, image_path=str(r.img_path), tweet_text=r.tweet_text,
                    true_damage=r.label, probe_label=d["t3"]["label"], probe_confidence=round(d["t3"]["probe_confidence"], 3),
                    neighbour_majority=d["t3"]["knn_label"], neighbours_agree=bool(d["t3"]["neighbours_agree"]),
                    probe_correct=bool(probe[i] == r.label), neighbour_majority_correct=bool(knn[i] == r.label),
                    neighbour_labels=[n["label"] for n in d["neighbours"]],
                    t1_pred=d["t1"]["label"], t2_pred=d["t2"]["label"], t2_true=humanitarian.get(r.image_id),
                    selection_rule="highest calibrated probe confidence among agree-and-correct queries of this class" if kind == "agree and correct"
                    else f"disagreement closest to the median calibrated confidence ({med:.3f}) among {len(dis)} disagreements"))
(ROOT / "demo" / "live_examples.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf8")
for e in res:
    print(f"[{e['kind']}] {e['event']} | {e['image_id']} | true {e['true_damage']!r} | probe {e['probe_label']!r} conf {e['probe_confidence']:.2f} | "
          f"neighbours {e['neighbour_majority']!r} {e['neighbour_labels']} | probe correct {e['probe_correct']}, neighbours correct {e['neighbour_majority_correct']} | "
          f"T1 {e['t1_pred']} | T2 {e['t2_pred']!r} (true {e['t2_true']!r})\n    image: {e['rel_image']}\n    tweet: {e['tweet_text'][:150]!r}")
