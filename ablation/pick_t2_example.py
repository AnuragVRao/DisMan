"""Add one Task 2 live-demo example to demo/live_examples.json (the four Task 3 examples already there are kept as they are).

Rule: among humanitarian-TEST rows whose true label is "Not humanitarian" or "Rescue, volunteering or donation effort" (not infrastructure damage),
keep those where the demo's L4a model predicts the true label AND Task 1 predicts "not_informative"; take the row with the highest temperature-scaled
T2 confidence (T from demo/calibration.json). If no row has Task 1 = not_informative, the Task 1 condition would be dropped (reported if so).
The pick is then run through the real demo pipeline (default settings) and its outputs are asserted equal to the offline selection.
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import DistilBertConfig, DistilBertTokenizerFast

from common import ABL_ART, ART, DATA, ROOT, SPLIT_DIR, cc

sys.path.insert(0, str(ROOT / "demo"))
from pipeline import Pipeline  # noqa: E402

TARGET = ("Not humanitarian", "Rescue, volunteering or donation effort")
LIVE = ROOT / "demo" / "live_examples.json"
live = json.load(open(LIVE, encoding="utf8"))
assert not any(e["kind"].startswith("T2") for e in live), "a T2 example is already present"


def no_miss():
    raise RuntimeError("cache miss")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tr, te = (cc.load_split("humanitarian", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "test"))
labels = sorted(tr["label"].unique()); l2i = {l: i for i, l in enumerate(labels)}
cal = json.load(open(ROOT / "demo" / "calibration.json"))
assert cal["t2"]["checkpoint"] == "t2_L4a_seed42.pt"
T = cal["t2"]["temperature"]

tok = DistilBertTokenizerFast.from_pretrained(ART / "tokenizer"); bcfg = DistilBertConfig.from_pretrained(ART / "distilbert_config")
ids, mask = cc.tokenize(tok, te.text.tolist(), 128)
vec = cc.cached(ART / "emb" / "clip_humanitarian_test.npy", te.image_id.tolist(), no_miss)
model = cc.DualStreamClassifier(len(labels), bert_config=bcfg).to(DEVICE)
model.load_state_dict(torch.load(ABL_ART / "ckpt" / "t2_L4a_seed42.pt", map_location=DEVICE)); model.eval()
z = []
with torch.no_grad():
    for i in range(0, len(ids), 128):
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            z.append(model(torch.as_tensor(vec[i:i + 128]).float().to(DEVICE), ids[i:i + 128].to(DEVICE), mask[i:i + 128].to(DEVICE)).float().cpu())
z = torch.cat(z).double()
p = torch.softmax(z / T, -1).numpy()
pred = np.array([labels[i] for i in p.argmax(1)]); conf = p.max(1)

t1 = joblib.load(ART / "task1_pipeline.joblib")
t1_pred = t1.predict(te["text"])                                   # the same cleaned text the demo feeds Task 1
cand_all = np.where(np.isin(te["label"].to_numpy(), TARGET) & (pred == te["label"].to_numpy()))[0]
cand = cand_all[t1_pred[cand_all] == "not_informative"]
print(f"correct predictions with a true label in {TARGET}: {len(cand_all)}; of these Task 1 = not_informative: {len(cand)}")
use = cand if len(cand) else cand_all
i = int(use[np.argmax(conf[use])])
r = te.iloc[i]

t1_true = None
inf = pd.read_csv(SPLIT_DIR / "task_informative_text_img_test.tsv", sep="\t", dtype=str)
m = inf[inf["image_id"] == r.image_id]
if len(m):
    t1_true = str(m.iloc[0]["label_text"])
pipe = Pipeline()
d = pipe.predict(r.img_path, r.tweet_text)
assert d["t2"]["label"] == pred[i] == r.label and abs(d["t2"]["probs"][pred[i]] - conf[i]) < 5e-3, "demo output differs from the offline selection"
entry = dict(kind="T2 non-infrastructure", image_id=r.image_id, event=r.event_name, rel_image=r.rel_image, image_path=str(r.img_path), tweet_text=r.tweet_text,
             t2_true=r.label, t2_pred=d["t2"]["label"], t2_confidence=round(float(d["t2"]["probs"][d["t2"]["label"]]), 3),
             t1_pred=d["t1"]["label"], t1_pred_prob=round(float(d["t1"]["prob"]), 3), t1_true=t1_true,
             t3_label=d["t3"]["label"], t3_neighbours_agree=bool(d["t3"].get("neighbours_agree", True)), t3_probe_confidence=round(float(d["t3"]["probe_confidence"]), 3),
             selection_rule=("highest temperature-scaled T2 confidence among correct L4a predictions whose true label is 'Not humanitarian' or "
                             f"'Rescue, volunteering or donation effort' and that Task 1 flags as not_informative ({len(cand)} candidates)"
                             if len(cand) else "no candidate had Task 1 = not_informative; highest-confidence correct prediction of the two classes"))
live.append(entry)
LIVE.write_text(json.dumps(live, indent=1, ensure_ascii=False), encoding="utf8")
print(json.dumps({k: entry[k] for k in ("event", "image_id", "rel_image", "t2_true", "t2_pred", "t2_confidence", "t1_pred", "t1_pred_prob", "t1_true", "t3_label")}, ensure_ascii=False))
print("tweet:", entry["tweet_text"][:200].encode("ascii", "replace").decode())
print("rule :", entry["selection_rule"])
