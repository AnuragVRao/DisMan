"""Smoke test for the demo's defaults (hybrid Task 3 ON, temperature-scaled confidences). Run with CAPSTONE_T3_HYBRID / CAPSTONE_T2_CKPT unset.

1. Default Pipeline(): hybrid ON, L4a seed-42 T2 checkpoint, both temperatures read from demo/calibration.json; on N real test images the argmax is
   unchanged by the temperatures (T2 label == unscaled label, T3 label == unscaled argmax), the displayed confidences really are scaled,
   and the demo's probe label equals the offline probe prediction.
2. Pipeline(hybrid_t3=False) (= CAPSTONE_T3_HYBRID=0): v2's 5-NN vote, no hybrid keys, label equal to the default pipeline's neighbour majority.
3. A non-L4a T2 checkpoint (v2's own) is NOT scaled with the L4a temperature.
4. The Gradio handler shows exactly the agree / disagree wording by default and the original wording with CAPSTONE_T3_HYBRID=0.
"""
import json
import os
import subprocess
import sys

import numpy as np

from common import ART, DATA, ROOT, SPLIT_DIR, cc

sys.path.insert(0, str(ROOT / "demo"))
for k in ("CAPSTONE_T3_HYBRID", "CAPSTONE_T2_CKPT"):
    os.environ.pop(k, None)
from pipeline import Pipeline  # noqa: E402

cal = json.load(open(ROOT / "demo" / "calibration.json"))
te = cc.load_split("damage", "test", DATA, SPLIT_DIR, verbose=False)[0]
N = 60
rows = te.iloc[:N]

# 1. default
dflt = Pipeline()
assert dflt.hybrid is not None and dflt.t2_ckpt == cal["t2"]["checkpoint"]
assert dflt.t2_T == cal["t2"]["temperature"] and dflt.hybrid.temperature == cal["t3_probe"]["temperature"]
print(f"1. default: hybrid ON, T2 checkpoint {dflt.t2_ckpt}, T2 T = {dflt.t2_T:.4f}, T3 probe T = {dflt.hybrid.temperature:.4f} (both read from demo/calibration.json)")
res = [dflt.predict(r.img_path, r.tweet_text) for r in rows.itertuples()]
t2_same = sum(r["t2"]["label"] == r["t2"]["label_unscaled"] for r in res)
t3_same = sum(r["t3"]["label"] == max(r["t3"]["probe_probs_unscaled"], key=r["t3"]["probe_probs_unscaled"].get) for r in res)
print(f"   argmax unchanged by temperature scaling on the {N} smoke-test rows: T2 {t2_same}/{N}, T3 {t3_same}/{N}")
assert t2_same == N and t3_same == N
conf_scaled = np.array([r["t3"]["probe_confidence"] for r in res]); conf_raw = np.array([max(r["t3"]["probe_probs_unscaled"].values()) for r in res])
t2_scaled = np.array([max(r["t2"]["probs"].values()) for r in res])
print(f"   displayed confidences are scaled: T3 probe mean {conf_raw.mean():.3f} -> {conf_scaled.mean():.3f} (T < 1 sharpens); "
      f"T2 top-class mean {t2_scaled.mean():.3f} (T = {dflt.t2_T:.3f} > 1 softens)")
assert conf_scaled.mean() > conf_raw.mean()
V = cc.t3_vectors(cc.cached(ART / "emb" / "clip_damage_test.npy", te.image_id.tolist(), lambda: (_ for _ in ()).throw(RuntimeError())),
                  cc.cached(ART / "emb" / "sbert_damage_test.npy", te.image_id.tolist(), lambda: (_ for _ in ()).throw(RuntimeError())))[:N]
off = [str(c) for c in dflt.hybrid.clf.classes_[dflt.hybrid.clf.predict_proba(V).argmax(1)]]
same = sum(r["t3"]["label"] == p for r, p in zip(res, off))
print(f"   demo probe label equals the offline probe prediction on {same}/{N} rows")
assert same >= N - 2
dis = [(r, t) for r, t in zip(res, rows.itertuples()) if not r["t3"]["neighbours_agree"]]
agr = [(r, t) for r, t in zip(res, rows.itertuples()) if r["t3"]["neighbours_agree"]]
print(f"   neighbours disagree with the probe on {len(dis)}/{N} rows")

# 2. CAPSTONE_T3_HYBRID=0 equivalent
old = Pipeline(hybrid_t3=False)
assert old.hybrid is None
res0 = [old.predict(r.img_path, r.tweet_text) for r in rows.itertuples()]
assert all("knn_label" not in r["t3"] for r in res0)
eq = sum(a["t3"]["label"] == b["t3"]["knn_label"] for a, b in zip(res0, res))
print(f"2. hybrid off (v2 behaviour): no hybrid keys; its label equals the default pipeline's neighbour majority on {eq}/{N} rows")
assert eq == N

# 3 + 4. subprocesses (app.py builds its Pipeline at import time)
env0 = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONWARNINGS="ignore")
chk = "import sys; sys.path.insert(0, r'%s'); from pipeline import Pipeline; p = Pipeline(); print('T2T::%%s::%%s' %% (p.t2_ckpt, p.t2_T))" % str(ROOT / "demo")
out = subprocess.run([sys.executable, "-c", chk], capture_output=True, text=True, cwd=ROOT,
                     env=dict(env0, CAPSTONE_T2_CKPT=str(ART / "task2_model.pt"))).stdout
line = [l for l in out.splitlines() if l.startswith("T2T::")][0].split("::")
print(f"3. T2 checkpoint {line[1]}: temperature {line[2]} (the L4a temperature is not applied to other checkpoints)")
assert line[2] == "1.0"

code = ("import sys; sys.path.insert(0, r'%s'); import app; from PIL import Image; out = app.run(Image.open(r'%s'), %r); "
        "print('T3MD::' + out[3].replace(chr(10), ' | '))")
for label, (r, t), flag in (("agree", agr[0], None), ("disagree", dis[0], None), ("agree, CAPSTONE_T3_HYBRID=0", agr[0], "0")):
    env = dict(env0, **({"CAPSTONE_T3_HYBRID": flag} if flag else {}))
    p = subprocess.run([sys.executable, "-c", code % (str(ROOT / "demo"), t.img_path, t.tweet_text)], capture_output=True, text=True, env=env, cwd=ROOT)
    md = [l for l in p.stdout.splitlines() if l.startswith("T3MD::")]
    assert md, p.stderr[-800:]
    print(f"4. app.run [{label}]: {md[0][6:]}")
    if flag is None and label == "agree":
        assert "Similar past cases agree." in md[0] and "Confidence: " in md[0] and "Probe" not in md[0] and "temperature" not in md[0]
    if flag is None and label == "disagree":
        assert "(lower confidence)." in md[0] and "Similar past cases suggest" in md[0] and "Recommend human review." in md[0]
        assert "Confidence: " in md[0] and "Probe" not in md[0] and "temperature" not in md[0]
    if flag == "0":
        assert md[0][6:].startswith("**Predicted damage severity:") and "Similar past cases" not in md[0]
print("SMOKE TEST PASSED")
