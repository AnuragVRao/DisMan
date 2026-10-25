"""Smoke test for the E2 hybrid flag in the demo (run with CAPSTONE_T3_HYBRID unset; the script sets it itself where needed).

1. Default (flag off): Pipeline() returns the original Task 3 output (k-NN majority, no hybrid keys).
2. Flag on (Pipeline(hybrid_t3=True)): over real test images, the demo's Task 3 label equals the offline probe prediction (up to on-the-fly vs cached
   embedding noise) and its k-NN field equals the offline 5-NN; one disagreement case is printed.
3. The Gradio handler (demo/app.py run()) renders the neighbour-majority note only when the env var CAPSTONE_T3_HYBRID=1.
"""
import os
import subprocess
import sys

import numpy as np

from common import ART, DATA, ROOT, SPLIT_DIR, cc

sys.path.insert(0, str(ROOT / "demo"))
os.environ.pop("CAPSTONE_T3_HYBRID", None)
from pipeline import Pipeline  # noqa: E402

te = cc.load_split("damage", "test", DATA, SPLIT_DIR, verbose=False)[0]
N = 60
rows = te.iloc[:N]

off = Pipeline()                                    # 1. default
assert off.hybrid is None
res_off = [off.predict(r.img_path, r.tweet_text) for r in rows.itertuples()]
assert all("knn_label" not in r["t3"] and "probe_probs" not in r["t3"] for r in res_off)
print(f"1. default (flag off): hybrid is None; {N} predictions carry no hybrid keys; Task 3 label = k-NN majority")

on = Pipeline(hybrid_t3=True)                       # 2. flag on
res_on = [on.predict(r.img_path, r.tweet_text) for r in rows.itertuples()]
assert all("knn_label" in r["t3"] for r in res_on)
knn_same = sum(a["t3"]["label"] == b["t3"]["knn_label"] for a, b in zip(res_off, res_on))
print(f"2. flag on: demo k-NN field equals the flag-off label on {knn_same}/{N} rows (same index, same query path)")
assert knn_same == N
# offline probe on the cached test embeddings (the exact E2 evaluation path)
sys.path.insert(0, str(ROOT / "ablation"))
import faiss
V = cc.t3_vectors(cc.cached(ART / "emb" / "clip_damage_test.npy", te.image_id.tolist(), lambda: (_ for _ in ()).throw(RuntimeError())),
                  cc.cached(ART / "emb" / "sbert_damage_test.npy", te.image_id.tolist(), lambda: (_ for _ in ()).throw(RuntimeError())))[:N]
probs = on.hybrid.clf.predict_proba(V)
off_probe = [str(on.hybrid.clf.classes_[i]) for i in probs.argmax(1)]
same = sum(r["t3"]["label"] == p for r, p in zip(res_on, off_probe))
print(f"   demo hybrid label equals the offline probe prediction on {same}/{N} rows")
assert same >= N - 2          # on-the-fly image preprocessing can flip a borderline row
dis = [(r, t) for r, t in zip(res_on, rows.itertuples()) if not r["t3"]["neighbours_agree"]]
print(f"   neighbour majority disagrees with the probe on {len(dis)}/{N} of these rows")
r, t = dis[0]
print(f"   example: true {t.label!r} | probe {r['t3']['label']!r} | neighbour majority {r['t3']['knn_label']!r} | "
      f"neighbours {[(n['label'][:6], round(n['score'], 2)) for n in r['neighbours']]}")

# 3. the Gradio handler, in a subprocess per setting (app.py builds its Pipeline at import time)
code = ("import sys,os; sys.path.insert(0,r'%s'); import app; from PIL import Image; "
        "out=app.run(Image.open(r'%s'), %r); print('T3MD::' + out[3].replace(chr(10), ' | '))")
for flag in ("0", "1"):
    env = dict(os.environ, CAPSTONE_T3_HYBRID=flag, HF_HUB_OFFLINE="1", PYTHONWARNINGS="ignore")
    # use a row whose neighbours disagree with the probe so the note is visible when the flag is on
    t = dis[0][1]
    p = subprocess.run([sys.executable, "-c", code % (str(ROOT / "demo"), t.img_path, t.tweet_text)], capture_output=True, text=True, env=env, cwd=ROOT)
    line = [l for l in p.stdout.splitlines() if l.startswith("T3MD::")]
    assert line, p.stderr[-800:]
    print(f"3. app.run with CAPSTONE_T3_HYBRID={flag}: {line[0][6:]}")
    assert ("Neighbour majority" in line[0]) == (flag == "1")
print("SMOKE TEST PASSED")
