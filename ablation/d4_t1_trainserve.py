"""Day 4 train/serve check for the deployed Task 1 model: 20 fixed-seed T1 test tweets go through the demo's own code path (Pipeline.predict, real image) and T1Bertweet.predict,
and are compared with the stored bertweet_deploy run (test probs are softmax of the unscaled logits; the demo applies the dev temperature on top)."""
import json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo")); sys.path.insert(0, str(ROOT / "ablation")); sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import d4_t1_data as d4
import metrics_all as M
from pipeline import Pipeline

D = d4.data(); S = D["S"]["test"]; run = M.load_run("T1", "bertweet_deploy", 42)
assert (run["y_test"] == D["y"]["test"]).all()
idx = np.random.default_rng(0).choice(len(S), 20, replace=False)
P = Pipeline(device="cuda", t1_backend="bertweet"); T = P.t1_bw.T
stored_raw = np.array(run["probs_test"])                                   # softmax(logits), unscaled
stored_cal = M.apply_temperature(np.log(np.clip(stored_raw, 1e-9, 1)), T)
res = dict(temperature=T, text_column_trained_on="tweet_text (raw, BERTweet normalisation)", text_passed_by_demo="the raw tweet string (examples.json tweet_text / user textbox)", rows=[])
for i in idx:
    r = S.iloc[i]; p_cal_demo = np.array([P.predict(Image.open(r.img_path).convert("RGB"), r.tweet_text)["t1"]["probs"][l] for l in d4.LABELS])
    p_cal2, p_raw2 = P.t1_bw.predict(r.tweet_text)
    res["rows"].append(dict(i=int(i), image_id=r.image_id, argmax_stored_cal=int(stored_cal[i].argmax()), argmax_demo_cal=int(p_cal_demo.argmax()), argmax_stored_raw=int(stored_raw[i].argmax()),
                            argmax_demo_raw=int(p_raw2.argmax()), max_abs_diff_raw=float(abs(p_raw2 - stored_raw[i]).max()), max_abs_diff_cal=float(abs(p_cal_demo - stored_cal[i]).max())))
R = res["rows"]
res["summary"] = dict(n=len(R), argmax_match_calibrated=sum(r["argmax_stored_cal"] == r["argmax_demo_cal"] for r in R), argmax_match_unscaled=sum(r["argmax_stored_raw"] == r["argmax_demo_raw"] for r in R),
                      max_abs_prob_diff_unscaled=max(r["max_abs_diff_raw"] for r in R), median_abs_prob_diff_unscaled=float(np.median([r["max_abs_diff_raw"] for r in R])), max_abs_prob_diff_calibrated=max(r["max_abs_diff_cal"] for r in R))
(ROOT / "results" / "ablation" / "D4_t1_trainserve.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res["summary"], indent=1))
