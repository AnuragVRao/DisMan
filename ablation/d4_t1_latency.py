"""Day 4 step 4: Task 1 latency, TF-IDF + LR vs fine-tuned BERTweet. 100 requests each after 5 warm-ups, cycling through 3 fixed test examples (demo/examples.json, else the first T1 test rows).
(a) model call only: text -> probabilities (TF-IDF: clean_tweet + predict_proba; BERTweet: tokenizer + forward + softmax). TF-IDF + LR has no GPU path, so its row is CPU for both columns.
(b) end-to-end request: Pipeline.predict(image, text), i.e. the whole cascade (CLIP, Task 2, Task 3, retrieval), with the Task 1 backend switched. Median and p95 in ms.
Writes results/ablation/D4_t1_latency.json. Run alone on the machine (no training job on the GPU)."""
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from PIL import Image  # noqa: E402

import crisis_common as cc  # noqa: E402
from pipeline import ART, DATA, T1_BERTWEET, Pipeline  # noqa: E402
from t1_bertweet import T1Bertweet  # noqa: E402

N, WARM = 100, 5


def examples():
    f = ROOT / "demo" / "examples.json"
    ex = json.load(open(f))[:3]
    return [(Image.open(DATA / e["rel_image"]).convert("RGB"), e["tweet_text"]) for e in ex]


def stats(ts):
    ts = sorted(ts); return dict(median_ms=1000 * statistics.median(ts), p95_ms=1000 * ts[int(0.95 * len(ts)) - 1], mean_ms=1000 * float(np.mean(ts)), n=len(ts))


def timeit(fn, ex, sync=False):
    for i in range(WARM):
        fn(ex[i % len(ex)])
    ts = []
    for i in range(N):
        a = ex[i % len(ex)]
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter(); fn(a)
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return stats(ts)


if __name__ == "__main__":
    ex = examples(); out = dict(gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, n_requests=N, warmup=WARM, n_examples=len(ex), model_call={}, end_to_end={})
    import joblib
    tfidf = joblib.load(ART / "task1_pipeline.joblib")
    out["model_call"]["tfidf_lr_cpu"] = timeit(lambda e: tfidf.predict_proba([cc.clean_tweet(e[1])]), ex)
    for dev in ("cuda", "cpu"):
        bw = T1Bertweet(T1_BERTWEET, torch.device(dev), 1.0)
        out["model_call"][f"bertweet_{dev}"] = timeit(lambda e: bw.predict(e[1]), ex, sync=True); del bw
    for dev in ("cuda", "cpu"):
        for be in ("tfidf", "bertweet"):
            P = Pipeline(device=dev, t1_backend=be)
            out["end_to_end"][f"{be}_{dev}"] = timeit(lambda e: P.predict(e[0], e[1]), ex, sync=True); del P
            if dev == "cuda":
                torch.cuda.empty_cache()
    (ROOT / "results" / "ablation" / "D4_t1_latency.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
