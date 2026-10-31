"""Day 4 latency reconciliation, part (b): full HTTP round trip through a RUNNING Gradio server (demo/app.py), via gradio_client, for the TF-IDF and the BERTweet Task 1 backend.
Each request uploads the example image, sends the tweet text, and waits for all outputs, including the 5 neighbour gallery images (gradio_client downloads output files).
30 timed requests after 3 warm-ups, cycling through 3 fixed examples. Merges the result into results/ablation/D4_t1_latency.json and adds a `what_is_measured` field to every row."""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "ablation" / "D4_t1_latency.json"
N, WARM = 30, 3
PORT = {"tfidf": 7861, "bertweet": 7862}


def stats(ts):
    ts = sorted(ts); return dict(median_ms=1000 * float(np.median(ts)), p95_ms=1000 * ts[int(0.95 * len(ts)) - 1], mean_ms=1000 * float(np.mean(ts)), n=len(ts))


def measure(backend):
    from gradio_client import Client, handle_file
    env = dict(os.environ, CAPSTONE_T1=backend, GRADIO_SERVER_PORT=str(PORT[backend]), HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8")
    log = open(ROOT / "results" / "ablation" / f"d4_t1_http_{backend}.log", "w")
    srv = subprocess.Popen([sys.executable, str(ROOT / "demo" / "app.py")], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{PORT[backend]}"
    try:
        for _ in range(240):
            try:
                if urllib.request.urlopen(url, timeout=2).status == 200:
                    break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("server did not start")
        client = Client(url, verbose=False)
        ex = json.load(open(ROOT / "demo" / "examples.json"))[:3]
        sys.path.insert(0, str(ROOT / "demo"))
        data = [(str(ROOT / "data" / "CrisisMMD_v2.0" / e["rel_image"]), e["tweet_text"]) for e in ex]
        call = lambda a: client.predict(handle_file(a[0]), a[1], api_name="/run")
        for i in range(WARM):
            call(data[i % 3])
        ts = []
        for i in range(N):
            t0 = time.perf_counter(); out = call(data[i % 3]); ts.append(time.perf_counter() - t0)
            assert len(out) == 7 and out[5], "unexpected handler outputs / empty gallery"
        return stats(ts)
    finally:
        srv.terminate()
        try:
            srv.wait(15)
        except Exception:
            srv.kill()


if __name__ == "__main__":
    d = json.loads(OUT.read_text())
    d["http_round_trip"] = {f"{b}_gpu_server": measure(b) for b in ("tfidf", "bertweet")}
    what = dict(
        model_call="text -> Task 1 probabilities only (TF-IDF: clean_tweet + predict_proba; BERTweet: tokenizer + forward + softmax); in-process, no image, no other tasks, no UI",
        end_to_end="Pipeline.predict(image, text) in-process: CLIP image features, Task 1, Task 2, Task 3 with retrieval; excludes the Gradio handler, JSON/HTTP, file upload and gallery image transfer (the 'handler only' number)",
        http_round_trip="gradio_client -> running demo/app.py (GPU) -> all outputs incl. the 5 gallery images downloaded; includes image upload, Gradio queue, handler and file transfer")
    for key, w in what.items():
        for row in d[key].values():
            row["what_is_measured"] = w
    d["note"] = ("The numbers in README.md / report SIV-E ('0.1-0.2 s model call, ~1.8 s end to end') have no benchmark script or log in the repo (first stated in the README, commit e1afcb4, 2026-10-22); "
                 "what they measured is not recorded.")
    OUT.write_text(json.dumps(d, indent=1))
    for k, v in d["http_round_trip"].items():
        print(f"{k:22s} median {v['median_ms']:7.1f} ms  p95 {v['p95_ms']:7.1f} ms  mean {v['mean_ms']:7.1f} ms  (n={v['n']})")
