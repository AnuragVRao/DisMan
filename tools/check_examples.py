"""Run the preloaded examples through the pipeline and list correct/wrong per task (ground truth only for display)."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, "demo")
from pipeline import DATA, Pipeline

t0 = time.time(); P = Pipeline(); print(f"load {time.time()-t0:.1f}s on {P.device}")
ex = json.load(open("demo/examples.json"))
P.predict(DATA / ex[0]["rel_image"], ex[0]["tweet_text"])  # warm-up (CUDA kernels), not timed
res = []
for e in ex:
    r = P.predict(DATA / e["rel_image"], e["tweet_text"]); res.append(r)
    ok = lambda p, t: "n/a " if t == "n/a" else ("OK  " if p == t else "WRONG")
    print(f"\n{e['event']} | {e['tweet_text'][:60]!r}  [{r['latency_s']:.2f}s]")
    print(f"  T1 pred {r['t1']['label']:16s} (p={r['t1']['prob']:.2f}) true {e['t1_true']:16s} {ok(r['t1']['label'], e['t1_true'])}")
    print(f"  T2 pred {r['t2']['label']:42s} true {e['t2_true']:42s} {ok(r['t2']['label'], e['t2_true'])}")
    print(f"  T3 pred {r['t3']['label']:20s} share {max(r['t3']['vote_share'].values()):.1f} true {e['t3_true']:20s} {ok(r['t3']['label'], e['t3_true'])}")
    print("  neighbours:", [(n['label'][:6], round(n['score'], 2)) for n in r['neighbours']])
    assert all(Path(n["image_path"]).exists() for n in r["neighbours"])
lat = [r["latency_s"] for r in res]
print(f"\nlatency (warm): mean {sum(lat)/len(lat):.2f}s, max {max(lat):.2f}s over {len(lat)} examples")
