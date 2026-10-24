"""Seed-major T2 batch (Part A ladder + Part B variants), run sequentially on the GPU, resumable.

Tier 1: seed 42 for L0 L1 L2 L3 L3b L4a L4 + Part B (text_only image_only concat no_t2i) on L4a's config
Tier 2: seeds 43, 44 for L3 L3b L4a L4 + Part B
Tier 3: seed 43 for L0 L1 L2
Tier 4: seed 44 for L0 L1 L2

A job is skipped if its result JSON already exists (so the batch can be re-launched after an interruption). A crashed job is
retried once, then recorded as failed and the batch continues. Results below the majority-class baseline are listed in
results/ablation/ALERTS.txt (stop-and-ask condition in ABLATION_PLAN.md); they do not halt the overnight run.
Progress: results/ablation/batch_status.json; per-job logs: results/ablation/logs/.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results" / "ablation"
LOGS = RES / "logs"
PY = sys.executable
VARIANTS = ["text_only", "image_only", "concat", "no_t2i"]


def jobs_for(tier):
    ladder = lambda rungs, seed: [dict(tier=tier, part="A", name=f"t2_{r}", cfg=r, variant="full", seed=seed) for r in rungs]
    partb = lambda seed: [dict(tier=tier, part="B", name=f"t2_{v}", cfg="L4a", variant=v, seed=seed) for v in VARIANTS]
    if tier == 1:   # cheap rungs first so the key comparisons appear early
        return ladder(["L4", "L4a", "L3b"], 42) + partb(42) + ladder(["L3", "L2", "L1", "L0"], 42)
    if tier == 2:
        return [j for s in (43, 44) for j in ladder(["L4", "L4a", "L3b"], s) + partb(s) + ladder(["L3"], s)]
    if tier == 3:
        return ladder(["L2", "L1", "L0"], 43)
    if tier == 4:
        return ladder(["L2", "L1", "L0"], 44)
    raise ValueError(tier)


def result_path(j):
    return RES / j["part"] / f"{j['name']}_seed{j['seed']}.json"


def done(j):
    p = result_path(j)
    if not p.exists():
        return False
    d = json.loads(p.read_text())
    return not d.get("aborted") and "tier" in d


def run(j):
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{j['name']}_seed{j['seed']}.log"
    cmd = [PY, "-u", str(HERE / "t2_train.py"), "--config", j["cfg"], "--variant", j["variant"],
           "--seed", str(j["seed"]), "--tier", str(j["tier"])]
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONWARNINGS="ignore")
    with open(log, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=ROOT).returncode


def main():
    tiers = [int(t) for t in (sys.argv[1] if len(sys.argv) > 1 else "1,2,3,4").split(",")]
    status = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "jobs": {}}
    sp = RES / "batch_status.json"
    for tier in tiers:
        for j in jobs_for(tier):
            key = f"tier{tier}/{j['name']}_seed{j['seed']}"
            if done(j):
                status["jobs"][key] = "skipped (exists)"; continue
            t0 = time.time()
            rc = run(j)
            if rc != 0:
                rc = run(j)                                    # one retry
            ok = rc == 0 and done(j)
            info = {"ok": ok, "minutes": round((time.time() - t0) / 60, 1)}
            if ok:
                d = json.loads(result_path(j).read_text())
                info.update(test_macro_f1=round(d["test_macro_f1"], 4), test_acc=round(d["test_acc"], 4))
                if d.get("below_majority_baseline"):
                    with open(RES / "ALERTS.txt", "a") as f:
                        f.write(f"{key}: test macro-F1 {d['test_macro_f1']:.4f} < majority baseline {d['majority_baseline_macro_f1']:.4f}\n")
            status["jobs"][key] = info
            status["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            sp.write_text(json.dumps(status, indent=1))
            print(key, info, flush=True)
    status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    sp.write_text(json.dumps(status, indent=1))
    print("BATCH DONE", flush=True)


if __name__ == "__main__":
    main()
