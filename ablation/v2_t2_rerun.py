"""Re-run the ORIGINAL v2 Task 2 training by executing capstone-v2.ipynb's own cells (determinism check, ABLATION_PLAN.md).

Cells executed: 2 (config), 5 (splits), 12 (cached CLIP vectors), 14 (T2 prep), 15 (T2 training), 16 (T2 evaluation), with
CAPSTONE_TRAIN=1. Every write the notebook would make into artifacts/ or results/ is redirected to
artifacts/ablation/v2_rerun/ and results/ablation/v2_rerun/ (each patch is asserted to apply exactly once), so v2's files
are never touched. The model code, seed handling and training loop are the notebook's own, unmodified.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ["CAPSTONE_TRAIN"] = "1"
seed_dir = ROOT / "artifacts" / "ablation" / "v2_rerun"
res_dir = ROOT / "results" / "ablation" / "v2_rerun"
seed_dir.mkdir(parents=True, exist_ok=True); res_dir.mkdir(parents=True, exist_ok=True)

nb = json.load(open(ROOT / "capstone-v2.ipynb", encoding="utf8"))
cells = {i: "".join(c["source"]) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"}

PATCHES = {
    14: [('if TRAIN or not (ART / "tokenizer").exists():', 'if False:')],           # do not re-save tokenizer/config into artifacts/
    15: [('torch.save(best_state, ART / "task2_model.pt")', f'torch.save(best_state, Path(r"{seed_dir}") / "task2_model.pt")'),
         ('open(ART / "task2_labels.json", "w")', f'open(Path(r"{seed_dir}") / "task2_labels.json", "w")'),
         ('open(RES / "task2_history.json", "w")', f'open(Path(r"{res_dir}") / "task2_history.json", "w")')],
    16: [('t2_out.to_csv(RES / "preds_task2_test.csv", index=False)', f't2_out.to_csv(Path(r"{res_dir}") / "preds_task2_test.csv", index=False)')],
}
ns = {"__name__": "__main__"}
t0 = time.time()
for i in (2, 5, 12, 14, 15, 16):
    src = cells[i]
    for old, new in PATCHES.get(i, []):
        assert src.count(old) == 1, f"patch not applied exactly once in cell {i}: {old}"
        src = src.replace(old, new)
    if i == 16:
        ns["RESULTS"] = {}                 # normally created by the Task 1 cell, which is not executed here
    exec(compile(src, f"capstone-v2.ipynb[cell {i}]", "exec"), ns)

m = ns["RESULTS"]["T2_test"]
(res_dir / "v2_rerun_metrics.json").write_text(json.dumps(dict(T2_test=m, T2_dev=ns["RESULTS"]["T2_dev"],
                                                               wall_clock_s=time.time() - t0), indent=1))
print("V2_RERUN T2 test:", {k: round(float(v), 4) for k, v in m.items()}, f"| wall {(time.time()-t0)/60:.1f} min")
