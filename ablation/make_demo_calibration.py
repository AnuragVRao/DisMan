"""Write demo/calibration.json from the calibration results (results/ablation/X/c_calibration_*), so the demo's temperatures are data, not constants.

T2: L4a seed-42 temperature (applies only to that checkpoint). T3: the logistic probe's temperature. Both were fitted on the DEV split by
minimising NLL (ablation/c_calibration.py). A single temperature never changes the argmax, only the displayed confidence.
"""
import json

from common import RES, ROOT

t3 = json.load(open(RES / "X" / "c_calibration_t3_probe_seed42.json"))
t2 = json.load(open(RES / "X" / "c_calibration_t2_L4a_seed42.json"))
cfg = {
    "description": "Temperature scaling for the confidences shown in the demo (fitted on the dev split; argmax unchanged). Regenerate with ablation/make_demo_calibration.py.",
    "t2": {"checkpoint": "t2_L4a_seed42.pt", "temperature": t2["temperature"], "ece_before": t2["ece_before"], "ece_after": t2["ece_after"],
           "source": "results/ablation/X/c_calibration_t2_L4a_seed42.json",
           "note": "applies only to this checkpoint; any other T2 checkpoint is shown unscaled"},
    "t3_probe": {"temperature": t3["temperature"], "ece_before": t3["ece_before"], "ece_after": t3["ece_after"],
                 "source": "results/ablation/X/c_calibration_t3_probe_seed42.json"},
}
(ROOT / "demo" / "calibration.json").write_text(json.dumps(cfg, indent=1), encoding="utf8")
print(json.dumps({k: (v["temperature"] if isinstance(v, dict) else None) for k, v in cfg.items() if isinstance(v, dict)}))
