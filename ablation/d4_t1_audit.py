"""Day 4: (1) back-fill provenance keys into the Task 1 run JSONs made before d4_prov existed (metadata only: asserts that every pre-existing key keeps its parsed value and only
the provenance keys are added); (2) retroactive check, without rerunning anything: each T1 run's stored y_dev / y_test equals d4_t1_data.data()'s labels in the same order, and
d4_prov.assert_run passes. Prints pass/fail per run; writes results/ablation/D4_T1_audit.json."""
import json

import numpy as np

import d4_prov
import d4_t1_data as d4
import metrics_all as M
from common import RES

KEYS = ("task_name", "label_col", "train_split_sha256", "init_from", "feature_cache")


def backfill():
    done = []
    for p in sorted(M.RUN_DIR.glob("T1_*.json")):
        r = json.loads(p.read_text())
        if all(k in r for k in KEYS):
            continue
        name = r["name"]; init, feat = d4_prov.BACKFILL[name]
        new = d4_prov.prov("informative", init, feature=feat)
        r2 = dict(r, **new)
        assert all(r2[k] == r[k] for k in r), f"{p.name}: an existing value changed"
        assert set(r2) - set(r) <= set(KEYS), f"{p.name}: unexpected new keys"
        p.write_text(json.dumps(r2)); done.append(p.name)
    return done


def audit():
    D = d4.data(); out = {}
    for p in sorted(M.RUN_DIR.glob("T1_*.json")):
        r = json.loads(p.read_text()); ok, why = True, []
        for sp in ("dev", "test"):
            if not np.array_equal(np.array(r[f"y_{sp}"]), D["y"][sp]):
                ok = False; why.append(f"y_{sp} differs from data() labels")
        try:
            d4_prov.assert_run(r, "informative", p.name)
        except AssertionError as e:
            ok = False; why.append(str(e))
        out[p.stem] = dict(ok=ok, problems=why)
    return out


if __name__ == "__main__":
    print("back-filled:", len(backfill()), "run JSONs")
    res = audit()
    for k, v in res.items():
        print(f"{'PASS' if v['ok'] else 'FAIL'}  {k}  {'; '.join(v['problems'])}")
    print(f"{sum(v['ok'] for v in res.values())}/{len(res)} pass")
    (RES / "D4_T1_audit.json").write_text(json.dumps(res, indent=1))
    assert all(v["ok"] for v in res.values())
