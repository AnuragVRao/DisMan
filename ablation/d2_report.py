"""Day 2, step 5: the single comparison table and the rule's pick per decision (ablation/SELECTION_RULE.md, Day 2 addendum). Reads results/ablation/H/ only.

Per arm (3 seeds): dev macro-F1 (mean +- seed std), test macro-F1 (mean +- seed std) and its TEST-SAMPLING 95% CI (reporting only, never used to select; a different quantity from the seed std),
test balanced accuracy, test recall of injured-or-dead (n = 10 test rows) and vehicle damage (n = 36), trainable parameters, backbone size, and from the dev bootstrap: best arm minus this arm,
tied or not, and the rule's pick. Labels follow the pre-registered rule; no multiplicity correction. Writes results/ablation/H/report_seed42.json and prints a markdown table.
"""
import json

import numpy as np

import bootstrap_utils as bu
from common import write_row
from d2_common import RES, SEEDS, data
from d2_select import ARMS

TITLE = {"d1": "D1 image backbone (image-only head, d_model 256, dropout 0.2, lr 1e-4)", "d2": "D2 text encoder (text-only head on the frozen sentence embedding)",
         "d3": "D3 head (D1 and D2 winners: L/14 + MiniLM)", "d4": "D4 hyperparameters (D3 winner; fixed 27-configuration grid)"}


def duplicates():
    """arm -> the first arm (in decision order D1..D4) with identical dev and test predictions on all seeds; such arms are the SAME runs and are not separate arms."""
    seen, canon = {}, {}
    for d in ("d1", "d2", "d3", "d4"):
        for a in ARMS[d]:
            key = tuple((tuple(r["pred_dev"]), tuple(r["pred_test"])) for r in (json.load(open(RES / "H" / f"{a}_seed{s}.json")) for s in SEEDS))
            canon[a] = seen.setdefault(key, a)
    return canon


def main():
    D = data(); yt = D["y"]["test"].numpy(); K = D["nc"]
    canon = duplicates(); n_slots = sum(len(v) for v in ARMS.values()); n_distinct = len(set(canon.values()))
    assert n_slots == 38 and n_distinct == 35, (n_slots, n_distinct)
    tidx = bu.resample_idx(len(yt), 10000)
    out, md = {}, []
    for d in ("d1", "d2", "d3", "d4"):
        sel = json.load(open(RES / "H" / f"selection_{d}_seed42.json"))
        md += [f"\n### {TITLE[d]}\n", "| Arm | Dev macro-F1 | Test macro-F1 (seed std) | Test-sampling 95% CI | Test bal. acc. | Test recall injured (n=10) | Test recall vehicle (n=36) | Trainable params | Backbone (M) | Dev: best - arm [95% CI] | Tied | Rule's pick | Note |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        rows = {}
        for a in sorted(ARMS[d], key=lambda a: -sel["mean_dev_macro_f1"][a]):
            R = [json.load(open(RES / "H" / f"{a}_seed{s}.json")) for s in SEEDS]
            tm = np.array([r["test_macro_f1"] for r in R]); pts = {s: np.array(r["pred_test"]) for s, r in zip(SEEDS, R)}
            vec = {s: bu.macro_f1_vec(yt, pts[s], K, tidx) for s in SEEDS}
            ci = bu.mean_ci(vec, point=float(tm.mean()))
            bd = sel["best_minus_arm"].get(a)
            cell = "best" if a == sel["best"] else f"{bd['diff']:+.4f} [{bd['ci95_low']:+.4f}, {bd['ci95_high']:+.4f}]"
            row = dict(dev=sel["mean_dev_macro_f1"][a], dev_std=sel["seed_std_dev"][a], test=float(tm.mean()), test_std=float(tm.std(ddof=1)), ci=[ci["ci95_low"], ci["ci95_high"]],
                       bal=float(np.mean([r["test_balanced_acc"] for r in R])), inj=float(np.mean([r["recall_injured"] for r in R])), veh=float(np.mean([r["recall_vehicle"] for r in R])),
                       params=R[0]["n_trainable"], backbone_m=R[0]["backbone_params_m"], best_minus_arm=bd, tied=a in sel["tied"], pick=a == sel["pick"],
                       missing_recall=max(r["recall_missing"] for r in R), n_test_injured=R[0]["n_test_injured"], n_test_vehicle=R[0]["n_test_vehicle"])
            assert row["n_test_injured"] == 10 and row["n_test_vehicle"] == 36 and row["missing_recall"] == 0.0
            rows[a] = row
            md.append(f"| {a} | {row['dev']:.4f} ± {row['dev_std']:.4f} | {row['test']:.4f} ± {row['test_std']:.4f} | [{ci['ci95_low']:.3f}, {ci['ci95_high']:.3f}] | {row['bal']:.3f} | {row['inj']:.3f} | {row['veh']:.3f} | "
                      f"{row['params']:,} | {row['backbone_m']:.1f} | {cell} | {'yes' if row['tied'] else 'no'} | {'**PICK**' if row['pick'] else ''} | {('same runs as ' + canon[a] + ' (not a separate arm)') if canon[a] != a else ''} |")
        md.append(f"\nRule's pick for {d.upper()}: **{sel['pick']}** ({len(sel['tied'])} of {len(ARMS[d])} arms tied with the best on dev: {', '.join(sel['tied'])})")
        out[d] = dict(pick=sel["pick"], best=sel["best"], tied=sel["tied"], rows=rows)
    text = "\n".join(md)
    print(text)
    write_row("H", "report", 42, dict(task="T2", n_boot_test=10000, picks={d: out[d]["pick"] for d in out}, decisions=out, duplicates={a: c for a, c in canon.items() if a != c},
                                      arm_slots=n_slots, distinct_configurations=n_distinct))
    open(RES / "H" / "report_table.md", "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    main()
