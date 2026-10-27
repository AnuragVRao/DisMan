"""G: POST-HOC sensitivity analysis of the pre-registered selection rule (ablation/SELECTION_RULE.md). POST-HOC: defined after the rule's outcome was known.

It does NOT change the headline decision (the pre-set rule did not adopt a new loss). It only shows how fragile the choice is by listing, side by side,
the winner each of four selection variants picks on each backbone. Reads results/ablation/G/g_imbalance_summary_seed42.json (written by
g_analyze_imbalance.py: dev macro-F1 and dev balanced accuracy per arm, tied sets from the paired dev bootstrap). Variants:
  (a) best mean dev macro-F1 (no tie handling)
  (b) the ORIGINAL rule: highest dev balanced accuracy among the arms tied with the best
  (c) simplest arm among the tied set, "closest to unweighted" by this fixed order (written here, before looking at the output):
      none < none_posthoc < focal1 < focal2 < sqrt_inv < eff_num < inv < focal2_inv < sampler < logit_adj
  (d) sqrt_inv as a midpoint reference (not a selection: it is shown with its dev score, its gap to the best arm and whether it is tied)
Test numbers are shown for information only (never used to pick). Writes results/ablation/G/g_POSTHOC_selection_sensitivity_seed42.json.
"""
import json

from common import RES, write_row

SIMPLICITY = ("none", "none_posthoc", "focal1", "focal2", "sqrt_inv", "eff_num", "inv", "focal2_inv", "sampler", "logit_adj")


def main():
    S = json.load(open(RES / "G" / "g_imbalance_summary_seed42.json"))["feats"]
    out = {}
    print("POST-HOC selection sensitivity (does not change the headline; characterises how fragile the choice is)\n")
    for feat, f in S.items():
        A = f["arms"]; tied = f["tied"]
        out[feat] = dict(
            a_best_dev_macro_f1=f["dev_best"],
            b_original_rule=f["selected"],
            c_simplest_among_tied=min(tied, key=SIMPLICITY.index),
            d_sqrt_inv_reference=dict(arm="sqrt_inv", dev_f1=A["sqrt_inv"]["dev_f1"], gap_to_best=A[f["dev_best"]]["dev_f1"] - A["sqrt_inv"]["dev_f1"],
                                      tied_with_best="sqrt_inv" in tied),
            tied_set=tied, tied_set_size=len(tied), n_arms=len(A))
        print(f"[{feat}] best dev arm {f['dev_best']}; tied set ({len(tied)} of {len(A)} arms): {tied}")
    print(f"\n{'variant':58s} {'B/32 winner':>14s} {'L/14 winner':>14s}  agree?")
    rows = [("(a) best mean dev macro-F1", "a_best_dev_macro_f1"), ("(b) ORIGINAL rule: top dev balanced acc among tied", "b_original_rule"),
            ("(c) simplest arm among tied (closest to unweighted)", "c_simplest_among_tied")]
    for name, key in rows:
        b, l = out["b32"][key], out["l14"][key]
        print(f"{name:58s} {b:>14s} {l:>14s}  {'yes' if b == l else 'NO'}")
    print(f"{'(d) sqrt_inv midpoint reference':58s} {'sqrt_inv':>14s} {'sqrt_inv':>14s}  (by construction)")
    for feat in ("b32", "l14"):
        d = out[feat]["d_sqrt_inv_reference"]
        print(f"      [{feat}] sqrt_inv dev macro-F1 {d['dev_f1']:.3f}, gap to the best arm {d['gap_to_best']:.3f}, tied with the best: {d['tied_with_best']}")
    print("\nFor information only (test, never used to select):")
    for feat in ("b32", "l14"):
        for name, key in rows:
            a = out[feat][key]
            print(f"   [{feat}] {name[:34]:34s} -> {a:12s} test macro-F1 {S[feat]['arms'][a]['test_f1']:.3f}  test balanced acc {S[feat]['arms'][a]['test_bal_acc']:.3f}")
    write_row("G", "g_POSTHOC_selection_sensitivity", 42, dict(task="T2", POSTHOC=True, simplicity_order=SIMPLICITY, feats=out))


if __name__ == "__main__":
    main()
