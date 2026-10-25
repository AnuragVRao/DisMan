"""D follow-up (CPU only, no training): per-class pooled F1, T2 leave-one-disaster-out vs in-distribution, seed 42, the 2,237 pooled test rows.

Uses the per-row predictions stored in results/ablation/X/d_lodo_t2_<event>_seed42.json (`pred` = LODO, `indist_pred` = the existing image-only checkpoint),
reassembled in test order. Reports per-class F1 for both, the difference, the class's test support and the fraction of its test rows that come from the two
earthquake events (iraq_iran_earthquake, mexico_earthquake), plus each class's share of the total macro-F1 drop. Writes results/ablation/X/d_lodo_t2_perclass_seed42.json.
"""
import json

import numpy as np
from sklearn.metrics import f1_score

from common import DATA, RES, SPLIT_DIR, cc, write_row

EQ = ("iraq_iran_earthquake", "mexico_earthquake")
tr = cc.load_split("humanitarian", "train", DATA, SPLIT_DIR, verbose=False)[0]
te = cc.load_split("humanitarian", "test", DATA, SPLIT_DIR, verbose=False)[0]
labels = sorted(tr["label"].unique()); l2i = {l: i for i, l in enumerate(labels)}; ALL = list(range(len(labels)))
y = np.array([l2i[l] for l in te["label"]])
lodo, indist = np.full(len(y), -1), np.full(len(y), -1)
for e in sorted(te.event_name.unique()):
    d = json.load(open(RES / "X" / f"d_lodo_t2_{e}_seed42.json"))
    m = (te.event_name == e).to_numpy()
    assert m.sum() == len(d["pred"]) == len(d["indist_pred"])
    lodo[m], indist[m] = d["pred"], d["indist_pred"]
assert (lodo >= 0).all() and (indist >= 0).all()
ov = json.load(open(RES / "X" / "d_lodo_t2_overall_seed42.json"))["bootstrap_lodo_minus_indist_seed42"]
f_l = f1_score(y, lodo, labels=ALL, average=None, zero_division=0); f_i = f1_score(y, indist, labels=ALL, average=None, zero_division=0)
assert abs(f_l.mean() - ov["lodo_macro_f1"]) < 1e-12 and abs(f_i.mean() - ov["indist_macro_f1"]) < 1e-12      # reproduces the pooled bootstrap inputs

eq_mask = te.event_name.isin(EQ).to_numpy()
drop_total = float((f_i - f_l).sum())
rows = []
for c, name in enumerate(labels):
    sup = int((y == c).sum()); eq = int(((y == c) & eq_mask).sum())
    rows.append(dict(cls=name, support=sup, support_earthquake=eq, earthquake_fraction=eq / sup, f1_lodo=float(f_l[c]), f1_indist=float(f_i[c]),
                     diff=float(f_l[c] - f_i[c]), share_of_total_drop=float((f_i[c] - f_l[c]) / drop_total)))
rows_s = sorted(rows, key=lambda r: r["diff"])
print(f"{'class':42s} {'support':>7s} {'eq rows':>7s} {'eq frac':>7s} {'F1 LODO':>8s} {'F1 ID':>7s} {'diff':>7s} {'share of drop':>13s}")
for r in rows:
    print(f"{r['cls']:42s} {r['support']:7d} {r['support_earthquake']:7d} {r['earthquake_fraction']:7.3f} {r['f1_lodo']:8.3f} {r['f1_indist']:7.3f} {r['diff']:+7.3f} {r['share_of_total_drop']:13.1%}")
print(f"macro-F1 {f_l.mean():.4f} vs {f_i.mean():.4f}; diff {f_l.mean() - f_i.mean():+.4f}; sum of per-class drops {drop_total:.4f}")
mostly_eq = [r for r in rows if r["earthquake_fraction"] > 0.5]
print(f"classes with more than half of their test rows from the two earthquakes: {[(r['cls'], r['support']) for r in mostly_eq]}; "
      f"their share of the total drop: {sum(r['share_of_total_drop'] for r in mostly_eq):.1%}")
print(f"all test rows from the two earthquakes: {int(eq_mask.sum())} of {len(y)} ({eq_mask.mean():.1%})")
corr = float(np.corrcoef([r["earthquake_fraction"] for r in rows], [r["diff"] for r in rows])[0, 1])
print(f"Pearson correlation across the 8 classes: earthquake fraction vs F1 difference = {corr:+.3f}")
write_row("X", "d_lodo_t2_perclass", 42, dict(task="T2", tier="cpu", n_test=int(len(y)), earthquake_events=list(EQ), earthquake_rows=int(eq_mask.sum()),
                                              classes=rows, macro_f1_lodo=float(f_l.mean()), macro_f1_indist=float(f_i.mean()), sum_of_class_drops=drop_total,
                                              classes_mostly_earthquake=[r["cls"] for r in mostly_eq],
                                              share_of_drop_mostly_earthquake=float(sum(r["share_of_total_drop"] for r in mostly_eq)),
                                              pearson_eqfrac_vs_diff=corr))
