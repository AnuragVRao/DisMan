"""E2: hybrid Task 3 on the leakage-free split (CPU). The logistic probe on the fused embedding predicts; the 5-NN search supplies the evidence.

Runs the same HybridT3 class the demo uses (demo/hybrid.py) on the saved v2 index (damage-train minus tweet siblings: 2,361 rows) with the
cached test embeddings. Reports test metrics, the rate at which the 5-NN majority disagrees with the probe (and each side's accuracy on the
agreeing / disagreeing queries), and the paired bootstrap of the hybrid's macro-F1 against v2's 5-NN. Because the hybrid's label IS the
probe's label, its row must equal the probe row of Part B / E1; this is asserted. Writes results/ablation/X/e2_hybrid_t3_seed42.json.
"""
import json
import os
import sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import faiss
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression

from bootstrap_utils import boot, confusion, macro_f1
from common import ART, DATA, RES, ROOT, SPLIT_DIR, cc, write_row

sys.path.insert(0, str(ROOT / "demo"))
from hybrid import HybridT3  # noqa: E402

faiss.omp_set_num_threads(4)


def no_miss():
    raise RuntimeError("cache miss")


te = cc.load_split("damage", "test", DATA, SPLIT_DIR, verbose=False)[0]
V = cc.t3_vectors(cc.cached(ART / "emb" / "clip_damage_test.npy", te.image_id.tolist(), no_miss),
                  cc.cached(ART / "emb" / "sbert_damage_test.npy", te.image_id.tolist(), no_miss))
index = faiss.read_index(str(ART / "task3_index.faiss"))
meta = pd.read_csv(ART / "task3_meta.csv", dtype=str, keep_default_na=False)
assert index.ntotal == len(meta) == 2361
hyb = HybridT3(index, meta["label"])
scores, idx = index.search(V, 5)

out = [hyb.predict(V[i:i + 1], meta["label"].to_numpy()[idx[i]], scores[i]) for i in range(len(V))]
y = te["label"].to_numpy(); hyb_lab = np.array([o["label"] for o in out]); knn_lab = np.array([o["knn_label"] for o in out])
L = {l: i for i, l in enumerate(cc.T3_LABELS)}
yi, hi, ki = (np.array([L[x] for x in a]) for a in (y, hyb_lab, knn_lab))

# the hybrid IS the probe: its predictions must equal Part B's probe row exactly
stored_probe = json.load(open(RES / "B" / "t3_lr_fused_seed42.json"))
m = cc.metrics(y, hyb_lab, cc.T3_LABELS)
assert abs(m["macro_f1"] - stored_probe["test_macro_f1"]) < 1e-12 and abs(m["acc"] - stored_probe["test_acc"]) < 1e-12, "hybrid != stored probe row"
assert stored_probe["C"] == 1.0
stored_knn = json.load(open(RES / "B" / "t3_fused_k5_majority_seed42.json"))
mk = cc.metrics(y, knn_lab, cc.T3_LABELS)
assert abs(mk["macro_f1"] - stored_knn["test_macro_f1"]) < 1e-12 and abs(mk["acc"] - 0.6673) < 5e-5            # = v2's 5-NN headline

disagree = hyb_lab != knn_lab
pcf = f1_score(y, hyb_lab, labels=cc.T3_LABELS, average=None, zero_division=0)
row_boot = boot("t3_hybrid_vs_knn_fused", yi, hi, ki, 3, [0, 1, 2], "macro_f1", write=False)
e1 = json.load(open(RES / "X" / "e1_t3_probe_vs_knn_fused_seed42.json"))
for k in ("diff", "ci95_low", "ci95_high", "score_a", "score_b"):
    assert abs(row_boot[k] - e1[k]) < 1e-12, f"hybrid bootstrap {k} differs from the E1 probe row"
print("hybrid predictions, metrics and bootstrap are identical to the E1 / Part B probe row (asserted)")

n = len(y)
rowj = dict(task="T3", tier="cpu", n_test=n, index_rows=int(index.ntotal), probe_C=1.0, k=5,
            test_acc=m["acc"], test_macro_f1=m["macro_f1"], test_weighted_f1=m["weighted_f1"],
            per_class_f1=dict(zip(cc.T3_LABELS, map(float, pcf))), confusion_matrix=confusion_matrix(y, hyb_lab, labels=cc.T3_LABELS).tolist(),
            knn_test_acc=mk["acc"], knn_test_macro_f1=mk["macro_f1"], knn_test_weighted_f1=mk["weighted_f1"],
            disagreement_rate=float(disagree.mean()), n_disagree=int(disagree.sum()),
            probe_acc_where_agree=float((hyb_lab == y)[~disagree].mean()), probe_acc_where_disagree=float((hyb_lab == y)[disagree].mean()),
            knn_acc_where_disagree=float((knn_lab == y)[disagree].mean()),
            bootstrap_hybrid_minus_knn=dict(diff=row_boot["diff"], ci95_low=row_boot["ci95_low"], ci95_high=row_boot["ci95_high"],
                                            n_boot=row_boot["n_boot"], excludes_zero=row_boot["ci_excludes_zero"]),
            equals_probe_row=True)
write_row("X", "e2_hybrid_t3", 42, rowj)
print(f"hybrid (probe label) : acc {m['acc']:.4f} macro-F1 {m['macro_f1']:.4f} weighted-F1 {m['weighted_f1']:.4f}")
print(f"v2 5-NN              : acc {mk['acc']:.4f} macro-F1 {mk['macro_f1']:.4f} weighted-F1 {mk['weighted_f1']:.4f}")
print(f"neighbour majority disagrees with the probe on {int(disagree.sum())} of {n} queries ({100 * disagree.mean():.1f}%)")
print(f"  where they AGREE ({int((~disagree).sum())}): accuracy {rowj['probe_acc_where_agree']:.3f}")
print(f"  where they DISAGREE ({int(disagree.sum())}): probe accuracy {rowj['probe_acc_where_disagree']:.3f} | 5-NN accuracy {rowj['knn_acc_where_disagree']:.3f}")
print(f"bootstrap hybrid - 5-NN macro-F1: {row_boot['diff']:+.4f}  95% CI [{row_boot['ci95_low']:+.4f}, {row_boot['ci95_high']:+.4f}]")
