"""Hybrid Task 3 (ablation E2). A logistic probe on the fused CLIP+SBERT embedding predicts the damage label; the k-NN search still supplies
the retrieved similar past cases, and the neighbour majority is returned next to the probe's label so disagreements are visible.

Default OFF in the demo (set CAPSTONE_T3_HYBRID=1). The probe is fitted on the vectors already stored in the Task 3 FAISS index
(damage-train minus tweet siblings of the test split) with the same settings as the ablation's probe: class_weight="balanced",
C chosen on the development split (results/ablation/B/t3_lr_fused_seed42.json: C = 1). Probabilities are NOT calibrated.
"""
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import crisis_common as cc  # noqa: E402

PROBE_C = 1.0


class HybridT3:
    def __init__(self, index, labels, C=PROBE_C):
        X = index.reconstruct_n(0, index.ntotal)                 # the fused vectors stored in the index, in meta order
        self.clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(X, [str(l) for l in labels])

    def predict(self, q, neighbour_labels, neighbour_scores):
        """q: [1, 896] fused query vector; neighbour_*: the k retrieved cases (labels, similarities)."""
        probs = self.clf.predict_proba(q)[0]
        label = str(self.clf.classes_[int(probs.argmax())])
        knn_label, share = cc.knn_vote(list(neighbour_labels), list(neighbour_scores))
        return dict(label=label, probs=dict(zip(map(str, self.clf.classes_), map(float, probs))),
                    knn_label=knn_label, agrees=bool(label == knn_label), vote_share=share)
