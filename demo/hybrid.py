"""Hybrid Task 3 (ablation E2). A logistic probe on the fused CLIP+SBERT embedding predicts the damage label; the k-NN search still supplies
the retrieved similar past cases, and the neighbour majority is returned next to the probe's label so disagreements are visible.

Default ON in the demo (CAPSTONE_T3_HYBRID=0 restores v2's 5-NN vote). The probe is fitted on the vectors already stored in the Task 3 FAISS index
(damage-train minus tweet siblings of the test split) with the same settings as the ablation's probe: class_weight="balanced",
C chosen on the development split (results/ablation/B/t3_lr_fused_seed42.json: C = 1). The displayed probabilities are temperature-scaled with the
temperature in demo/calibration.json (fitted on dev; test ECE 0.077 -> 0.038); the predicted label does not depend on the temperature.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import crisis_common as cc  # noqa: E402

PROBE_C = 1.0


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


class HybridT3:
    def __init__(self, index, labels, C=PROBE_C, temperature=1.0):
        X = index.reconstruct_n(0, index.ntotal)                 # the fused vectors stored in the index, in meta order
        self.clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(X, [str(l) for l in labels])
        self.temperature = float(temperature)                    # demo/calibration.json (fitted on dev); 1.0 = unscaled

    def predict(self, q, neighbour_labels, neighbour_scores):
        """q: [1, 896] fused query vector; neighbour_*: the k retrieved cases (labels, similarities).

        The displayed probabilities are temperature-scaled (softmax(logits / T)); the argmax, hence the label, is the same for any T > 0."""
        z = self.clf.decision_function(q)[0]
        raw, probs = _softmax(z), _softmax(z / self.temperature)
        label = str(self.clf.classes_[int(probs.argmax())])
        assert label == str(self.clf.classes_[int(raw.argmax())])
        knn_label, share = cc.knn_vote(list(neighbour_labels), list(neighbour_scores))
        cls = list(map(str, self.clf.classes_))
        return dict(label=label, probs=dict(zip(cls, map(float, probs))), probs_unscaled=dict(zip(cls, map(float, raw))),
                    confidence=float(probs.max()), knn_label=knn_label, agrees=bool(label == knn_label), vote_share=share)
