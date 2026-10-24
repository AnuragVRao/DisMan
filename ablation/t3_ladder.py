"""Task 3 v1 -> v2 ladder (ABLATION_PLAN.md, Part A). No training, one deterministic run per rung.

Every rung is evaluated on the same queries: the 529-row corrected damage TEST split, scored on the 3 real classes.
Embeddings: HF CLIP ViT-B/32 (QuickGELU) + SBERT all-MiniLM-L6-v2, fused and re-normalised, exact inner-product FAISS, k=5,
v2's vote (cc.knn_vote: vote share, ties broken by summed similarity; label order Severe/Mild/Little[/Don't know]).

  L0  all 18,082 annotation rows (incl. the test images), v1 labels (14,556 rows labelled "Don't know")
  L1  test rows removed                                                          (17,553)
  L2  "Don't know" rows removed (= 14,455 unlabelled + 101 explicit): train+dev  (2,997)
  L3  dev rows removed: damage train only                                        (2,468)
  L4  train rows sharing a tweet_id with the test split removed (= v2 index)     (2,361)
"""
import faiss
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from common import ABL_EMB, DONT_KNOW, T3_LABELS_V1, cc, load_annotations, write_row

K = 5
ann = load_annotations()
V = cc.t3_vectors(np.load(ABL_EMB / "clip_annot_hf.npy"), np.load(ABL_EMB / "sbert_annot.npy"))
assert len(V) == len(ann)

test_mask = (ann.split == "test").to_numpy()
tr_mask = (ann.split == "train").to_numpy()
dev_mask = (ann.split == "dev").to_numpy()
dk = (ann.label_v1 == DONT_KNOW).to_numpy()
sibling = ann.tweet_id.isin(set(ann.tweet_id[test_mask])).to_numpy() & tr_mask
test_idx = np.where(test_mask)[0]

RUNGS = {
    "L0": np.ones(len(ann), bool),
    "L1": ~test_mask,
    "L2": ~test_mask & ~dk,
    "L3": tr_mask,
    "L4": tr_mask & ~sibling,
}
assert [int(m.sum()) for m in RUNGS.values()] == [18082, 17553, 2997, 2468, 2361], "unexpected rung sizes"

y_true = ann.label_v1.to_numpy()[test_idx]
assert set(y_true) <= set(cc.T3_LABELS)
Q = V[test_idx]
tweet = ann.tweet_id.to_numpy(); imgid = ann.image_id.to_numpy()

for name, mask in RUNGS.items():
    rows = np.where(mask)[0]
    index = faiss.IndexFlatIP(V.shape[1]); index.add(V[rows])
    scores, nn_ = index.search(Q, K)
    nn_rows = rows[nn_]                                   # positions in the annotation table
    labs = ann.label_v1.to_numpy()[nn_rows]
    pred = np.array([cc.knn_vote(list(l), list(s), T3_LABELS_V1)[0] for l, s in zip(labs, scores)])
    m = cc.metrics(y_true, pred, cc.T3_LABELS)
    self_in_topk = int((imgid[nn_rows] == imgid[test_idx][:, None]).any(1).sum())
    self_top1 = int((imgid[nn_rows[:, 0]] == imgid[test_idx]).sum())
    sib_in_topk = int(((tweet[nn_rows] == tweet[test_idx][:, None]) & (imgid[nn_rows] != imgid[test_idx][:, None])).any(1).sum())
    dk_votes = int((labs == DONT_KNOW).sum())
    per_class = f1_score(y_true, pred, labels=cc.T3_LABELS, average=None, zero_division=0)
    cm = confusion_matrix(y_true, pred, labels=T3_LABELS_V1)
    row = dict(task="T3", tier="cpu", config=f"t3_{name}", index_rows=int(mask.sum()), n_test=len(test_idx),
               test_acc=m["acc"], test_macro_f1=m["macro_f1"], test_weighted_f1=m["weighted_f1"],
               per_class_f1=dict(zip(cc.T3_LABELS, map(float, per_class))), confusion_matrix=cm.tolist(),
               confusion_labels=T3_LABELS_V1, predicted_dont_know=int((pred == DONT_KNOW).sum()),
               test_queries_with_self_in_top5=self_in_topk, test_queries_with_self_top1=self_top1,
               test_queries_with_same_tweet_sibling_in_top5=sib_in_topk,
               neighbour_votes_for_dont_know=dk_votes, neighbour_votes_total=int(labs.size),
               tie_break="cc.knn_vote (v2)", k=K, clip="openai/clip-vit-base-patch32 (HF, QuickGELU)")
    write_row("A", f"t3_{name}", 42, row)
    print(f"{name}: index={int(mask.sum()):6d} acc {m['acc']:.4f} macroF1 {m['macro_f1']:.4f} wF1 {m['weighted_f1']:.4f} | "
          f"self in top-5: {self_in_topk}/{len(test_idx)} (top-1: {self_top1}) | sibling-tweet in top-5: {sib_in_topk} | "
          f"DK votes {dk_votes}/{labs.size} | predicted DK {row['predicted_dont_know']}")

# regression check: L4 must reproduce the v2 headline (results/metrics.json)
import json
v2 = json.load(open(cc.__file__.replace("crisis_common.py", "results/metrics.json")))["T3_test"]
print("v2 headline:", {k: round(v, 4) for k, v in v2.items()})
