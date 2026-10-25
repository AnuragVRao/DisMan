"""B: Task 3 on CLIP ViT-L/14 embeddings (CPU; extension experiment, nothing existing is changed).

Same construction as t3_partb.py: fused = l2(concat(l2(image), SBERT text)) via cc.t3_vectors (the rule v2 uses), image-only = l2(image); the same tweet-disjoint
index rows (damage train minus rows sharing a tweet_id with the evaluated split; test: 2,361 rows); 5-NN majority vote (cc.knn_vote); logistic probe
(class_weight="balanced", C tuned on dev from {0.01, 0.1, 1, 10, 100}) fitted on the same index rows. B/32 is recomputed with the same code and asserted to reproduce the
stored Part B rows, so the L/14 numbers come from an identical path. Writes results/ablation/X/b_t3_<enc>_<emb>_<knn5|probe>_seed42.json (with test predictions).
"""
import json
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import faiss
import numpy as np
from sklearn.linear_model import LogisticRegression

from common import ABL_ART, ART, DATA, RES, SPLIT_DIR, cc, write_row

faiss.omp_set_num_threads(4)
LABELS = cc.T3_LABELS
L14 = ABL_ART / "clip_l14"


def no_miss():
    raise RuntimeError("cache miss")


def main():
    S = {s: cc.load_split("damage", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "dev", "test")}
    txt = {s: cc.cached(ART / "emb" / f"sbert_damage_{s}.npy", S[s].image_id.tolist(), no_miss) for s in S}
    img = {"B32": {s: cc.cached(ART / "emb" / f"clip_damage_{s}.npy", S[s].image_id.tolist(), no_miss) for s in S},
           "L14": {s: cc.cached(L14 / f"clip_damage_{s}.npy", S[s].image_id.tolist(), no_miss) for s in S}}
    assert img["L14"]["train"].shape[1] == 768 and img["B32"]["train"].shape[1] == 512
    keep = {sp: ~S["train"].tweet_id.isin(set(S[sp].tweet_id)).to_numpy() for sp in ("dev", "test")}
    ylab = S["train"].label.to_numpy()
    y = {sp: S[sp].label.to_numpy() for sp in ("dev", "test")}
    l3 = {l: i for i, l in enumerate(LABELS)}

    def knn_preds(V, sp):
        rows = np.where(keep[sp])[0]
        ix = faiss.IndexFlatIP(V["train"].shape[1]); ix.add(V["train"][rows])
        sims, nn_ = ix.search(V[sp], 5)
        return np.array([cc.knn_vote(list(ylab[rows][r]), list(s))[0] for r, s in zip(nn_, sims)])

    for enc in ("B32", "L14"):
        VEC = {"fused": {s: cc.t3_vectors(img[enc][s], txt[s]) for s in S}, "image": {s: cc.l2(img[enc][s]).astype("float32") for s in S}}
        for emb in ("fused", "image"):
            V = VEC[emb]
            # 5-NN majority (dev for the record, test for the result)
            pk_d, pk_t = knn_preds(V, "dev"), knn_preds(V, "test")
            mk_d, mk_t = cc.metrics(y["dev"], pk_d, LABELS), cc.metrics(y["test"], pk_t, LABELS)
            write_row("X", f"b_t3_{enc}_{emb}_knn5", 42, dict(task="T3", tier="cpu", encoder=enc, embedding=emb, classifier="5-NN majority", n_test=len(y["test"]),
                                                              index_rows_test=int(keep["test"].sum()), dev_macro_f1=mk_d["macro_f1"], test_acc=mk_t["acc"],
                                                              test_macro_f1=mk_t["macro_f1"], test_weighted_f1=mk_t["weighted_f1"], pred=[l3[p] for p in pk_t]))
            # logistic probe, C tuned on dev
            best = None
            for C in (0.01, 0.1, 1, 10, 100):
                clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(V["train"][keep["dev"]], ylab[keep["dev"]])
                f1 = cc.metrics(y["dev"], clf.predict(V["dev"]), LABELS)["macro_f1"]
                if best is None or f1 > best[1]:
                    best = (C, f1)
            clf = LogisticRegression(C=best[0], class_weight="balanced", max_iter=3000).fit(V["train"][keep["test"]], ylab[keep["test"]])
            pp = clf.predict(V["test"]); mp = cc.metrics(y["test"], pp, LABELS)
            write_row("X", f"b_t3_{enc}_{emb}_probe", 42, dict(task="T3", tier="cpu", encoder=enc, embedding=emb, classifier="logistic probe", C=best[0], n_test=len(y["test"]),
                                                               index_rows_test=int(keep["test"].sum()), dev_macro_f1=best[1], test_acc=mp["acc"], test_macro_f1=mp["macro_f1"],
                                                               test_weighted_f1=mp["weighted_f1"], pred=[l3[p] for p in pp]))
            print(f"{enc} {emb:5s}: 5-NN dev F1 {mk_d['macro_f1']:.4f} test acc {mk_t['acc']:.4f} F1 {mk_t['macro_f1']:.4f} | probe C={best[0]:<5} dev F1 {best[1]:.4f} "
                  f"test acc {mp['acc']:.4f} F1 {mp['macro_f1']:.4f}", flush=True)
            if enc == "B32":      # identical code path: must equal the stored Part B rows
                sk = json.load(open(RES / "B" / f"t3_{emb}_k5_majority_seed42.json")); sp_ = json.load(open(RES / "B" / f"t3_lr_{emb}_seed42.json"))
                assert abs(mk_t["macro_f1"] - sk["test_macro_f1"]) < 1e-12 and abs(mp["macro_f1"] - sp_["test_macro_f1"]) < 1e-12 and best[0] == sp_["C"], "B/32 != stored Part B"


if __name__ == "__main__":
    main()
