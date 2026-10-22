"""S1 follow-up rows for results/s1_zeroshot.csv: trivial baselines (analytic) and Task 3 k-NN with CLIP image embedding only.
No tuning: k=5, same train-only/tweet-disjoint index construction as the headline Task 3 row, same 529 test rows."""
import json
from pathlib import Path
import numpy as np, pandas as pd, faiss
from sklearn.metrics import recall_score
import crisis_common as cc

D = Path("data/CrisisMMD_v2.0"); S = D / "crisismmd_datasplit_all" / "crisismmd_datasplit_all"
EMB, RES, LAB = Path("artifacts/emb"), Path("results"), cc.T3_LABELS
tr, _ = cc.load_split("damage", "train", D, S, verbose=False); te, _ = cc.load_split("damage", "test", D, S, verbose=False)
yt = te.label.to_numpy(); n = len(yt)

def load(split, df, kind):
    assert json.loads((EMB / f"{kind}_damage_{split}.ids.json").read_text()) == list(df.image_id)
    return np.load(EMB / f"{kind}_damage_{split}.npy")
Ctr, Cte = load("train", tr, "clip"), load("test", te, "clip"); Str, Ste = load("train", tr, "sbert"), load("test", te, "sbert")
keep = ~tr.tweet_id.isin(set(te.tweet_id)).to_numpy()
print(f"index rows: {keep.sum()} of {len(tr)} train rows ({(~keep).sum()} share a tweet_id with test and are excluded); test rows {n}")
assert not set(tr.tweet_id[keep]) & set(te.tweet_id) and not set(tr.image_id[keep]) & set(te.image_id)
lab_idx = tr.label.to_numpy()[keep]

def knn_pred(Xtr, Xte, k=5):
    idx = faiss.IndexFlatIP(Xtr.shape[1]); idx.add(Xtr[keep].astype("float32"))
    sc, ii = idx.search(Xte.astype("float32"), k)
    return np.array([cc.knn_vote(list(lab_idx[i]), list(s))[0] for i, s in zip(ii, sc)])

p_img = knn_pred(cc.l2(Ctr), cc.l2(Cte))                                   # CLIP image only
p_full = knn_pred(cc.t3_vectors(Ctr, Str), cc.t3_vectors(Cte, Ste))          # reproduction check of the headline row
ref = pd.read_csv(RES / "preds_task3_test.csv")
assert (p_full == ref.pred.to_numpy()).all(), "reconstructed image+text k-NN does not reproduce the saved Task 3 predictions"
print("check: rebuilt image+text k-NN reproduces the saved Task 3 test predictions exactly")

def full(y, pred):
    m = cc.metrics(y, pred, labels=LAB); rec = recall_score(y, pred, labels=LAB, average=None, zero_division=0)
    dist = pd.Series(pred).value_counts().reindex(LAB, fill_value=0)
    return dict(m, **{f"recall_{l}": r for l, r in zip(LAB, rec)}, **{f"pred_share_{l}": dist[l] / len(pred) for l in LAB})

p = pd.Series(yt).value_counts(normalize=True).reindex(LAB).to_numpy()
print("test class proportions:", dict(zip(LAB, p.round(4))))
# majority class "always Severe": recall 1/0/0; F1_sev = 2p/(1+p); macro = F1_sev/3; weighted = p*F1_sev
maj = full(yt, np.array([LAB[int(p.argmax())]] * n)); assert LAB[int(p.argmax())] == "Severe damage"
# stratified random predicting class c with prob p_c (independent of truth): P=R=F1=p_c per class -> macro = mean(p) = 1/3, acc = weighted = sum p^2
strat = dict(acc=float((p ** 2).sum()), macro_f1=float(p.mean()), weighted_f1=float((p ** 2).sum()), n=n,
             **{f"recall_{l}": float(q) for l, q in zip(LAB, p)}, **{f"pred_share_{l}": float(q) for l, q in zip(LAB, p)})
new = pd.DataFrame([dict(method="Baseline: majority class (always Severe)", **maj),
                    dict(method="Baseline: stratified random at test class proportions (expected, analytic)", **strat),
                    dict(method="Task 3 k-NN, CLIP image embedding only (k=5, same index construction)", **full(yt, p_img))])
old = pd.read_csv(RES / "s1_zeroshot.csv"); old = old[~old.method.isin(new.method)]
out = pd.concat([old, new], ignore_index=True); out.to_csv(RES / "s1_zeroshot.csv", index=False)
cols = ["method", "acc", "macro_f1", "weighted_f1", "recall_Severe damage", "recall_Mild damage", "recall_Little or no damage"]
print(out[cols].round(4).to_string())
