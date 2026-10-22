"""S1: CLIP zero-shot damage severity (image only) on the damage split. Ensemble chosen by DEV macro-F1; TEST evaluated once.
Prompts come from results/s1_prompts.json (written and committed before this ran). Run from the repo root."""
import json, itertools
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score

import crisis_common as cc

cc.seed_all()
D = Path("data/CrisisMMD_v2.0"); S = D / "crisismmd_datasplit_all" / "crisismmd_datasplit_all"
EMB, RES, FIG = Path("artifacts/emb"), Path("results"), Path("results/figures")
LAB = cc.T3_LABELS
dev, _ = cc.load_split("damage", "dev", D, S); test, _ = cc.load_split("damage", "test", D, S)
print("rows: dev", len(dev), "test", len(test))

def img_feats(df, split):
    ids = json.loads((EMB / f"clip_damage_{split}.ids.json").read_text())
    assert ids == list(df.image_id), f"cached CLIP ids do not align with damage {split} rows"
    x = np.load(EMB / f"clip_damage_{split}.npy"); assert x.shape == (len(df), 512), x.shape   # get_image_features, projected, raw
    return cc.l2(x)
Xd, Xt = img_feats(dev, "dev"), img_feats(test, "test")
print("cached CLIP image features reused (512-d, ids aligned by image_id)")

dev_, clip_ = ("cuda" if torch.cuda.is_available() else "cpu"), None
clip_, proc = cc.load_clip(dev_, "artifacts/clip")

@torch.no_grad()
def text_emb(texts):
    tok = proc.tokenizer(texts, padding=True, return_tensors="pt").to(dev_)
    out = clip_.get_text_features(**tok)
    if not isinstance(out, torch.Tensor): out = out.text_embeds if hasattr(out, "text_embeds") else out.pooler_output
    return cc.l2(out.float().cpu().numpy())

P = json.load(open(RES / "s1_prompts.json"))
def prompts_for(name):
    if name in ("E1", "E2"): return P[name]
    ph = {c: [p.removeprefix("a photo of ") for p in v] for c, v in P["E2"].items()}
    return {c: [t.format(p) for p in v for t in P["E3"]["templates"]] for c, v in ph.items()}
def class_matrix(name):
    pr = prompts_for(name)
    return cc.l2(np.stack([cc.l2(text_emb(pr[c])).mean(0) for c in LAB]))        # mean of L2-normalised, re-normalised
W = {n: class_matrix(n) for n in ("E1", "E2", "E3")}

def full(y, pred):
    m = cc.metrics(y, pred, labels=LAB)
    rec = recall_score(y, pred, labels=LAB, average=None, zero_division=0)
    dist = pd.Series(pred).value_counts().reindex(LAB, fill_value=0)
    return dict(m, **{f"recall_{l}": r for l, r in zip(LAB, rec)}, **{f"pred_share_{l}": dist[l] / len(pred) for l in LAB})
yd, yt = dev.label.to_numpy(), test.label.to_numpy()
predict = lambda X, Wm, b=0: np.array(LAB)[(X @ Wm.T + b).argmax(1)]

# 1) candidate selection on DEV only
rows = [dict(ensemble=n, **full(yd, predict(Xd, W[n]))) for n in W]
cand = pd.DataFrame(rows); cand.to_csv(RES / "s1_dev_candidates.csv", index=False)
print(cand.round(4).to_string())
best = cand.sort_values("macro_f1", ascending=False).iloc[0].ensemble
print("chosen by DEV macro-F1:", best)

# 2) TEST once for the chosen ensemble
pt = predict(Xt, W[best]); r_zs = full(yt, pt)

# 3) dev-calibrated: two free additive biases (third fixed at 0), grid on DEV macro-F1
grid = np.arange(-0.06, 0.0601, 0.002); bestb, bestf = (0, 0), -1
for b1, b2 in itertools.product(grid, grid):
    f = cc.metrics(yd, predict(Xd, W[best], np.array([0, b1, b2])), labels=LAB)["macro_f1"]
    if f > bestf + 1e-12: bestf, bestb = f, (b1, b2)
bias = np.array([0, *bestb]); print("dev-calibrated biases (Severe, Mild, Little):", bias.round(3), "dev macro-F1", round(bestf, 4))
pc = predict(Xt, W[best], bias); r_cal = full(yt, pc)

# 4) k-NN row read from existing results (not recomputed)
ba = pd.read_csv(RES / "before_after.csv")
k = ba[(ba.task == "T3 Damage (3 cls)") & (ba.version == "corrected pipeline")].iloc[0]
pk = pd.read_csv(RES / "preds_task3_test.csv"); assert list(pk.image_id) == list(test.image_id)
r_knn = full(pk["true"].to_numpy(), pk["pred"].to_numpy())
assert abs(r_knn["macro_f1"] - k.macro_f1) < 1e-9
out = pd.DataFrame([dict(method="Task 3 k-NN (corrected; image+text, trained index)", **r_knn),
                    dict(method=f"CLIP zero-shot ({best}, image only, no training)", **r_zs),
                    dict(method=f"CLIP {best} + dev-calibrated biases (NOT pure zero-shot)", **r_cal)])
out.to_csv(RES / "s1_zeroshot.csv", index=False)
print(out.round(4).T.to_string())

# 5) confusion matrices (one figure, three panels, same style as the other figures)
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
from matplotlib.colors import LinearSegmentedColormap
ramp = LinearSegmentedColormap.from_list("s", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
fig, axs = plt.subplots(1, 3, figsize=(15, 4.8)); fig.patch.set_facecolor(SURF)
short = ["Severe", "Mild", "Little/none"]
for ax, (ttl, y, p, m) in zip(axs, [("Task 3 k-NN", pk["true"].to_numpy(), pk["pred"].to_numpy(), r_knn),
                                    (f"CLIP zero-shot ({best})", yt, pt, r_zs), ("Dev-calibrated biases", yt, pc, r_cal)]):
    cm = confusion_matrix(y, p, labels=LAB); nm = cm / cm.sum(1, keepdims=True)
    ax.imshow(nm, cmap=ramp, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3): ax.text(j, i, f"{cm[i,j]}\n{nm[i,j]:.0%}", ha="center", va="center", fontsize=9, color=SURF if nm[i, j] > .5 else INK)
    ax.set_xticks(range(3), short); ax.set_yticks(range(3), short); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{ttl}\nmacro-F1 {m['macro_f1']:.3f}, acc {m['acc']:.3f}", loc="left", fontsize=11, fontweight="bold")
    for s in ax.spines.values(): s.set_visible(False)
fig.suptitle(f"Damage severity, test n = {len(yt)} (row-normalised, counts above %)", x=0.01, ha="left", fontsize=12, color=INK2)
fig.tight_layout(); fig.savefig(FIG / "confusion_s1_zeroshot.png", dpi=150, facecolor=SURF)
json.dump(dict(chosen=best, biases=bias.tolist(), dev_macro_f1_calibrated=bestf), open(RES / "s1_selection.json", "w"), indent=2)
