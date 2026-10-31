"""Day 4 final round B2 + B3: duplicate checks for the TASK 1 split (SELECTION_RULE_DAY4.md rule 21; same format as the report's section on duplicate images).
Text (B2): exact / near (TF-IDF cosine >= 0.9) duplicates of the normalised test tweet text against the Task 1 TRAIN text; images (B3): identical md5 or CLIP B/32 cosine >= 0.98 against Task 1 train images.
Flagged test rows are removed and the stored predictions of the Task 1 models are re-scored ("before" must equal the stored run). Writes results/ablation/D4_T1_dups.{json,md}."""
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

import d4_dup_utils as U
import d4_t1_data as d4
import metrics_all as M
from common import ABL_ART, ART, RES, cc

SEEDS = (42, 43, 44)
TEXT_MODELS = [("tfidf_lr", "TF-IDF + LR", (42,)), ("bertweet", "BERTweet (3 seeds)", SEEDS), ("bertweet_deploy", "BERTweet, deployed checkpoint", (42,))]
IMG_MODELS = [("img_resnet50", "ResNet50 head", SEEDS), ("img_dinov2", "DINOv2 head", SEEDS), ("img_clip_l14", "CLIP L/14 head", SEEDS), ("bertweet", "BERTweet text-only (reference)", SEEDS),
              ("late_bertweet+img_clip_l14", "late fusion", SEEDS), ("early_bertweet_resnet50", "core model BERTweet + ResNet50", SEEDS), ("early_bertweet_clip_l14", "BERTweet + CLIP L/14", SEEDS)]


def scores(name, seeds, mask):
    """mean over seeds of accuracy / weighted F1 / macro-F1 on the rows in `mask` (stored probabilities)."""
    out = {k: [] for k in ("acc", "weighted_f1", "macro_f1")}
    for s in seeds:
        r = M.load_run("T1", name, s); y, p = r["y_test"][mask], np.array(r["probs_test"])[mask].argmax(1)
        out["acc"].append(float((y == p).mean())); out["weighted_f1"].append(float(f1_score(y, p, labels=[0, 1], average="weighted", zero_division=0))); out["macro_f1"].append(float(f1_score(y, p, labels=[0, 1], average="macro", zero_division=0)))
    return {k: float(np.mean(v)) for k, v in out.items()}


def rescored(models, drop, key):
    keep = ~drop; res = {}
    for name, label, seeds in models:
        before, after = scores(name, seeds, np.ones(len(drop), bool)), scores(name, seeds, keep)
        stored = np.mean([M.load_run("T1", name, s)["test"]["weighted_f1"] for s in seeds]); assert abs(before["weighted_f1"] - stored) < 1e-3, (name, before, stored)
        res[name] = dict(label=label, n_before=int(len(drop)), n_after=int(keep.sum()), before=before, after=after, delta={k: after[k] - before[k] for k in before})
    return res


if __name__ == "__main__":
    D = d4.data(); S = D["S"]; y = D["y"]
    tr, te = S["train"], S["test"]
    # ---------------- B2: text duplicates ----------------
    ntr, nte = [U.norm_text(t) for t in tr.tweet_text.fillna("")], [U.norm_text(t) for t in te.tweet_text.fillna("")]
    train_set = {}
    for t, l in zip(ntr, y["train"]):
        if t:
            train_set.setdefault(t, set()).add(int(l))
    exact = np.array([bool(t) and t in train_set for t in nte])
    sim = U.tfidf_sims(ntr, nte); smax = sim.max(1)
    near = smax >= U.TEXT_THR                                    # exact duplicates have cosine ~1, so they are inside this set
    near |= exact
    conflict = np.array([exact[i] and (int(y["test"][i]) not in train_set[nte[i]] or len(train_set[nte[i]]) > 1) for i in range(len(te))])
    agree = (pd.read_csv(d4.SPLIT_DIR / "task_informative_text_img_test.tsv", sep="\t", dtype=str).drop_duplicates(subset=["image_id"]).set_index("image_id").loc[D["ids"]["test"]]
             .pipe(lambda r: (r.label_text.str.strip() == r.label_image.str.strip()).values))
    text = dict(threshold=U.TEXT_THR, n_test=int(len(te)), n_exact=int(exact.sum()), n_near_incl_exact=int(near.sum()), n_near_not_exact=int((near & ~exact).sum()), n_exact_with_conflicting_or_mixed_train_label=int(conflict.sum()),
                n_empty_after_normalisation=int(sum(1 for t in nte if not t)), flagged_agreed=int((near & agree).sum()), flagged_disagreed=int((near & ~agree).sum()))
    text["rescored"] = rescored(TEXT_MODELS, near, "text")
    # ---------------- B3: image duplicates ----------------
    md5_tr, md5_te = np.array(U.md5s(tr.img_path.tolist())), np.array(U.md5s(te.img_path.tolist()))
    ex_img = np.isin(md5_te, md5_tr)
    dev_ = torch.device("cuda")
    clip, proc = cc.load_clip(dev_, ART / "clip")
    feats = {}
    for s in ("train", "test"):
        feats[s] = cc.cached(ABL_ART / "d4_feats" / "clip_b32" / f"informative_{s}.npy", D["ids"][s], lambda s=s: cc.embed_images(S[s].img_path.tolist(), clip, proc, dev_)[0])
    E = {s: cc.l2(feats[s]).astype("float32") for s in feats}
    isim = E["test"] @ E["train"].T; imax = isim.max(1); near_img = imax >= U.IMG_THR
    flag_img = near_img | ex_img
    same_tweet = np.array([bool(ex_img[i]) and te.tweet_id.iloc[i] in set(tr.tweet_id[md5_tr == md5_te[i]]) for i in range(len(te))])
    img = dict(threshold=U.IMG_THR, n_test=int(len(te)), n_exact_md5=int(ex_img.sum()), n_near_incl_exact=int(flag_img.sum()), n_near_not_exact=int((near_img & ~ex_img).sum()), exact_same_tweet_id=int(same_tweet.sum()),
               flagged_agreed=int((flag_img & agree).sum()), flagged_disagreed=int((flag_img & ~agree).sum()))
    img["rescored"] = rescored(IMG_MODELS, flag_img, "image")
    out = dict(text=text, image=img, note="'near' includes exact; rows flagged by either test are removed; train/index unchanged")
    (RES / "D4_T1_dups.json").write_text(json.dumps(out, indent=1))
    f = lambda d: f"{d['weighted_f1']:.4f} / {d['macro_f1']:.4f} / {d['acc']:.4f}"
    md = "# Task 1 duplicate checks (test rows against the Task 1 training split; rule 21)\n\n"
    md += (f"**Text.** {text['n_exact']} of {text['n_test']} test tweets have the same normalised text as a training tweet (exact), {text['n_near_incl_exact']} are exact or near-duplicates (TF-IDF cosine >= {U.TEXT_THR}; {text['n_near_not_exact']} are near but not exact); "
           f"{text['n_exact_with_conflicting_or_mixed_train_label']} exact duplicates have a training copy with a different or mixed label; {text['flagged_agreed']} flagged rows have agreeing text/image labels, {text['flagged_disagreed']} disagreeing.\n\n")
    md += "| Model | n before -> after | weighted F1 / macro-F1 / acc before | after | change (wF1, macro-F1) |\n|---|---|---|---|---|\n"
    for n, r in text["rescored"].items():
        md += f"| {r['label']} | {r['n_before']} -> {r['n_after']} | {f(r['before'])} | {f(r['after'])} | {r['delta']['weighted_f1']:+.4f}, {r['delta']['macro_f1']:+.4f} |\n"
    md += (f"\n**Images.** {img['n_exact_md5']} of {img['n_test']} test images have a byte-identical file in the training split ({img['exact_same_tweet_id']} with the same tweet id), {img['n_near_incl_exact']} are exact or near-duplicates (CLIP B/32 cosine >= {U.IMG_THR}; {img['n_near_not_exact']} near but not byte-identical).\n\n")
    md += "| Model | n before -> after | weighted F1 / macro-F1 / acc before | after | change (wF1, macro-F1) |\n|---|---|---|---|---|\n"
    for n, r in img["rescored"].items():
        md += f"| {r['label']} | {r['n_before']} -> {r['n_after']} | {f(r['before'])} | {f(r['after'])} | {r['delta']['weighted_f1']:+.4f}, {r['delta']['macro_f1']:+.4f} |\n"
    (RES / "D4_T1_dups.md").write_text(md, encoding="utf-8"); print(md)
