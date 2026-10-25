"""F: duplicate-image check (CPU only; no training, nothing existing is changed).

1. md5 of every image file under data/CrisisMMD_v2.0/data_image; exact-duplicate groups (distinct files with identical bytes), counted across different tweet ids.
2. (a) Task 3 test images whose identical file is in the Task 3 index (v2's index: damage-train minus tweet siblings of test, 2,361 rows);
   (b) Task 2 test images whose identical file is in Task 2 train/dev.
3. Near-duplicates: CLIP ViT-B/32 image-embedding cosine >= 0.98 (v2's cached vectors, L2-normalised): T3 test vs the index rows; T2 test vs T2 train.
4. If anything is flagged, the stored predictions are re-scored with the flagged test rows excluded (index/train unchanged): T3 hybrid (= the probe's label) and 5-NN,
   T2 L4a (3 seeds, CPU inference from the saved checkpoints) and image_only (3 seeds, stored predictions). "Before" is asserted equal to the stored results.
5. The five live demo examples: md5 asserted absent from the T3 index; top-1 neighbour similarity reported (fused index search and image-only cosine).
Writes results/ablation/X/f_duplicates_seed42.json.
"""
import hashlib
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import faiss
import numpy as np
import pandas as pd
import torch
from transformers import DistilBertConfig, DistilBertTokenizerFast

from common import ABL_ART, ART, DATA, RES, ROOT, SPLIT_DIR, cc, write_row

THR = 0.98
X = RES / "X"
LAB3 = cc.T3_LABELS


def no_miss():
    raise RuntimeError("cache miss")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    torch.set_num_threads(4)
    # ── 1. md5 of every image file; exact-duplicate groups across tweet ids ─────────────────────────────────────────
    files = sorted(str(p) for p in (DATA / "data_image").rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    with ThreadPoolExecutor(8) as ex:
        digests = list(ex.map(md5, files))
    MD5 = dict(zip(files, digests))
    from common import load_annotations
    ann = load_annotations().rename(columns={"image_path": "path"})
    tw_of = defaultdict(set)
    for p, t in zip(ann["path"], ann["tweet_id"]):
        tw_of[p].add(t)
    groups = defaultdict(list)
    for p, d in MD5.items():
        groups[d].append(p)
    dup = {d: ps for d, ps in groups.items() if len(ps) > 1}
    cross = {d: ps for d, ps in dup.items() if len(set().union(*[tw_of.get(p, {Path(p).stem.split("_")[0]}) for p in ps])) > 1}
    print(f"{len(files)} image files, {len(groups)} distinct md5; {len(dup)} exact-duplicate groups ({sum(len(v) for v in dup.values())} files); "
          f"{len(cross)} groups span different tweet ids ({sum(len(v) for v in cross.values())} files)", flush=True)
    unmapped = [p for p in files if p not in tw_of]

    # ── splits ──────────────────────────────────────────────────────────────────────────────────────────────────────
    S3 = {s: cc.load_split("damage", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "dev", "test")}
    S2 = {s: cc.load_split("humanitarian", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "dev", "test")}
    for d in (S3, S2):
        for df in d.values():
            df["md5"] = df["img_path"].map(MD5)
            assert df["md5"].notna().all()
    keep3 = ~S3["train"].tweet_id.isin(set(S3["test"].tweet_id)).to_numpy()
    idx3 = S3["train"][keep3].reset_index(drop=True)
    meta = pd.read_csv(ART / "task3_meta.csv", dtype=str, keep_default_na=False)
    assert len(idx3) == len(meta) == 2361 and list(idx3.image_id) == list(meta.image_id), "reconstructed index != v2 index"

    # ── 2. exact duplicates: (a) T3 test vs T3 index, (b) T2 test vs T2 train/dev ───────────────────────────────────
    te3, te2 = S3["test"], S2["test"]
    ex3 = te3.md5.isin(set(idx3.md5)).to_numpy()
    ref2 = pd.concat([S2["train"], S2["dev"]], ignore_index=True)
    ex2 = te2.md5.isin(set(ref2.md5)).to_numpy()
    md5_lab3 = idx3.groupby("md5")["label"].agg(set).to_dict()
    md5_lab2 = ref2.groupby("md5")["label"].agg(set).to_dict()
    md5_tw2 = ref2.groupby("md5")["tweet_id"].agg(set).to_dict()
    a_same_label = int(sum(te3.label.iloc[i] in md5_lab3[te3.md5.iloc[i]] for i in np.where(ex3)[0]))
    b_same_label = int(sum(te2.label.iloc[i] in md5_lab2[te2.md5.iloc[i]] for i in np.where(ex2)[0]))
    b_same_tweet = int(sum(te2.tweet_id.iloc[i] in md5_tw2[te2.md5.iloc[i]] for i in np.where(ex2)[0]))
    print(f"(a) T3 test rows with an identical file in the T3 index: {int(ex3.sum())} of {len(te3)} (identical label present among them: {a_same_label})")
    print(f"(b) T2 test rows with an identical file in T2 train/dev: {int(ex2.sum())} of {len(te2)} (same tweet id: {b_same_tweet}; identical label present: {b_same_label})", flush=True)

    # ── 3. near-duplicates, B/32 cosine ─────────────────────────────────────────────────────────────────────────────
    emb = lambda task, split, df: cc.l2(cc.cached(ART / "emb" / f"clip_{task}_{split}.npy", df.image_id.tolist(), no_miss)).astype("float32")
    E3 = {s: emb("damage", s, S3[s]) for s in S3}
    E2 = {s: emb("humanitarian", s, S2[s]) for s in S2}
    sim3 = E3["test"] @ E3["train"][keep3].T
    sim2 = E2["test"] @ E2["train"].T
    m3, a3 = sim3.max(1), sim3.argmax(1)
    m2, a2 = sim2.max(1), sim2.argmax(1)
    nd3, nd2 = m3 >= THR, m2 >= THR
    print(f"near-duplicates (cos >= {THR}): T3 test vs index {int(nd3.sum())} of {len(te3)} queries; T2 test vs T2 train {int(nd2.sum())} of {len(te2)} rows "
          f"(of which not byte-identical: {int((nd3 & ~ex3).sum())} and {int((nd2 & ~ex2).sum())})", flush=True)
    # a T2 near-duplicate in dev (not just train) is not counted here by design: the user's definition is T2 test vs T2 train

    def examples(m, am, nd, te, ref, ex_mask, n=3):
        ids = np.where(nd)[0]
        if len(ids) == 0:
            return []
        order = ids[np.argsort(-m[ids])]
        pick = order[np.linspace(0, len(order) - 1, min(n, len(order))).round().astype(int)]
        out = []
        for i in pick:
            r = ref.iloc[am[i]]
            out.append(dict(test_image_id=te.image_id.iloc[i], test_tweet_id=te.tweet_id.iloc[i], test_label=te.label.iloc[i], test_event=te.event_name.iloc[i], test_file=te.rel_image.iloc[i],
                            ref_image_id=r.image_id, ref_tweet_id=r.tweet_id, ref_label=r.label, ref_event=r.event_name, ref_file=r.rel_image, cosine=float(m[i]), byte_identical=bool(ex_mask[i])))
        return out
    ex3_list = examples(m3, a3, nd3, te3, idx3, ex3)
    ex2_list = examples(m2, a2, nd2, te2, S2["train"], ex2)
    for tag, lst in (("T3", ex3_list), ("T2", ex2_list)):
        for e in lst:
            print(f"  {tag} example: {e['test_image_id']} ({e['test_label']}, {e['test_event']}) ~ {e['ref_image_id']} ({e['ref_label']}, {e['ref_event']}) cos {e['cosine']:.4f} byte-identical {e['byte_identical']}")

    # ── 4. re-score with the flagged test rows excluded ─────────────────────────────────────────────────────────────
    drop3 = ex3 | nd3
    drop2 = ex2 | nd2
    rescored = {}
    if drop3.any() or drop2.any():
        l3 = {l: i for i, l in enumerate(LAB3)}
        y3 = te3.label.to_numpy()
        for name, cfg in (("T3 hybrid (probe label)", "b_t3_B32_fused_probe"), ("T3 5-NN majority", "b_t3_B32_fused_knn5")):
            pred = np.array(LAB3)[np.array(json.load(open(X / f"{cfg}_seed42.json"))["pred"])]
            stored = json.load(open(X / f"{cfg}_seed42.json"))["test_macro_f1"]
            before = cc.metrics(y3, pred, LAB3)
            assert abs(before["macro_f1"] - stored) < 1e-12
            k = ~drop3
            after = cc.metrics(y3[k], pred[k], LAB3)
            rescored[name] = dict(n_before=int(len(y3)), n_after=int(k.sum()), before_macro_f1=before["macro_f1"], after_macro_f1=after["macro_f1"], before_acc=before["acc"], after_acc=after["acc"])
        labels2 = sorted(S2["train"]["label"].unique())
        y2 = te2.label.to_numpy()
        k2 = ~drop2
        # image_only: stored predictions (validated equal to Part B in b_t2_l14.py)
        for seed in (42, 43, 44):
            d = json.load(open(X / f"b_t2_imageonly_B32_raw_seed{seed}.json"))
            pred = np.array(labels2)[np.array(d["pred"])]
            before = cc.metrics(y2, pred, labels2)
            assert abs(before["macro_f1"] - d["test_macro_f1"]) < 1e-12
            after = cc.metrics(y2[k2], pred[k2], labels2)
            rescored[f"T2 image_only seed {seed}"] = dict(n_before=int(len(y2)), n_after=int(k2.sum()), before_macro_f1=before["macro_f1"], after_macro_f1=after["macro_f1"],
                                                          before_acc=before["acc"], after_acc=after["acc"])
        # L4a: CPU inference from the saved checkpoints
        tok = DistilBertTokenizerFast.from_pretrained(ART / "tokenizer")
        bcfg = DistilBertConfig.from_pretrained(ART / "distilbert_config")
        ids, mask = cc.tokenize(tok, te2.text.tolist(), 128)
        vec = cc.cached(ART / "emb" / "clip_humanitarian_test.npy", te2.image_id.tolist(), no_miss)
        dev = torch.device("cpu")
        for seed in (42, 43, 44):
            model = cc.DualStreamClassifier(len(labels2), bert_config=bcfg).to(dev)
            model.load_state_dict(torch.load(ABL_ART / "ckpt" / f"t2_L4a_seed{seed}.pt", map_location=dev))
            model.eval()
            pred = np.array(labels2)[cc.t2_predict_proba(model, vec, ids, mask, dev).argmax(1)]
            stored = json.load(open(RES / "A" / f"t2_L4a_seed{seed}.json"))["test_macro_f1"]
            before = cc.metrics(y2, pred, labels2)
            assert abs(before["macro_f1"] - stored) < 1e-3, (seed, before["macro_f1"], stored)
            after = cc.metrics(y2[k2], pred[k2], labels2)
            rescored[f"T2 L4a seed {seed}"] = dict(n_before=int(len(y2)), n_after=int(k2.sum()), before_macro_f1=before["macro_f1"], after_macro_f1=after["macro_f1"], stored_macro_f1=stored,
                                                   before_acc=before["acc"], after_acc=after["acc"])
        for k_, v in rescored.items():
            print(f"  {k_:28s} n {v['n_before']} -> {v['n_after']}  macro-F1 {v['before_macro_f1']:.4f} -> {v['after_macro_f1']:.4f}")

    # ── 5. the five live demo examples ──────────────────────────────────────────────────────────────────────────────
    live = json.load(open(ROOT / "demo" / "live_examples.json", encoding="utf8"))
    index = faiss.read_index(str(ART / "task3_index.faiss"))
    idx_md5 = set(idx3.md5)
    sbert = cc.load_sbert(torch.device("cpu"), ART / "sbert")
    img_by_id = {}
    for df, E in ((S3["test"], E3["test"]), (S2["test"], E2["test"]), (S2["dev"], E2["dev"]), (S2["train"], E2["train"])):
        for i, v in zip(df.image_id, E):
            img_by_id.setdefault(i, v)
    idx_img = E3["train"][keep3]
    demo = []
    for e in live:
        d = md5(Path(e["image_path"]))
        assert d not in idx_md5, f"demo example {e['image_id']} has an identical file in the T3 index"
        img = img_by_id[e["image_id"]][None]
        q = cc.t3_vectors(img, cc.embed_texts([cc.clean_tweet(e["tweet_text"])], sbert))      # the demo's own query construction
        s_f, i_f = index.search(q, 1)
        cos = idx_img @ img[0]
        demo.append(dict(image_id=e["image_id"], kind=e["kind"], md5=d, in_t3_index=False, top1_fused_similarity=float(s_f[0, 0]), top1_image_cosine=float(cos.max()),
                         top1_image_neighbour=idx3.image_id.iloc[int(cos.argmax())]))
        print(f"  demo {e['image_id']:24s} ({e['kind']}): md5 not in index; top-1 fused sim {s_f[0, 0]:.4f}, top-1 image cosine {cos.max():.4f}")

    write_row("X", "f_duplicates", 42, dict(
        task="T2/T3", tier="cpu", threshold=THR, n_files=len(files), n_distinct_md5=len(groups), n_files_not_in_annotations=len(unmapped),
        exact_groups=len(dup), exact_group_files=sum(len(v) for v in dup.values()), exact_groups_cross_tweet=len(cross), exact_group_files_cross_tweet=sum(len(v) for v in cross.values()),
        largest_group=max(len(v) for v in dup.values()), t3_n_test=int(len(te3)), t3_index_rows=int(len(idx3)), t3_test_exact_in_index=int(ex3.sum()), t3_exact_same_label_present=a_same_label,
        t2_n_test=int(len(te2)), t2_ref_rows=int(len(ref2)), t2_test_exact_in_train_dev=int(ex2.sum()), t2_exact_same_tweet=b_same_tweet, t2_exact_same_label_present=b_same_label,
        t3_near_dup_queries=int(nd3.sum()), t3_near_not_byte_identical=int((nd3 & ~ex3).sum()), t2_near_dup_rows=int(nd2.sum()), t2_near_not_byte_identical=int((nd2 & ~ex2).sum()),
        t3_excluded=int(drop3.sum()), t2_excluded=int(drop2.sum()), examples_t3=ex3_list, examples_t2=ex2_list, rescored=rescored, demo_examples=demo))


if __name__ == "__main__":
    main()
