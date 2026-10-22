"""Generates capstone-v2.ipynb (run from the repo root: python tools/build_notebook.py)."""
import nbformat as nbf

cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


md(r'''
# CrisisMMD v2.0 — audited & corrected pipeline (v2)
Corrected rewrite of `capstone-final.ipynb` (left untouched). Local paths, one config block, runs top-to-bottom.

| Task | Model | Label source |
|---|---|---|
| 1 Informative | TF-IDF + LogisticRegression (text only) | `label_text` |
| 2 Humanitarian (8 classes) | frozen CLIP ViT-B/32 image vector + fine-tuned DistilBERT → cross-attention → MLP | `label_image` |
| 3 Damage severity (3 classes) | CLIP image + SBERT text → FAISS (train split only) → 5-NN vote | `label` |

**Audit fixes in this notebook**
1. `lstrip('data_image/')` → `removeprefix` (two whole events were silently dropped)
2. Task 3 FAISS index leaked the test images → index built from the damage TRAIN split only
3. Task 1 `target_names` were reversed → explicit `labels=`
4. `T3_CLIP` / `T3_CLIP_PROC` were never defined → defined explicitly (HF CLIP)
5. `TRAIN` flag → `False` loads from `artifacts/`
6. open_clip image features used the wrong activation (QuickGELU mismatch) → HF `CLIPModel`
7. Task 3 majority vote tie-break depended on Python set-hash order → deterministic tie-break

Rules: test split is never used for selection (dev only); every filter prints per-split and per-event counts; seed 42.
''')

md(r'''
## 0. Config (single block) — **Fix 5: `TRAIN` flag**
`TRAIN = False` loads models/index from `artifacts/` (fast, used by `jupyter nbconvert --execute`).
Set `CAPSTONE_TRAIN=1` (or edit the line) to retrain and re-export.
''')
code(r'''
import os, sys, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, joblib, faiss
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from transformers import DistilBertTokenizerFast, DistilBertConfig

ROOT = Path.cwd()                                   # run from the repo root
sys.path.insert(0, str(ROOT))
import crisis_common as cc

# ── paths ────────────────────────────────────────────────────────────────────
DATA      = ROOT / "data" / "CrisisMMD_v2.0"
SPLIT_DIR = DATA / "crisismmd_datasplit_all" / "crisismmd_datasplit_all"
ART, EMB, RES = ROOT / "artifacts", ROOT / "artifacts" / "emb", ROOT / "results"
for d in (ART, EMB, RES, RES / "figures"): d.mkdir(parents=True, exist_ok=True)

# ── flags / hyper-parameters ─────────────────────────────────────────────────
TRAIN = os.environ.get("CAPSTONE_TRAIN", "0") == "1"   # default: False -> load from artifacts/
SEED = cc.SEED
T2_BATCH, T2_MAX_LEN, T2_LR, T2_WARMUP, T2_EPOCHS = 32, 128, 1e-4, 3, 10
T3_K = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cc.seed_all(SEED)
T_START = time.time()
print(f"TRAIN={TRAIN}  device={DEVICE}  torch={torch.__version__}  seed={SEED}")
if DEVICE.type == "cuda": print(torch.cuda.get_device_name(0))
''')

md(r'''
## 1. Loading splits — **Fix 1: path bug**
The original code did `str(p).lstrip('data_image/')`. `str.lstrip(chars)` strips any leading **characters from the set**
`{d,a,t,_,i,m,g,e,/}`, not the prefix. `iraq_iran_earthquake` → `raq_iran_earthquake` and `mexico_earthquake` →
`xico_earthquake`, so the file lookup failed and those rows were silently dropped (~11% of every split).
Fix: `removeprefix('data_image/')` (in `crisis_common.load_split`). The loader prints row counts per split **and per event**
after every filter and returns a drop log with the reason for each drop.
''')
code(r'''
# demonstration of the bug on the real event names
for ev in ["iraq_iran_earthquake", "mexico_earthquake", "hurricane_harvey"]:
    p = f"data_image/{ev}/12_9_2017/x.jpg"
    print(f"{ev:22s} lstrip -> {p.lstrip('data_image/').split('/')[0]:22s} removeprefix -> {p.removeprefix('data_image/').split('/')[0]}")
''')
code(r'''
SPLITS, DROPLOG = {}, []
for task in ["informative", "humanitarian", "damage"]:
    for split in ["train", "dev", "test"]:
        df, log = cc.load_split(task, split, DATA, SPLIT_DIR)
        SPLITS[(task, split)] = df; DROPLOG += log
        print()
droplog = pd.DataFrame(DROPLOG)
print("=== DROP LOG (every drop, with reason) ===")
print(droplog.assign(per_event=droplog.per_event.astype(str)).to_string(index=False))
''')
code(r'''
# ── assertions for Fix 1 ─────────────────────────────────────────────────────
summary = pd.concat({f"{t}/{s}": d.event_name.value_counts() for (t, s), d in SPLITS.items()}, axis=1).fillna(0).astype(int)
print(summary.to_string())
for (t, s), d in SPLITS.items():
    ev = d.event_name.value_counts()
    for e in ["iraq_iran_earthquake", "mexico_earthquake"]:
        assert ev.get(e, 0) > 0, f"{e} missing from {t}/{s}"
missing_img = droplog[droplog.reason.str.startswith("image file missing")].n_dropped.sum()
assert missing_img == 0, f"{missing_img} rows dropped for a missing image path"
print("\nOK: both earthquake events present in every split of every task; rows dropped for missing image path = 0")
''')

md(r'''
## 2. Task 1 — Informative vs Not informative (text only)
**Fix 3 — report labels.** `sklearn` sorts labels alphabetically (`informative`, `not_informative`), but the original
passed `target_names=["not_informative", "informative"]`, so the per-class rows were mislabelled. We pass `labels=` explicitly
so each name is bound to its label.

**Regression check.** Task 1 is text-only and never touched image paths, so the corrected numbers must reproduce the old
headline (acc 0.7944 / weighted-F1 0.7970); only the per-class labels change.
''')
code(r'''
t1_tr, t1_dv, t1_te = (SPLITS[("informative", s)] for s in ("train", "dev", "test"))
if TRAIN:
    t1_model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=SEED)),
    ]).fit(t1_tr["text"], t1_tr["label"])
    joblib.dump(t1_model, ART / "task1_pipeline.joblib")
else:
    t1_model = joblib.load(ART / "task1_pipeline.joblib")

RESULTS = {}
for name, d in (("dev", t1_dv), ("test", t1_te)):
    pred = t1_model.predict(d["text"])
    RESULTS[f"T1_{name}"] = cc.metrics(d["label"], pred, cc.T1_LABELS)
    if name == "test": t1_pred = pred
print("T1 dev :", {k: round(v, 4) for k, v in RESULTS["T1_dev"].items()})
print("T1 test:", {k: round(v, 4) for k, v in RESULTS["T1_test"].items()})
print()
print(classification_report(t1_te["label"], t1_pred, labels=cc.T1_LABELS, target_names=cc.T1_LABELS, digits=4))
print(pd.DataFrame(confusion_matrix(t1_te["label"], t1_pred, labels=cc.T1_LABELS), index=cc.T1_LABELS, columns=cc.T1_LABELS))
t1_te.assign(pred=t1_pred)[["event_name", "tweet_id", "image_id", "label", "pred"]].to_csv(RES / "preds_task1_test.csv", index=False)

# regression check against the old headline numbers (to 3 decimals)
t1m = RESULTS["T1_test"]
assert round(t1m["acc"], 3) == round(0.7944, 3) and round(t1m["weighted_f1"], 3) == round(0.7970, 3), \
    f"Task 1 does not reproduce old numbers: acc={t1m['acc']:.4f} wF1={t1m['weighted_f1']:.4f}"
print("\nOK: Task 1 reproduces the old accuracy / weighted-F1 to 3 decimals.")
''')

md(r'''
## 3. Shared encoders — **Fix 4: define the CLIP objects** and **Fix 6: QuickGELU**
* **Fix 4.** `T3_CLIP` / `T3_CLIP_PROC` were used but never defined (the old notebook relied on stale Kaggle kernel state).
  They are now defined explicitly: HF `openai/clip-vit-base-patch32` + `CLIPProcessor`.
* **Fix 6.** The old Task 2 used `open_clip.create_model_and_transforms("ViT-B/32", pretrained="openai")`, which printed
  *"QuickGELU mismatch … (quick_gelu=False) vs pretrained tag 'openai' (quick_gelu=True)"*. The OpenAI weights were trained with
  QuickGELU, so the old image features used the wrong activation. HF `CLIPModel` uses QuickGELU for this checkpoint
  (`hidden_act = quick_gelu`, asserted below), so the same CLIP is now used for Tasks 2 and 3.
* The CLIP vision encoder is frozen and preprocessing is deterministic, so image embeddings are computed **once per split** and
  cached in `artifacts/emb/` (same for SBERT text embeddings). Task 2 then trains on cached vectors.
''')
code(r'''
T3_CLIP = T3_CLIP_PROC = None
def get_clip():
    global T3_CLIP, T3_CLIP_PROC
    if T3_CLIP is None:
        src = ART / "clip" if (ART / "clip").exists() else cc.CLIP_NAME
        T3_CLIP, T3_CLIP_PROC = cc.load_clip(DEVICE, src)
        assert T3_CLIP.config.vision_config.hidden_act == "quick_gelu", "CLIP is not using QuickGELU"
    return T3_CLIP, T3_CLIP_PROC

def clip_embeddings(task, split):
    df = SPLITS[(task, split)]
    def compute():
        clip, proc = get_clip()
        feats, bad = cc.embed_images(df["img_path"].tolist(), clip, proc, DEVICE)
        print(f"  [{task}/{split}] unreadable images replaced by grey: {bad}")
        return feats
    return cc.cached(EMB / f"clip_{task}_{split}.npy", df["image_id"].tolist(), compute)

t0 = time.time()
CLIP_EMB = {(t, s): clip_embeddings(t, s) for t in ("humanitarian", "damage") for s in ("train", "dev", "test")}
print({k: v.shape for k, v in CLIP_EMB.items()}, f"{time.time()-t0:.0f}s")
''')

md(r'''
## 4. Task 2 — Humanitarian categories (CLIP vector + DistilBERT → cross-attention → MLP)
Same head as before, trained on cached frozen-CLIP vectors. Changes vs the old run (all listed in `RESULTS.md`): path fix
(all events present), HF CLIP / QuickGELU, cached embeddings, ≤10 epochs, AMP. Best checkpoint chosen on **dev macro-F1**.

Note: the image side is a single pooled vector, so the txt→img cross-attention attends over **one key** and is effectively a
projection. It is left as-is (not redesigned).
''')
code(r'''
t2_tr, t2_dv, t2_te = (SPLITS[("humanitarian", s)] for s in ("train", "dev", "test"))
T2_LABELS = sorted(t2_tr["label"].unique()); L2I = {l: i for i, l in enumerate(T2_LABELS)}
NC = len(T2_LABELS); ALL_IDX = list(range(NC))
print(f"{NC} classes:", T2_LABELS)
assert set(t2_dv.label) <= set(T2_LABELS) and set(t2_te.label) <= set(T2_LABELS)
print(t2_tr.label.value_counts().to_string())

# tokenizer / DistilBERT config are saved into artifacts/ so the demo never needs the hub
if TRAIN or not (ART / "tokenizer").exists():
    DistilBertTokenizerFast.from_pretrained(cc.BERT_NAME).save_pretrained(ART / "tokenizer")
    DistilBertConfig.from_pretrained(cc.BERT_NAME).save_pretrained(ART / "distilbert_config")
tok = DistilBertTokenizerFast.from_pretrained(ART / "tokenizer")
bert_cfg = DistilBertConfig.from_pretrained(ART / "distilbert_config")

def prep(df, split):
    ids, mask = cc.tokenize(tok, df["text"].tolist(), T2_MAX_LEN)
    img = torch.tensor(CLIP_EMB[("humanitarian", split)], dtype=torch.float32)
    y = torch.tensor([L2I[l] for l in df["label"]])
    return img, ids, mask, y
T2D = {s: prep(d, s) for s, d in (("train", t2_tr), ("dev", t2_dv), ("test", t2_te))}
''')
code(r'''
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW, lr_scheduler

def t2_eval(model, split):
    img, ids, mask, y = T2D[split]
    probs = cc.t2_predict_proba(model, img.numpy(), ids, mask, DEVICE)
    pred = probs.argmax(1)
    return probs, pred, cc.metrics(y.numpy(), pred, ALL_IDX)

t2_model = cc.DualStreamClassifier(NC, bert_config=bert_cfg).to(DEVICE)
if TRAIN:
    cnt = t2_tr["label"].map(L2I).value_counts().reindex(ALL_IDX).values.astype("float32")
    w = 1.0 / cnt; w /= w.sum() / NC
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, device=DEVICE))
    opt = AdamW([{"params": t2_model.text_encoder.parameters(), "lr": T2_LR * 0.1},
                 *[{"params": m.parameters(), "lr": T2_LR} for m in
                   (t2_model.img_proj, t2_model.txt_proj, t2_model.img_to_txt, t2_model.txt_to_img, t2_model.classifier)]],
                weight_decay=1e-4)
    sched = lr_scheduler.SequentialLR(opt, [lr_scheduler.LinearLR(opt, 1e-6, 1.0, T2_WARMUP),
                                            lr_scheduler.CosineAnnealingLR(opt, T_max=T2_EPOCHS - T2_WARMUP, eta_min=1e-7)],
                                      milestones=[T2_WARMUP])
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    g = torch.Generator().manual_seed(SEED)
    dl = DataLoader(TensorDataset(*T2D["train"]), batch_size=T2_BATCH, shuffle=True, num_workers=0, generator=g)
    best_f1, best_state, history = -1.0, None, []
    for epoch in range(1, T2_EPOCHS + 1):
        t0 = time.time(); t2_model.train(); tot = 0.0
        for img, ids, mask, y in tqdm(dl, desc=f"[T2] epoch {epoch}/{T2_EPOCHS}", leave=False):
            img, ids, mask, y = img.to(DEVICE), ids.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                loss = crit(t2_model(img, ids, mask).float(), y)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(t2_model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); tot += loss.item()
        sched.step()
        _, _, dm = t2_eval(t2_model, "dev")                       # selection uses DEV only
        history.append(dict(epoch=epoch, train_loss=tot / len(dl), **{f"dev_{k}": v for k, v in dm.items() if k != "n"}))
        flag = ""
        if dm["macro_f1"] > best_f1:
            best_f1, flag = dm["macro_f1"], "  <- best (dev macro-F1)"
            best_state = {k: v.detach().cpu().clone() for k, v in t2_model.state_dict().items()}
        print(f"epoch {epoch:2d}  loss {tot/len(dl):.4f}  dev acc {dm['acc']:.4f}  macroF1 {dm['macro_f1']:.4f}  wF1 {dm['weighted_f1']:.4f}  {time.time()-t0:.0f}s{flag}")
    t2_model.load_state_dict(best_state)
    torch.save(best_state, ART / "task2_model.pt")
    json.dump(dict(labels=T2_LABELS, history=history, best_dev_macro_f1=best_f1), open(ART / "task2_labels.json", "w"), indent=1)
    json.dump(history, open(RES / "task2_history.json", "w"), indent=1)
else:
    t2_model.load_state_dict(torch.load(ART / "task2_model.pt", map_location=DEVICE))
    assert json.load(open(ART / "task2_labels.json"))["labels"] == T2_LABELS
''')
code(r'''
# ── Task 2 evaluation (test is evaluated once, after dev-based selection) ─────
for s in ("dev", "test"):
    probs, pred, m = t2_eval(t2_model, s); RESULTS[f"T2_{s}"] = m
    print(f"T2 {s:4s}:", {k: round(v, 4) for k, v in m.items()})
y_te = T2D["test"][3].numpy(); t2_pred = pred
print()
print(classification_report(y_te, t2_pred, labels=ALL_IDX, target_names=T2_LABELS, zero_division=0, digits=4))
print(pd.DataFrame(confusion_matrix(y_te, t2_pred, labels=ALL_IDX), index=[l[:20] for l in T2_LABELS], columns=[l[:8] for l in T2_LABELS]))
t2_out = t2_te[["event_name", "tweet_id", "image_id"]].assign(true=[T2_LABELS[i] for i in y_te], pred=[T2_LABELS[i] for i in t2_pred])
t2_out.to_csv(RES / "preds_task2_test.csv", index=False)
print("\nTest metrics by event (macro over classes present in that event):")
rows = []
for e, d in t2_out.groupby("event_name"):
    labs = sorted(set(d.true) | set(d.pred))
    rows.append(dict(event=e, n=len(d), **{k: round(v, 4) for k, v in cc.metrics(d.true, d.pred, labs).items() if k != "n"}))
print(pd.DataFrame(rows).to_string(index=False))
''')

md(r'''
## 5. Task 3 — Damage severity (CLIP + SBERT → FAISS → 5-NN)
**Fix 2 — leakage.** The original built the FAISS index from *every* annotation file, which contains the test images (each
test query retrieved itself as its own nearest neighbour). The index is now built from the **damage TRAIN split only** and
evaluated on the **damage TEST split**. `image_id` and `tweet_id` are asserted disjoint. The official split separates images, not tweets, so 107 train rows whose tweet also has an image in test are excluded from the index (logged); the unfiltered number is reported as a secondary row.

Also: the annotation files map a missing `image_damage` to a fourth class "Don't know"; the official damage splits contain
only 3 classes, so the old report listed a class that cannot occur. We report the 3 real classes.

**Fix 7 — deterministic vote.** `max(set(labels), key=labels.count)` breaks ties by Python set-hash order. Ties are now broken
by summed similarity of the tied neighbours.
''')
code(r'''
t3 = {s: SPLITS[("damage", s)] for s in ("train", "dev", "test")}
assert set(pd.concat(t3.values()).label) <= set(cc.T3_LABELS), "unexpected damage label"
print({s: d.label.value_counts().to_dict() for s, d in t3.items()})

sbert = None
def sbert_embeddings(split):
    global sbert
    df = t3[split]
    def compute():
        global sbert
        if sbert is None:
            sbert = cc.load_sbert(DEVICE, ART / "sbert" if (ART / "sbert").exists() else cc.SBERT_NAME)
        return cc.embed_texts(df["text"].tolist(), sbert)
    return cc.cached(EMB / f"sbert_damage_{split}.npy", df["image_id"].tolist(), compute)

T3V = {s: cc.t3_vectors(CLIP_EMB[("damage", s)], sbert_embeddings(s)) for s in ("train", "dev", "test")}
print({s: v.shape for s, v in T3V.items()})

META_COLS = ["tweet_id", "image_id", "rel_image", "tweet_text", "text", "event_name", "label"]
# The official split separates IMAGES, but a tweet with several images can have one in train and one in test
# (identical text). To keep tweet_id disjoint too, train rows whose tweet_id occurs in test are left out of the index.
tw_leak = t3["train"]["tweet_id"].isin(set(t3["test"]["tweet_id"]))
print(f"index exclusion: {int(tw_leak.sum())} of {len(tw_leak)} train rows share a tweet_id with the test split "
      f"(reason: same tweet text would be retrievable). By event: {t3['train'][tw_leak].groupby('event_name').size().to_dict()}")
keep = ~tw_leak.to_numpy()
if TRAIN:
    T3_INDEX = faiss.IndexFlatIP(T3V["train"].shape[1]); T3_INDEX.add(T3V["train"][keep])
    t3_meta = t3["train"][META_COLS][keep].reset_index(drop=True)
    faiss.write_index(T3_INDEX, str(ART / "task3_index.faiss")); t3_meta.to_csv(ART / "task3_meta.csv", index=False)
else:
    T3_INDEX = faiss.read_index(str(ART / "task3_index.faiss"))
    t3_meta = pd.read_csv(ART / "task3_meta.csv", dtype=str, keep_default_na=False)
assert T3_INDEX.ntotal == len(t3_meta) == int(keep.sum())
print(f"index: {T3_INDEX.ntotal} vectors (damage TRAIN only, tweet-disjoint from test), dim {T3_INDEX.d}")

# ── leakage assertions ────────────────────────────────────────────────────────
for col in ("image_id", "tweet_id"):
    inter = set(t3_meta[col].astype(str)) & set(t3["test"][col].astype(str))
    print(f"index ∩ test on {col}: {len(inter)}")
    assert not inter, f"LEAK: {len(inter)} shared {col}"
for s in ("dev",):
    print(f"index ∩ {s} on image_id:", len(set(t3_meta.image_id.astype(str)) & set(t3[s].image_id.astype(str))))
print("OK: index and test IDs are disjoint")
''')
code(r'''
def t3_run(split, index=None, meta=None):
    index = index or T3_INDEX; meta = t3_meta if meta is None else meta
    scores, idx = index.search(T3V[split], T3_K)
    labs = meta["label"].to_numpy()
    return [cc.knn_vote(list(labs[i]), list(s))[0] for i, s in zip(idx, scores)]

for s in ("dev", "test"):
    pred = t3_run(s); RESULTS[f"T3_{s}"] = cc.metrics(t3[s]["label"], pred, cc.T3_LABELS)
    print(f"T3 {s:4s}:", {k: round(v, 4) for k, v in RESULTS[f"T3_{s}"].items()})
t3_pred = pred
print()
print(classification_report(t3["test"]["label"], t3_pred, labels=cc.T3_LABELS, target_names=cc.T3_LABELS, zero_division=0, digits=4))
print(pd.DataFrame(confusion_matrix(t3["test"]["label"], t3_pred, labels=cc.T3_LABELS), index=cc.T3_LABELS, columns=[l[:8] for l in cc.T3_LABELS]))
# Secondary (not the headline): index = ALL damage train rows (image-disjoint only; same tweet text may be present)
full = faiss.IndexFlatIP(T3V["train"].shape[1]); full.add(T3V["train"])
sec = t3_run("test", full, t3["train"][META_COLS].reset_index(drop=True))
RESULTS["T3_test_imageonly"] = cc.metrics(t3["test"]["label"], sec, cc.T3_LABELS)
print("T3 test, index = all train rows (image-disjoint only):", {k: round(v, 4) for k, v in RESULTS["T3_test_imageonly"].items()})
t3["test"].assign(pred=t3_pred)[["event_name", "tweet_id", "image_id", "label", "pred"]].rename(columns={"label": "true"}).to_csv(RES / "preds_task3_test.csv", index=False)
''')

md(r'''
## 6. Export (only when `TRAIN=True`) and before/after table
`artifacts/` holds everything the demo loads (offline): Task 1 pipeline, Task 2 weights + labels, tokenizer, DistilBERT config,
local copies of CLIP and SBERT, Task 3 train-only FAISS index + metadata, and `config.json`. `artifacts/` is git-ignored.
''')
code(r'''
if TRAIN:
    clip, proc = get_clip(); clip.save_pretrained(ART / "clip"); proc.save_pretrained(ART / "clip")
    if sbert is None: sbert = cc.load_sbert(DEVICE)
    sbert.save(str(ART / "sbert"))
    json.dump(dict(seed=SEED, clip=cc.CLIP_NAME, sbert=cc.SBERT_NAME, bert=cc.BERT_NAME, t2_labels=T2_LABELS,
                   t2_max_len=T2_MAX_LEN, t2_d_model=256, t2_heads=4, t2_dropout=0.2, t3_labels=cc.T3_LABELS, t3_k=T3_K,
                   t1_labels=cc.T1_LABELS, t3_index_size=int(T3_INDEX.ntotal), t3_dim=int(T3_INDEX.d)),
              open(ART / "config.json", "w"), indent=1)
    print("exported:", sorted(p.name for p in ART.iterdir()))
else:
    print("TRAIN=False: artifacts loaded, nothing exported.")
''')
code(r'''
OLD = {"T1": dict(acc=0.7944, macro_f1=np.nan, weighted_f1=0.7970),
       "T2": dict(acc=0.7807, macro_f1=0.4696, weighted_f1=np.nan),
       "T3": dict(acc=0.6784, macro_f1=np.nan, weighted_f1=0.6964)}
NOTE = {"T1": "same data; only the report labels changed",
        "T2": "corrected pipeline = path fix + HF CLIP/QuickGELU + cached embeddings + 10-epoch cap + AMP (delta not attributable to any one change)",
        "T3": "index = damage TRAIN minus rows sharing a tweet_id with test; evaluated on damage TEST; 3 real classes; deterministic tie-break"}
NAME = {"T1": "T1 Informative (text-only)", "T2": "T2 Humanitarian (8 cls)", "T3": "T3 Damage (3 cls)"}
rows = []
for t in ("T1", "T2", "T3"):
    rows.append(dict(task=NAME[t], version="old (leaky/buggy, as reported)", n_test=np.nan, **OLD[t], notes="numbers from capstone-final.ipynb; blank = not reported then"))
    m = RESULTS[f"{t}_test"]
    rows.append(dict(task=NAME[t], version="corrected pipeline", n_test=m["n"], accuracy=None, **{"acc": m["acc"], "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"]}, notes=NOTE[t]))
m = RESULTS["T3_test_imageonly"]
rows.append(dict(task=NAME["T3"], version="corrected, secondary: official split (image-disjoint only)", n_test=m["n"], acc=m["acc"],
                 macro_f1=m["macro_f1"], weighted_f1=m["weighted_f1"], notes="index = all damage-train rows; 62 test tweets also have another image in train"))
ba = pd.DataFrame(rows).drop(columns="accuracy").rename(columns={"acc": "accuracy"})
ba.to_csv(RES / "before_after.csv", index=False)
json.dump({k: {a: float(b) for a, b in v.items()} for k, v in RESULTS.items()}, open(RES / "metrics.json", "w"), indent=1)
print(ba.round(4).drop(columns="notes").to_string(index=False))
print(f"\nSaved results/before_after.csv, results/metrics.json  |  notebook wall-time {time.time()-T_START:.0f}s")
''')

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
nbf.write(nb, "capstone-v2.ipynb")
print("wrote capstone-v2.ipynb with", len(cells), "cells")
