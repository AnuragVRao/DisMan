# v1 configuration (recovered from `capstone-final.ipynb`)

Source: `capstone-final.ipynb` (the original Kaggle notebook, unmodified). It has 31 cells (0-30). Citations use **cell N** (0-indexed
cell position) and quote the code. Logged outputs in the notebook are used to confirm settings where a quote alone is not enough.
No `CLAUDE.md` exists in the repo; this was recovered from the notebook, `README.md`, `RESULTS.md` and `capstone-v2.ipynb` only.

Items marked **AMBIGUOUS** could not be recovered with certainty; they are collected in the last section.

## Environment (from notebook)

* Kaggle kernel, Python 3.12 (`/usr/local/lib/python3.12/dist-packages/...` in warnings), `Device: cuda` (cell 4 output). GPU model not recorded.
* Seeding, cell 4, done once at import time and never again:
  `SEED = 42` / `random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)`. No `cudnn.deterministic`, no DataLoader generator/worker seed.
* Paths, cell 6: `BASE = Path("/kaggle/input/datasets/chhitanshushekhar/capstone-dataset/capstone_dataset/CrisisMMD_v2.0")`,
  `IMG_DIR = BASE / "data_image"`, `SPLIT_DIR = BASE / "crisismmd_datasplit_all" / "crisismmd_datasplit_all"`, `ANNOT_DIR = BASE / "annotations"`.
* Text cleaning (all tasks), cell 8: `clean_tweet` strips leading `RT`, URLs, maps `@\w+` to `@user`, strips `#`, expands 10 abbreviations, collapses whitespace.
  `v2` reuses this unchanged (`crisis_common.py`: "unchanged from the original notebook").

## Image path resolution (T2 and the T3 evaluation split)

Cell 17 (`t2_load`) and cell 28 (`t3_load_split`) both use:

```python
first = str(row['image']).split(',')[0].strip().lstrip('data_image/')
full  = IMG_DIR / first
return str(full) if full.exists() else None
```

`str.lstrip` strips a *character set* (`d a t _ i m g e /`), not a prefix. So `iraq_iran_earthquake/...` loses `i` and becomes `raq_iran_earthquake/...`
and `mexico_earthquake/...` loses `me` and becomes `xico_earthquake/...`. These paths do not exist, so `resolve_img` returns `None`
and the row is dropped (`df[df['img_path'].notna()]`).

I re-ran this logic on the raw split files in `data/` to confirm it reproduces the notebook's logged sizes:

| Split | Rows in TSV | Resolving under `lstrip` | v1 logged (cell 17 / 28 output) |
|---|---|---|---|
| T2 train | 13608 | 12124 | 12124 |
| T2 dev | 2237 | 1993 | 1993 |
| T2 test | 2237 | 1988 | 1988 |
| T3 damage test | 529 | 485 | 485 |

In the damage test set the 44 dropped rows are iraq_iran_earthquake (23) and mexico_earthquake (21).
Tasks 1 and the T3 *index* do not use this path code (see below), so they were not affected.

## Task 1 (informative vs not informative)

Cells 10-13.

* Data: `pd.read_csv(TASK1_*, sep="\t").drop_duplicates(subset=["image_id"])` for train, dev, test. Text: `fillna("").apply(clean_tweet)` on `tweet_text`.
  No image path resolution, so no rows lost. Logged: Train 13608, Dev 2237, Test 2237 (cell 10 output).
* Label column: `label_text`. Train distribution logged: informative 9638, not_informative 3970.
* Model (cell 11):
  ```python
  Pipeline([
      ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
      ("clf",   LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")),
  ])
  t1_model.fit(t1_train["tweet_text"], t1_train["label_text"])
  ```
  Other settings are sklearn defaults (solver `lbfgs`, L2 penalty, `lowercase=True`, no stop-word list). **The dev split is loaded but never used**: no tuning.
* Test metrics logged (cell 13): accuracy 0.7944, weighted F1 0.7970. These equal the v2 README T1 row (0.7944 / 0.7970), so T1 is unchanged between v1 and v2 as far as the headline goes.
* Cosmetic v1 bug: `classification_report(..., target_names=["not_informative", "informative"])` names the classes in the wrong order
  (sklearn sorts labels alphabetically: informative first), so the per-class labels in the printed report are swapped. Accuracy and weighted F1 are unaffected.

## Task 2 (humanitarian, 8 classes)

Cells 15-21.

| Setting | v1 value | Source |
|---|---|---|
| Epochs | **30** | cell 16 `T2_EPOCHS = 30` |
| Batch size | 32 (train, dev, test) | cell 16 `T2_BATCH = 32`; cell 18 `DataLoader(..., batch_size=T2_BATCH, ..., num_workers=2, pin_memory=True)`, `shuffle=True` for train only |
| Max text length | 128, `padding='max_length'`, truncation | cell 16 `T2_MAX_LEN = 128`; cell 18 |
| Base LR | 1e-4 | cell 16 `T2_LR = 1e-4` |
| Optimiser | AdamW, `weight_decay=1e-4`, per-group LR | cell 20 |
| LR groups | DistilBERT `T2_LR * 0.1` (1e-5); `img_proj`, `txt_proj`, `img_to_txt`, `txt_to_img`, `classifier` all `T2_LR` | cell 20 `optim.AdamW([{'params': t2_model.text_encoder.parameters(), 'lr': T2_LR * 0.1}, ...], weight_decay=1e-4)` |
| Scheduler | `SequentialLR`: `LinearLR(opt, 1e-6, 1.0, T2_WARMUP)` for 3 epochs, then `CosineAnnealingLR(opt, T_max=T2_EPOCHS - T2_WARMUP, eta_min=1e-7)`; `milestones=[T2_WARMUP]`; **stepped once per epoch** | cell 20; `t2_scheduler.step()` after each epoch loop; `T2_WARMUP = 3` (cell 16) |
| Gradient clipping | `clip_grad_norm_(t2_model.parameters(), 1.0)` | cell 20 |
| Loss | `CrossEntropyLoss(weight=...)`, weights = inverse train class count, rescaled to mean 1: `class_weights = 1.0 / label_counts_arr; class_weights /= class_weights.sum() / T2_NUM_CLASSES`. No label smoothing | cell 20 |
| Mixed precision | none (no `autocast`/`GradScaler`) | cells 20 |
| Feature caching | none; images are re-encoded through CLIP each step | cell 18 `__getitem__` + cell 19 `forward` |
| Model | frozen CLIP visual -> `img_proj` (Linear 512->256 + LayerNorm); DistilBERT -> `txt_proj` (Linear 768->256 + LayerNorm); `img_to_txt` and `txt_to_img` cross-attention blocks (`d_model=256`, 4 heads, dropout 0.2); concat -> MLP (Dropout, Linear 512->256, ReLU, Dropout, Linear 256->8). Params 155,202,056 (logged) | cells 16, 19 |
| CLIP library / activation | **`open_clip.create_model_and_transforms("ViT-B/32", pretrained='openai')`**; the vision tower is `clip_model_t2.visual`, output = `visual.output_dim` (512, the projected vector). Warning logged in cell 15: *"QuickGELU mismatch between final model config (quick_gelu=False) and pretrained tag 'openai' (quick_gelu=True)"*, i.e. v1 ran the OpenAI weights with plain GELU instead of QuickGELU. This is the "wrong activation" that v2 fixes with HF `CLIPModel` | cell 15; v2 `capstone-v2.ipynb` cell 11 ("Fix 6") |
| CLIP frozen? | **Yes**: `for p in self.vision_encoder.parameters(): p.requires_grad = False` (weights frozen; the module is still put in train mode by `t2_model.train()`, which has no effect on a ViT without dropout/BN) | cell 19 |
| CLIP preprocessing | the `clip_preprocess_t2` transform returned by open_clip; unreadable image -> grey `Image.new('RGB', (224, 224), (128, 128, 128))` | cells 15, 18 |
| Training label | `df['label'] = df['label_image']` (image label), while the "Unknown" filter is applied to `label_text`: `df = df[df['label_text'] != 'Unknown']` | cell 17 `t2_load` |
| Classes | 8: Affected individuals, Infrastructure and utility damage, Injured or dead people, Missing or found people, **Not humanitarian**, Other relevant information, Rescue/volunteering/donation effort, Vehicle damage | cell 17 output |
| Checkpoint rule | **best dev macro-F1, strict `>`, over all 30 epochs; checkpoint overwritten at each new best, reloaded before test** (no early stopping, no epoch cap) | cell 20 `if f1 > best_t2_f1: best_t2_f1 = f1; torch.save(...)`; cell 21 `load_state_dict(torch.load(...))` |
| Logged outcome | **0.5741 = macro-F1 on the T2 DEV split** (1993 rows, the best epoch, epoch 20 in the log; this is the checkpoint-selection score). **0.7807 = accuracy and 0.4696 = macro-F1 on the T2 TEST split** (1988 rows, the buggy v1 test set, reloaded best checkpoint). The 0.4696 test macro-F1 is *not* comparable to 0.5741: different split, and dev is the selected maximum | cells 20, 21, 30 outputs |

Data used by v1 T2: train 12124 / dev 1993 / test 1988 (earthquake events missing; see path section). The 3-epoch linear warm-up and 27-epoch cosine
depend on `T2_EPOCHS = 30`, so any change of the epoch budget also changes the LR schedule.

## Task 3 (damage severity)

Cells 22-28.

### What the v1 FAISS index contained

* Rows come from **every** `.tsv` in `ANNOT_DIR`, concatenated with no split filter and no de-duplication:
  ```python
  for f in os.listdir(ANNOT_DIR):
      if f.endswith('.tsv'):
          dfs.append(pd.read_csv(ANNOT_DIR / f, sep='\t'))
  df = pd.concat(dfs, ignore_index=True)
  ```
  The index therefore holds the train, dev **and test** images of the damage task, plus every other annotated image that has no damage label.
  Cell 23 output: *"Task 3 — Total samples with valid images: 18082"* and cell 26 output: *"FAISS index ready — 18082 vectors, dim=896"*.
* Paths here use `str(BASE / p)` with no `lstrip`, so all 7 events (including both earthquakes) are in the index. Verified on the raw data: all 18082 annotation rows resolve.
* **All 529 corrected damage-test images are in this index** (matched by `image_id`). v1's evaluation used 485 of them (the `lstrip` bug removed the 44 earthquake rows), so 485 test queries could retrieve themselves.
  The cell 27 demo shows it directly: `Scores: [1. 0.474 0.454 0.45 0.449]` (top hit is the query itself, similarity 1.0).

### Embedding, k, voting

* Image: HF-style CLIP, `T3_CLIP.get_image_features(**inputs)` with `T3_CLIP_PROC(images=..., return_tensors="pt")`, then L2-normalised (512-d). **Which CLIP checkpoint is AMBIGUOUS** (see below).
* Text: `SentenceTransformer("all-MiniLM-L6-v2")`, `encode(texts, batch_size=64, normalize_embeddings=True)` (384-d), over `clean_text`; empty text becomes the string `"no text"` in the index, but a zero vector for a query with empty text.
* Fusion (cell 26): `np.concatenate([image_embedding, text_embedding])` (896-d), `faiss.normalize_L2`, `faiss.IndexFlatIP` (exact inner product; cosine after normalisation). Image and text are weighted equally.
* **k = 5** (cell 27 `def t3_predict(image_path, text="", k=5)`; cell 28 calls `k=5`).
* **Voting**: unweighted majority of the 5 neighbour labels, `return max(set(labels), key=labels.count)`. Similarity scores are not used.
  Tie-break (e.g. a 2-2-1 vote) is whichever tied label comes first in `set` iteration order, which for strings depends on `PYTHONHASHSEED` and so can differ between processes. v1 T3 is therefore not strictly deterministic when ties occur.

### Evaluation set and labels

* Cell 28 `t3_load_split(TASK3_TEST)` -> 485 rows (see path section). Query = `T3_CLIP_PROC` image embedding + `clean_text` SBERT embedding, built the same way as the index.
  Label column: the damage TSV has `label`, not `label_image`, so `label_col = 'label_image' if 'label_image' in df.columns else 'label'` resolves to `label`.
* Unreadable image fallback in T3 is a **black** 224x224 image (`Image.new("RGB", (224, 224))`), not grey as in T2.
* Logged v1 result (cell 28), on the 485-row v1 **test** split: test accuracy 0.6784, test weighted F1 0.6964; the report includes a fourth class "Don't know or can't judge" with support 0 (precision/recall/F1 0.00), which also drags v1's macro average (0.46).

### Exactly what the label pollution was

The index labels come from the annotation column `image_damage`, mapped with `t3_normalise`:

```python
df['image_damage']    = df['image_damage'].fillna('dont_know_or_cant_judge')
df['damage_severity'] = df['image_damage'].apply(t3_normalise)
```

Any missing `image_damage` is turned into the class **"Don't know or can't judge"**. I counted the raw annotation files:

| `image_damage` in annotations | Rows | Becomes v1 label |
|---|---|---|
| missing (NaN) | 14,455 | Don't know or can't judge |
| `dont_know_or_cant_judge` (explicit) | 101 | Don't know or can't judge |
| `severe_damage` | 2,212 | Severe damage |
| `mild_damage` | 839 | Mild damage |
| `little_or_no_damage` | 475 | Little or no damage |
| **Total** | **18,082** | |

So 14,556 of 18,082 index rows (80.5%) carry the "Don't know" label, matching the cell 23 output (`Don't know or can't judge 14556`). The official damage splits contain
only the three real classes (train 2468, dev 529, test 529 rows; test: severe 332, mild 126, little 71), so "Don't know" can never be a correct answer on test.
The 14,455 NaN rows are not "don't know" judgements at all: they are images that were never asked the damage question
(8,706 are `not_informative` images and 5,749 are `informative` images without a damage label; cross-tab of `image_info` vs `image_damage`).
Of the 101 explicit "don't know" rows, 99 are informative and 2 are not.
Note the plan's phrase "irrelevant-class pollution" describes this: the index is dominated by unlabeled/irrelevant rows pooled into a fake fourth class.

## Resolved after Step 0 (verified, not assumed)

1. **T2 image augmentation: yes, v1 used random augmentation, on train, dev AND test.** Cell 15 unpacks
   `clip_model_t2, clip_preprocess_t2, _ = open_clip.create_model_and_transforms(...)`. open_clip returns `(model, preprocess_train, preprocess_val)`,
   so v1 kept the *train* transform. With open_clip 3.3.0 (installed here) it is
   `RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.75, 1.333), bicubic) -> ToTensor -> Normalize`; the val transform
   (`Resize(224) -> CenterCrop`) was discarded. Cell 18 applies `clip_preprocess_t2` to all three splits, so even the dev and test scores
   in v1 are measured on randomly cropped images. (The Kaggle open_clip version is not recorded; the train transform has had this default since open_clip 2.x.)
2. **CLIP train/eval mode: no effect.** v1 left the CLIP tower in train mode (`t2_model.train()`); it has no dropout. Measured max |train - eval| output difference: 2.9e-6.
3. **Consequence: caching CLIP features is NOT exact for L0-L2.** Features change with every random crop, so the epoch-1 cache check was not run (it would differ by construction).
   Caching only becomes exact once the transform is deterministic, which is what v2 did (HF `CLIPProcessor`: resize + centre crop). So v2's "cached embeddings" step
   silently also removed the augmentation. L0-L2 therefore run CLIP online with the v1 random-crop transform.
4. **T3 CLIP checkpoint verified.** Rebuilt the 18,082-row v1 index with HF `openai/clip-vit-base-patch32` (QuickGELU) + SBERT and searched all rows for the demo's top-5 scores
   `[1, 0.474, 0.454, 0.45, 0.449]`: exactly one row reproduces them (image 873936011822333953_0, scores 1.0000 / 0.4743 / 0.4537 / 0.4500 / 0.4489, label and all five neighbours "Don't know"
   as in the demo). The next-closest row misses by 0.0035 (12x worse). So the v1 T3 CLIP is the HF `openai/clip-vit-base-patch32`; the open_clip fallback was not needed.
   Script: `ablation/t3_verify_clip.py`.
5. **v2 schedule vs v1 beyond the epoch count.** Warm-up (`LinearLR(1e-6 -> 1.0, 3 epochs)`), `CosineAnnealingLR(T_max = epochs - 3, eta_min = 1e-7)`, `SequentialLR` milestone 3, AdamW and
   its per-group LRs, weight decay 1e-4, grad clip 1.0, loss weights and the strict `>` best-dev-macro-F1 rule are identical. The only schedule change is `T2_EPOCHS` 30 -> 10 (cosine over 27 -> 7 epochs).
   So L3 = "epoch budget 30 -> 10" and nothing else. Everything else v1 -> v2 differs in the L4 bundle: deterministic transform (no augmentation), cached features, AMP (fp16 autocast + GradScaler),
   seeded DataLoader generator, `num_workers=0`, eval under autocast.
6. **v2's T3 index** (`artifacts/task3_meta.csv`): 2,361 rows = damage train (2,468) minus the 107 train rows whose `tweet_id` also occurs in the test split. Verified by image_id set equality.
   Tie-break: `cc.knn_vote` (vote share, then summed similarity). Used for every T3 rung.

## Ambiguous / not recoverable (remaining)

1. **Package versions and GPU.** Only Python 3.12 and CUDA availability are recorded. Library versions are not (`!pip install -q open_clip_torch sentence-transformers faiss-cpu`, unpinned).
2. **Exact v1 T2 numerical reproducibility.** Seed 42 is set once at the top; init and shuffle order depend on earlier RNG use and on non-deterministic CUDA kernels. Random crops add further run-to-run variation.
   L0/L1 reproduce the *configuration*, not the logged 0.5741 dev / 0.4696 test numbers.
3. **v1 T3 tie-break** was hash-order dependent, so v1's logged 0.6784 / 0.6964 are one draw. Ladder rungs use v2's deterministic rule, as agreed.
4. **Epoch 27 line** is missing from the extracted log (progress-bar artefact); no effect on any setting.
5. **open_clip version used on Kaggle** (see item 1 above; assumed to return the same random-crop train transform).

## Library versions used for the ablation re-runs (not v1's; v1's are unrecorded)

Python 3.13 (`C:\Users\anura\AppData\Local\Programs\Python\Python313`), torch 2.13.0+cu126, torchvision 0.28.0+cu126, transformers 4.57.1,
open_clip_torch 3.3.0, timm 1.0.30, sentence-transformers 5.1.2, faiss-cpu 1.12.0, scikit-learn 1.7.0, pandas 2.3.0; GPU: NVIDIA GeForce RTX 4050 Laptop (6 GB).
`open_clip_torch`, `timm`, `ftfy`, `wcwidth` were added for this study with `pip install --no-deps` (the v2 requirements did not include them).

## Ladder conventions added during the study

* **L4a** = L3b + AMP with pretrained DistilBERT (v2 as intended). **L4** = v2 exactly (randomly initialised DistilBERT). L4a -> L4 differs only in the encoder initialisation.
* **Shuffle order.** v2's `tqdm(dl)` makes `tqdm.auto` call `iter(dl)` twice, shifting the shuffle by one generator draw per epoch (see `docs/ablation_findings.md`).
  Every rung in `ablation/t2_train.py` reproduces that shift. It is equivalent to changing the seed, not a systematic effect, so it gets no rung of its own;
  the seeds {42, 43, 44} average over it.
