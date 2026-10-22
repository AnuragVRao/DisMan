# CrisisMMD pipeline: audit, corrections and results

> Draft. Sections still to be added in Block 4: architecture description, figures, reference-baseline row, full limitations pass.

## 1. Bug audit

| # | Bug | Why it mattered | Fix |
|---|---|---|---|
| 1 | `str.lstrip('data_image/')` strips a **character set**, not a prefix. `iraq_iran_earthquake` became `raq_iran_earthquake` and `mexico_earthquake` became `xico_earthquake`. | The image lookup failed, so those two events were silently dropped, along with about 11% of every split (e.g. informative train 12112/13608 rows survived). | `removeprefix('data_image/')`. After the fix 100% of images resolve in every split; asserted. |
| 2 | Task 3 FAISS index was built from **all** annotation files, which include the test images. | Each test query could retrieve itself (see section 2). | Index built from the damage **train** split only; evaluated on damage **test**; `image_id` and `tweet_id` intersections asserted empty. |
| 3 | Task 1 report: sklearn sorts labels alphabetically (`informative`, `not_informative`) but `target_names` was passed reversed. | Per-class precision/recall were attached to the wrong names. Headline accuracy was unaffected. | `labels=` passed explicitly. |
| 4 | `T3_CLIP` / `T3_CLIP_PROC` used but never defined. | The notebook only ran because of stale Kaggle kernel state. | Defined explicitly (HF `openai/clip-vit-base-patch32` + `CLIPProcessor`). |
| 5 | No way to run without retraining. | Reproduction required hours of training. | `TRAIN` flag; `False` loads from `artifacts/`. |
| 6 | **QuickGELU mismatch.** The old Task 2 loaded `open_clip` ViT-B/32 with `pretrained='openai'`, which warned `QuickGELU mismatch (quick_gelu=False) vs pretrained tag 'openai' (quick_gelu=True)`. | The OpenAI weights were trained with QuickGELU, so the old image features used the wrong activation. | HF `CLIPModel`, which uses QuickGELU for this checkpoint (asserted in the notebook). One CLIP now serves Tasks 2 and 3. |
| 7 | Task 3 vote used `max(set(labels), key=labels.count)`. | Ties were broken by Python set-hash order. | Ties broken by summed neighbour similarity (deterministic). |
| 8 | Task 3 split overlap by tweet (found during the fix): the official split separates images, not tweets, so a tweet with two images can appear in two splits with identical text. | 62 test tweets (67 rows) have a twin in train; their text embeddings match exactly. | The 107 train rows sharing a tweet with test are left out of the index (logged). The official-split number is reported as a secondary row. |

## 2. Why corrected Task 3 dropped by less than one might expect

The old Task 3 index had two opposing defects, which together probably explain why the old number was not dramatically higher than the corrected one. This is an explanation of the likely mechanism, **not a measured ablation**.

* **Inflation (self-match).** The index contained the test images themselves, so each test query retrieved itself as a neighbour. That is one guaranteed-correct vote out of five.
* **Deflation (absent class).** The annotation files contain 18,082 rows, and 14,455 of them have no damage label, which the old code mapped to "Don't know or can't judge". The old index therefore held 14,556 "Don't know" vectors (80.5% of the index), while the official damage splits contain no such class. Neighbours voting for a class that cannot occur in the test set can only cost accuracy.

These two effects pull in opposite directions and plausibly roughly cancelled. Removing both gives a result that is smaller on accuracy and weighted-F1 but is at least a valid evaluation.

## 3. Before/after table (test split)

| Task | Version | Accuracy | Macro-F1 | Weighted-F1 | n (test) |
|---|---|---|---|---|---|
| T1 Informative (text-only) | old | 0.7944 | not reported | 0.7970 | not reported |
| | corrected | 0.7944 | 0.7513 | 0.7970 | 2237 |
| T2 Humanitarian (8 classes) | old | 0.7807 | 0.4696 | not reported | not reported |
| | corrected pipeline* | 0.7975 | 0.5865 | 0.7968 | 2237 |
| T3 Damage (3 classes) | old | 0.6784 | not reported | 0.6964 | not reported |
| | corrected, tweet-disjoint index (headline) | 0.6673 | 0.4914 | 0.6202 | 529 |
| | corrected, official split (image-disjoint only), secondary | 0.6749 | 0.5091 | 0.6309 | 529 |

\* "Corrected pipeline" for Task 2 bundles **five** changes: the path fix, HF CLIP / QuickGELU fix, cached embeddings, a 10-epoch cap, and AMP. The delta is not attributable to any single one of them.

**Footnote (old vs corrected are not on identical test sets).** The old rows were evaluated on test sets that were missing the Iraq-Iran and Mexico earthquake samples because of the path bug (e.g. old humanitarian test 1988 of 2237 rows). Old and corrected numbers therefore do not describe the same test sets.

Task 1 reproduces the old accuracy and weighted-F1 to three decimals (it is text-only and never used image paths); only the per-class labels changed.

## 4. Split integrity

* `image_id` is disjoint across train/dev/test in all three tasks (asserted).
* The Task 1 and Task 2 official splits were verified **tweet-disjoint**; one Task 2 test row has identical cleaned text to a train row, which is negligible.
* Task 3 (damage) has one row per image, so tweets with several images straddle splits: 62 tweets (67 test rows) are shared between train and test, and **16 tweets are shared between dev and test**. Test results are unaffected, since nothing in Task 3 is fitted or tuned on dev (k is fixed at 5). The headline index excludes train rows sharing a tweet with test, and the Task 3 dev number is computed with an index that excludes train rows sharing a tweet with dev. Counts per pair are in `results/split_overlap.csv`. Effect on the Task 3 dev number (measured): accuracy / macro-F1 / weighted-F1 = 0.6730 / 0.4972 / 0.6379 with the full train index, versus 0.6635 / 0.4832 / 0.6291 with the tweet-disjoint index, i.e. twins inflated dev slightly.

## 5. Limitations (partial list; to be completed in Block 4)

* **Task 3 Mild-damage recall is 18%** (Severe 92%, Little or no damage 32%). Severe dominates the train split (1548 of 2468 rows), so the k=5 majority vote is biased towards it.
* **Task 2 cross-attention.** The image side is one pooled CLIP vector, so the txt-to-img attention attends over a single key and is effectively a projection. It was not redesigned.
* Task 2 has tiny classes (e.g. "Missing or found people": 3 test samples, F1 = 0), so macro-F1 is noisy.
* Task 1 and Task 2 labels come from different annotators/modalities (Task 1 uses the tweet-text label, Task 2 the image label), so the two can legitimately disagree on the same tweet.

## 6. Future work

* **Distance-weighted / class-balanced voting** for Task 3, tuned on dev. **If anything is ever tuned on Task 3 dev, the 16 dev/test shared tweets must be excluded first.**
* Leave-one-event-out generalisation.
* Zero-shot VLM (LLaVA).
* Patch-token cross-attention for Task 2.
* Satellite imagery (xBD).
