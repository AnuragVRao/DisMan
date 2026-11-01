# CrisisMMD Disaster-Triage Pipeline

An audited, corrected and extended multimodal pipeline for disaster tweets, built on **CrisisMMD v2.0**. Given a tweet's image and text it answers three questions and shows the supporting evidence through a live Gradio demo.

| Task | Question | Model | Input |
|---|---|---|---|
| 1 | Is the tweet informative? | Fine-tuned BERTweet (TF-IDF + logistic regression kept as a fallback) | text |
| 2 | Which humanitarian category? (8 classes) | Frozen CLIP ViT-B/32 image vector + fine-tuned DistilBERT, cross-attention, MLP | image + text |
| 3 | How severe is the damage? (3 classes) | CLIP image + SBERT text vector, logistic probe; FAISS 5-NN retrieves similar past cases as evidence | image + text |

The project began as a Kaggle notebook (`capstone-final.ipynb`, kept unmodified). We audited it, fixed a set of verified bugs, rebuilt it as `capstone-v2.ipynb`, and then ran a series of pre-registered ablations and model explorations on top of it.

---

## Contents
1. [Results](#1-results)
2. [The audit: what was wrong with the original notebook](#2-the-audit)
3. [Ablation and exploration studies](#3-ablation-and-exploration-studies)
4. [Live demo](#4-live-demo)
5. [Reproducing the work](#5-reproducing-the-work)
6. [Repository layout](#6-repository-layout)
7. [Limitations and future work](#7-limitations-and-future-work)
8. [Data and references](#8-data-and-references)

---

## 1. Results

Corrected v2 baseline, test split, seed 42:

| Task | Accuracy | Macro-F1 | Weighted-F1 | n |
|---|---|---|---|---|
| T1 Informative (text-only, TF-IDF + LR) | 0.7944 | 0.7513 | 0.7970 | 2237 |
| T2 Humanitarian (8 classes) | 0.7975 | 0.5865 | 0.7968 | 2237 |
| T3 Damage severity (3 classes, tweet-disjoint index) | 0.6673 | 0.4914 | 0.6202 | 529 |

Models that the demo now serves:

* **Task 1:** fine-tuned BERTweet, selected over TF-IDF + LR by a rule fixed before the runs. Test weighted-F1 0.8382 ± 0.0022 and macro-F1 0.7970 ± 0.0009 (3-seed means); the deployed checkpoint scores 0.8443 / 0.8024.
* **Task 2:** the L4a seed-42 model from the ablation study (test accuracy 0.8033, macro-F1 0.6080).
* **Task 3:** a hybrid. A logistic probe predicts the level (macro-F1 0.6146 against 0.4914 for the plain 5-NN vote; paired bootstrap difference +0.123, 95% CI [+0.062, +0.186]). The five nearest past cases are shown as evidence, and probe/neighbour disagreement (34.8% of test queries) is flagged for human review.
* Confidences are temperature-scaled (`demo/calibration.json`).

Task 3 is lower than the original notebook's number on purpose: the old evaluation was inflated by test-set leakage and polluted by a class that does not exist in the test split (section 2).

### Can damage severity be classified with no training? (CLIP zero-shot)

Prompts were fixed before any result existed, the best of three prompt sets was chosen on dev, and test was scored once (529 damage-test rows).

| Method | Accuracy | Macro-F1 | Recall Severe / Mild / Little |
|---|---|---|---|
| Task 3 k-NN, image + text | 0.667 | 0.491 | 92% / 18% / 32% |
| Task 3 k-NN, CLIP image only | 0.675 | 0.507 | 91% / 27% / 28% |
| CLIP zero-shot (prompt set E3) | 0.303 | 0.299 | 26% / 37% / 38% |
| Zero-shot + bias terms fitted on dev | 0.567 | 0.397 | 74% / 37% / 8% |
| Always "Severe damage" | 0.628 | 0.257 | 100% / 0% / 0% |
| Stratified random (expected) | 0.469 | 0.333 | 63% / 24% / 13% |

Zero-shot is below chance, so labelled data is what makes Task 3 work. Adding tweet text to the k-NN makes no meaningful difference. Because "always Severe" already reaches 0.628 accuracy, macro-F1 is the headline metric throughout. Files: `results/s1_*`.

---

## 2. The audit

Full evidence is in [RESULTS.md](RESULTS.md). The main findings:

1. **Path bug.** `lstrip('data_image/')` strips a character set, not a prefix, which silently dropped the Iraq-Iran and Mexico earthquake events (about 11% of every split). Fixed with `removeprefix`; every image now resolves, asserted.
2. **Task 3 leakage and pollution.** The FAISS index was built from all annotations, so every test query retrieved itself, and 80% of the index was a "Don't know" class that does not exist in the test split. The index is now built from the damage-train split only.
3. **Tweet-level overlap.** The official split separates images, not tweets, so multi-image tweets appear in both train and test with identical text (62 test tweets have a twin in train). These are excluded from the index.
4. **Smaller fixes.** Task 1 report labels were in the wrong order; `T3_CLIP` was used but never defined; a QuickGELU mismatch affected the old `open_clip` image features; the k-NN vote tie-break depended on set-hash order (now deterministic); a `TRAIN` flag now allows running without retraining.

Task 3 ladder (v1 index to v2), 529 test queries:

| Rung | Acc. | Macro-F1 |
|---|---|---|
| L0 v1 index (includes test, "Don't know" pollution) | 0.7221 | 0.6880 |
| L1 test rows removed | 0.5028 | 0.3777 |
| L2 "Don't know" rows removed | 0.6767 | 0.5037 |
| L3 dev rows removed (train only) | 0.6749 | 0.5091 |
| L4 tweet siblings removed (= v2) | 0.6673 | 0.4914 |

The original notebook's Task 3 number looked reasonable only because two opposing defects (self-match inflation, absent-class deflation) roughly cancelled.

---

## 3. Ablation and exploration studies

All studies follow the same rules: one corrected test set, model selection on **dev only**, test scored afterwards, seeds 42-44 for trained models, selection rules committed before the runs (`ablation/SELECTION_RULE*.md`), and paired bootstrap intervals for comparisons. Per-run results are JSON files under `results/ablation/`. Detailed write-ups: [docs/ablation_findings.md](docs/ablation_findings.md), [ABLATION_PLAN.md](ABLATION_PLAN.md), [docs/report_dossier.md](docs/report_dossier.md).

| Study | What it covers | Headline finding |
|---|---|---|
| **A. v1 to v2 ladder** | Task 2 and Task 3 rungs from the original notebook to the corrected pipeline | Task 3 changes explained above; Task 1 reproduces exactly |
| **B. Component ablation** | Image-only, text-only, concat, no text-to-image attention; Task 3 k, voting scheme, probe | Image carries most of the signal for Task 2; text adds little |
| **C. Robustness** | Missing modality, calibration, leave-one-disaster-out | Task 2 drops about 0.07 macro-F1 on an unseen event (95% CI [-0.109, -0.036]) |
| **D. Extra analyses** | Label-agreement subsets, bootstrap CIs, duplicate-image check, class imbalance and loss comparison | Most loss and imbalance choices are statistically tied |
| **Architecture search** | Image backbone, text encoder, head, hyperparameters, fine-tuned vs frozen | CLIP L/14 is the only resolvable pick; other choices are ties |
| **Task 1 exploration** | TF-IDF models, BiLSTM + GloVe, BERTweet, image heads, early and late fusion | BERTweet beats TF-IDF + LR by +0.0195 dev weighted-F1 (CI [+0.005, +0.034]); fusion adds about +0.006 on test |
| **Task 2 exploration** | Classical and BERTweet text models, image heads, fusion | Image-only CLIP L/14 (test macro-F1 0.644) matches or beats fused models; text-only tops out near 0.31 |
| **Preprocessing A/B (Task 1)** | `preprocessing/` branch text and a minimal-cleaning variant against raw text | No arm meets the pre-registered "helps" rule; raw tweet text is kept |

Key take-aways: Task 2 macro-F1 is limited by tiny classes (for example "Missing or found people" has 3 test rows); most architectural differences are within noise at this test size; and Task 1 image and text labels disagree on about 31% of test rows, so Task 1 (text label) and Task 2 (image label) can legitimately disagree.

### Preprocessed datasets

`preprocessing/` contains leakage-checked, ready-to-train versions of the data for all three tasks: cleaned text (BERTweet and TF-IDF variants), verified and resized images, and weighted, balanced, oversampled or undersampled training splits. See [preprocessing/README.md](preprocessing/README.md).

---

## 4. Live demo

### Requirements
* Python 3.13 with a CUDA build of PyTorch (tested on an RTX 4050 Laptop GPU, 6 GB); falls back to CPU.
* `torch`, `transformers==4.57.1`, `sentence-transformers`, `faiss-cpu`, `scikit-learn`, `pandas`, `joblib`, `pillow`, `matplotlib`, `gradio` (6.x), `emoji`, `huggingface-hub>=0.34,<1.0` (pin it; gradio otherwise pulls a newer hub that breaks transformers). `nbconvert`, `nbformat` and `ipykernel` are needed for the notebook. Pinned versions for the exploration scripts are in `env/requirements_day4.txt`.

### Start
`artifacts/` must exist (section 5). From the repository root, using the interpreter that has the packages installed (deactivate any other virtual environment first):

```powershell
$env:HF_HUB_OFFLINE = "1"
python demo/app.py
```

Open **http://127.0.0.1:7860**. Everything loads from `artifacts/`; no internet is needed.

### Using it
1. Upload an image, paste the tweet text and press **Analyse**, or click one of the six preloaded test-split examples. Ten more upload-ready cases are in `demo_examples/` (see its `MANIFEST.md`).
2. Outputs:
   * **Informative?** A badge from the text-only Task 1 model with a calibrated confidence. It never gates Tasks 2 and 3.
   * **Humanitarian category:** probabilities over 8 classes, including "Not humanitarian".
   * **Damage severity:** the predicted level and the 5-NN vote share. The vote share is agreement among neighbours, not a calibrated probability.
   * **Top-5 similar past cases:** image, tweet and label from the train-only index.
3. If Task 1 and Task 2 disagree, both are shown with a note.

### Configuration

| Environment variable | Effect |
|---|---|
| `CAPSTONE_T1=tfidf` | Serve the TF-IDF + LR Task 1 model instead of BERTweet (fastest start on CPU) |
| `CAPSTONE_T3_HYBRID=0` | Use v2's plain 5-NN vote for Task 3 instead of the hybrid |
| `CAPSTONE_T2_CKPT=artifacts/task2_model.pt` | Serve v2's own Task 2 checkpoint instead of L4a |

### Performance (RTX 4050 laptop GPU, medians)
Task 1 model call 0.28 ms (TF-IDF) or 8.4 ms (BERTweet); whole cascade in-process 25.6 / 35.1 ms; full HTTP round trip through Gradio including upload and the five retrieved images about 1.75 / 1.78 s. Source: `results/ablation/D4_t1_latency.json`.

---

## 5. Reproducing the work

Run everything from the repository root.

**Data (not in git).** Place CrisisMMD v2.0 at:
```
data/CrisisMMD_v2.0/
    annotations/   data_image/   json/
    crisismmd_datasplit_all/crisismmd_datasplit_all/   (task_*_text_img_{train,dev,test}.tsv)
```

**`artifacts/` (git-ignored).** Models, tokenizers, the FAISS index and cached embeddings. Recreate it with the full retrain below, or unzip a backup copy into the repository root. The BERTweet checkpoint (`artifacts/task1_bertweet/`, about 517 MB) is a seed-42 retrain and must be transferred separately.

**Fast check (about 10-15 s):** reruns the notebook from saved models and reproduces the reported test metrics.
```
python -m jupyter nbconvert --to notebook --execute --inplace capstone-v2.ipynb
```

**Full retrain (about 12 minutes on the RTX 4050 with cached embeddings):** recomputes embeddings, retrains Task 2 (at most 10 epochs, AMP, best epoch by dev macro-F1), rebuilds the index and re-exports `artifacts/`. AMP/cuDNN is not bit-reproducible, so a retrain gives slightly different numbers than the saved artifacts.
```
set CAPSTONE_TRAIN=1
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 capstone-v2.ipynb
```

**Regenerate notebook, figures and demo examples:**
```
python tools/build_notebook.py     # capstone-v2.ipynb
python tools/make_figures.py       # results/figures/*.png
python tools/select_examples.py    # demo/examples.json (asserts no tweet/text/image twin in the index)
python tools/check_examples.py     # runs the preloaded examples through the pipeline
python tools/build_demo_folder.py  # demo_examples/, then tools/make_contact_sheet.py
```

**Zero-shot experiment** (needs `artifacts/emb/`; set `PYTHONPATH=.`):
```
python tools/s1_write_prompts.py
python s1_zeroshot.py
python tools/s1_extras.py
python tools/s1_sanity.py
```

**Model exploration (Task 1 and Task 2).** Run from `ablation/`, each script reads or writes `results/ablation/`:
```
cd ablation
python d4_t1_classical.py; python d4_t1_step2.py            # Task 1 classical models
python d4_t1_bertweet.py                                    # Task 1 BERTweet, seeds 42-44 (`42 --save` for the deployed checkpoint)
python d4_t1_bilstm.py                                      # BiLSTM + GloVe (needs artifacts/glove/)
python d4_t1_extract_images.py resnet50 dinov2 clip_l14; python d4_t1_heads.py resnet50 dinov2 clip_l14
python d4_t1_fusion.py T1 bertweet img_clip_l14; python d4_t1_early_fusion.py resnet50
python d4_t1_switch.py                                      # pre-registered switch rule
python d4_t2_classical.py; python d4_t2_bertweet_text.py matched
python d4_t2_early_fusion.py; python d4_t2_summary.py
python d4_t1_lodo.py                                        # Task 1 leave-one-disaster-out (about 35 min)
python d4_error_analysis.py
python d5_prep_c.py; python d5_t1_bertweet_prep.py --c 42 43 44; python d5_summary.py   # preprocessing A/B
python verify_findings.py --write-log                       # re-derives every reported number
```
Run Task 1 scripts and Task 2 scripts in separate processes; `d4_t1_data.py` and `d4_t2_data.py` assert this. The complete command list is in `docs/ablation_findings.md`.

---

## 6. Repository layout

```
capstone-final.ipynb     original Kaggle notebook (unmodified)
capstone-v2.ipynb        corrected pipeline (TRAIN flag: False loads artifacts/, True retrains)
crisis_common.py         shared loaders, embedders, models and metrics
s1_zeroshot.py           CLIP zero-shot damage-severity experiment
demo/                    Gradio app, pipeline, hybrid Task 3, BERTweet Task 1, calibration, examples
demo_examples/           ten upload-ready cases with MANIFEST.md (images git-ignored)
ablation/                ablation, exploration, bootstrap, verification and report-building scripts
preprocessing/           cleaning and split scripts plus preprocessed datasets for Tasks 1-3
tools/                   notebook builder, figure generator, example selection, zero-shot helpers
results/                 metrics, before/after table, predictions, figures, results/ablation/* run JSONs
docs/                    ablation findings, v1 configuration, error analysis, report dossier
report_v2/               report draft (LaTeX) and figures
env/                     pinned requirements
artifacts/               trained models, FAISS index, cached embeddings (git-ignored)
RESULTS.md               bug audit, before/after table, architecture, limitations
ABLATION_PLAN.md         ablation protocol
```

---

## 7. Limitations and future work

* Task 3 retrieves labels of past cases. It is biased towards "Severe damage" (Mild-damage recall 18% for the plain 5-NN), and its vote share is not a probability.
* Task 2's cross-attention attends over a single pooled image vector, so it is effectively a projection.
* Small classes make Task 2 macro-F1 noisy; the test-sampling 95% CI of a macro-F1 is about ±0.04, larger than most gaps between architectures. "No benefit detected" means "not resolvable at this size", not "absent".
* Performance falls on unseen disaster events (leave-one-disaster-out), so the in-distribution numbers overstate generalisation.
* Several comparisons are exploratory and uncorrected for multiple testing.
* Future work: Task 2 leave-one-event-out with the fusion model, patch-token cross-attention, a zero-shot vision-language model, and satellite imagery (xBD).

---

## 8. Data and references

* Alam, Ofli, Imran (2018). *CrisisMMD: Multimodal Twitter Datasets from Natural Disasters.*
* Ofli, Alam, Imran (2020). *Analysis of Social Media Data using Multimodal Deep Learning for Disaster Response.* arXiv:2004.11838.
* Models: OpenAI CLIP ViT-B/32 and ViT-L/14, `distilbert-base-uncased`, `vinai/bertweet-base`, `sentence-transformers/all-MiniLM-L6-v2`.
