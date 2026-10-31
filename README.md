# CrisisMMD disaster-triage pipeline

An audited and corrected multimodal pipeline for disaster tweets (CrisisMMD v2.0), wrapped into a live triage demo.
Give it a tweet's image and text; it answers three questions and shows the evidence.

| Stage | Question | Model | Input |
|---|---|---|---|
| Task 1 | Is the tweet informative? | TF-IDF + Logistic Regression | text only |
| Task 2 | Which humanitarian category? (8 classes) | frozen CLIP ViT-B/32 image vector + fine-tuned DistilBERT, cross-attention, MLP | image + text |
| Task 3 | How severe is the damage? (3 classes) | CLIP image + SBERT text vector, FAISS index, 5-NN vote | image + text |

The demo also shows the five most similar past cases (with their images, tweets and labels) from the damage-train index.

This project started from `capstone-final.ipynb` (a Kaggle notebook, kept unmodified). We audited it, fixed a set of verified bugs
and rebuilt it as `capstone-v2.ipynb`. The full audit, tables, figures and limitations are in **[RESULTS.md](RESULTS.md)**.

## Results (test split, seed 42)

| Task | Accuracy | Macro-F1 | Weighted-F1 | n |
|---|---|---|---|---|
| T1 Informative (text-only) | 0.7944 | 0.7513 | 0.7970 | 2237 |
| T2 Humanitarian (8 classes) | 0.7975 | 0.5865 | 0.7968 | 2237 |
| T3 Damage severity (3 classes, tweet-disjoint index) | 0.6673 | 0.4914 | 0.6202 | 529 |

> **Task 1 has since changed.** The table above is the corrected v2 baseline (TF-IDF + logistic regression). The demo now serves a fine-tuned BERTweet for Task 1 (test weighted F1 0.8382 ± 0.0022 and macro-F1 0.7970 ± 0.0009 as 3-seed means; deployed checkpoint 0.8443 / 0.8024), chosen by a pre-registered dev rule; see "Task 1 backend" below and `docs/report_dossier.md`.

> **What does the demo serve?** The live demo serves the ablation study's L4a seed-42 Task 2 model (test accuracy 0.8033, macro-F1 0.6080) and a hybrid Task 3 (a logistic probe predicts the damage level, the five most similar past cases are shown as evidence, and a disagreement is flagged for human review), with temperature-scaled confidences (`demo/calibration.json`). The numbers listed above are for the v2 pipeline. `CAPSTONE_T3_HYBRID=0` restores v2's 5-NN vote and `CAPSTONE_T2_CKPT=artifacts/task2_model.pt` selects v2's own Task 2 checkpoint (see `docs/ablation_findings.md`).

Task 3 is lower than the old notebook's number on purpose: the old evaluation was inflated by test-set leakage and polluted by a
class that does not exist in the test split (details in RESULTS.md). The old and corrected rows are not on identical test sets
(the old path bug removed two events), and the comparison with the CrisisMMD authors' numbers is indicative only. Both caveats are
explained in RESULTS.md. The old-vs-corrected table is `results/before_after.csv`.

### Extra experiment: can damage severity be classified with no training? (S1)

We tested CLIP zero-shot (image only, text prompts as class descriptions) on the same 529 damage-test rows. Prompt wording was fixed and
committed before any result existed, the best of three prompt sets was chosen on **dev** only, and test was scored once.

| Method | Accuracy | Macro-F1 | Weighted-F1 | Recall Severe / Mild / Little |
|---|---|---|---|---|
| Task 3 k-NN, image + text (headline) | 0.667 | 0.491 | 0.620 | 92% / 18% / 32% |
| Task 3 k-NN, CLIP image only | 0.675 | 0.507 | 0.641 | 91% / 27% / 28% |
| CLIP zero-shot (prompt set E3) | 0.303 | 0.299 | 0.349 | 26% / 37% / 38% |
| Zero-shot + bias terms fitted on dev (not pure zero-shot) | 0.567 | 0.397 | 0.563 | 74% / 37% / 8% |
| Baseline: always "Severe damage" | 0.628 | 0.257 | 0.484 | 100% / 0% / 0% |
| Baseline: stratified random (expected) | 0.469 | 0.333 | 0.469 | 63% / 24% / 13% |

Reading it: zero-shot is **below chance**, so labelled data is what makes Task 3 work. Adding the tweet text to the k-NN makes no
meaningful difference (0.491 vs 0.507 macro-F1, within noise on 529 rows with a 13% minority class). "Always Severe" already scores 0.628
accuracy, which is why macro-F1 is the headline. Feature extraction was checked (text and image features are the projected 512-d
vectors; a cat-vs-disaster test picks "disaster" 20/20; matched cosine similarities are 0.22-0.29): see `results/s1_sanity.txt`.
Files: `results/s1_zeroshot.csv`, `results/s1_dev_candidates.csv`, `results/s1_prompts.json`, `results/figures/confusion_s1_zeroshot.png`.

## Live demo

### Requirements
* Windows, Python 3.13 with a CUDA build of PyTorch (tested on an RTX 4050 Laptop GPU, 6 GB). The demo falls back to CPU if no GPU is found.
* Packages: `torch`, `transformers==4.57.1`, `sentence-transformers`, `faiss-cpu`, `scikit-learn`, `pandas`, `joblib`, `pillow`,
  `matplotlib`, `gradio` (6.x), `huggingface-hub<1.0` (gradio pulls in a newer hub that breaks transformers 4.57; pin it as shown),
  `nbconvert`/`nbformat`/`ipykernel` for the notebook.

```
pip install "huggingface-hub>=0.34,<1.0"
```

### Start it
`artifacts/` must exist (see "Reproduce"). From the repository root, in **PowerShell**, use the Python 3.13 interpreter that has the
packages installed. If another virtual environment (such as a `venv`) is active it will not have them: run `deactivate` first, or call the
interpreter by its full path as below.

```
$env:HF_HUB_OFFLINE = "1"
& "C:\Users\anura\AppData\Local\Programs\Python\Python313\python.exe" demo/app.py
```

In Command Prompt, use `set HF_HUB_OFFLINE=1` for the first line instead.

Then open **http://127.0.0.1:7860**. Everything loads from `artifacts/`, so no internet connection is needed. Loading takes a few
seconds. Measured on the RTX 4050 laptop GPU (`results/ablation/D4_t1_latency.json`, medians): the Task 1 model call takes 0.28 ms (TF-IDF) or 8.4 ms (BERTweet), the whole cascade in-process 25.6 / 35.1 ms (without upload or file transfer), and a full HTTP round trip through the running Gradio server, including the image upload and the five retrieved images, about 1.75 / 1.78 s. (The earlier "0.1-0.2 s model call" has no recorded benchmark; the "1.8 s" figure matches the HTTP round trip.)

### Task 1 backend (BERTweet by default, TF-IDF as a fallback)
Task 1 uses a fine-tuned BERTweet (`vinai/bertweet-base`), chosen over TF-IDF + logistic regression by a rule fixed in advance (`ablation/SELECTION_RULE_DAY4.md`, results in `results/ablation/D4_switch_rule_T1.json`).
* **Checkpoint:** `artifacts/task1_bertweet/` (about 517 MB: `model.safetensors`, tokenizer files, `t1_meta.json` with the labels, maximum length and dev temperature). It is **git-ignored and must be handed over separately** (zip the folder with the rest of `artifacts/`). The demo loads it from disk only; the `emoji` package must be installed for BERTweet's tokenizer normalisation.
* **It is a retrain of seed 42** (the original runs saved no weights): dev weighted F1 0.8289 against 0.8321 originally, 94.3% test-prediction agreement. See the "Deviations" section of `ablation/SELECTION_RULE_DAY4.md`.
* **Fallback:** `CAPSTONE_T1=tfidf` (environment variable) or `Pipeline(t1_backend="tfidf")` serves the original TF-IDF + logistic regression from `artifacts/task1_pipeline.joblib`, which stays in `artifacts/` (git-ignored). On CPU only, set `CAPSTONE_T1=tfidf` for the fastest start.
* **Confidence:** the displayed probability is temperature-scaled (T = 1.825, fitted on the dev logits of the deployed checkpoint, stored in `demo/calibration.json`).

### Use it
1. Upload an image and paste the tweet text, then press **Analyse**. Or click one of the six **Preloaded examples** (test-split tweets).
2. Read the four outputs:
   * **Informative?** A badge from the text-only Task 1 model (a fine-tuned BERTweet with a temperature-calibrated confidence; see "Task 1 backend"). It is a badge, never a gate: Tasks 2 and 3 always run.
   * **Humanitarian category:** probability bars over 8 classes. This includes its own "Not humanitarian" class.
   * **Damage severity:** the predicted label and a bar titled *agreement among similar past cases (k=5 vote share)*. This is a vote
     share among retrieved neighbours, **not a calibrated probability**.
   * **Top-5 similar past cases:** images, tweets and labels from the train-only index.
3. If Task 1 and Task 2 disagree, both are shown with a note. Task 1 is trained on tweet-text labels and Task 2 on image labels, so a
   tweet can be informative as text while its image is not humanitarian, or vice versa.

### Extra demo examples

`demo_examples/` holds ten upload-ready cases (image + tweet text) with a `MANIFEST.md` of true labels versus model output and why each is worth showing: 7 good picks, 1 case where Task 1 and Task 2 disagree, and 2 deliberate failures. They come from the test split with no twin in the damage-train index and are different from the six built-in examples. Rebuild with `python tools/build_demo_folder.py` (images are git-ignored because they come from the dataset).

### What to expect (be honest on stage)
* The preloaded examples are test-split tweets with no twin in the damage-train index, chosen by their true labels (not by whether
  the model gets them right). Four of the six examples have a wrong prediction in at least one task: Task 3 on the Iraq-Iran
  "I am alive" tweet and on the Irma "Restoration of Water Supply" tweet, and Task 2 on the Harvey "A picture worth a thousand words"
  tweet and on the Irma "Cuba announces program" tweet.
* Task 3 is biased towards "Severe damage" (Mild-damage recall is 18%); see the limitations in RESULTS.md.
* A tweet shown in the neighbour panel is never the query tweet for the preloaded examples (this is asserted in `tools/select_examples.py`).

## Presentation guide (suggested flow, about 8 minutes)

1. **Problem (30 s).** Disaster tweets arrive with an image and text. Triage needs: is it informative, which humanitarian need, how bad is the damage.
2. **Architecture (1 min).** Show `results/figures/architecture.png`: three independent stages over one tweet; Task 1 is a badge, never a gate.
3. **The audit (2 min).** This is the main story. The original notebook scored well for the wrong reasons:
   * a path bug silently dropped two whole events (about 11% of every split);
   * the Task 3 index contained the test images, and 80% of it was a class ("Don't know") that does not exist in the test split;
   * a second, subtler leak: tweets with several images appear in both train and test. We exclude them and document it.
   After fixing, Task 3 *drops* (0.678 to 0.667 accuracy, macro-F1 0.491). We report it and did not tune it back up.
4. **Live demo (3 min).** Start with preloaded examples, then upload files from `demo_examples/` (see `MANIFEST.md`):
   case 01 (clean success), case 08 (Task 1 and Task 2 disagree, note appears), case 10 (vote share 1.0 but wrong: vote share is not accuracy), case 09 (confident Task 2 error).
5. **Honest limits (1 min).** Mild-damage recall is 18%; Task 2 macro-F1 (0.587) is much lower than accuracy because of tiny classes; S1 shows zero-shot does not work here.
6. **Q&A cheat sheet.**
   * *Why is Task 3 lower than before?* The old number used leaked/polluted data; see RESULTS.md section 2.
   * *Why macro-F1?* Classes are imbalanced; "always Severe" gets 0.628 accuracy but 0.257 macro-F1.
   * *Why not tune Task 3?* Test is never used for selection; dev tuning is listed as future work.
   * *Is the comparison with the original paper fair?* Indicative only (different splits and label handling); see RESULTS.md section 3b.

## Reproduce

Everything runs from the repository root with `python` = the Python 3.13 interpreter.

**Data** (not in git): place CrisisMMD v2.0 at
```
data/CrisisMMD_v2.0/
    annotations/   data_image/   json/
    crisismmd_datasplit_all/crisismmd_datasplit_all/   (task_*_text_img_{train,dev,test}.tsv)
```

**Fast check (about 10-15 s):** re-runs the whole notebook from the saved models and reproduces the reported test metrics.
```
python -m jupyter nbconvert --to notebook --execute --inplace capstone-v2.ipynb
```

**Full retrain (about 12 minutes on the RTX 4050 with cached embeddings, about 4 minutes more on a cold start):** recomputes the embeddings, retrains Task 2 (at most 10 epochs, AMP, best epoch
by dev macro-F1), rebuilds the index and re-exports `artifacts/`.
```
set CAPSTONE_TRAIN=1
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 capstone-v2.ipynb
```
Training uses seed 42, but AMP/cuDNN is not bit-reproducible, so a retrain gives slightly different numbers than the ones reported.
The saved artifacts do reproduce the reported test metrics exactly.

**Regenerate** the notebook, the figures, and the demo examples:
```
python tools/build_notebook.py     # writes capstone-v2.ipynb
python tools/make_figures.py       # results/figures/*.png (reads saved predictions, no retraining)
python tools/select_examples.py    # demo/examples.json (asserts no tweet/text/image twin in the damage-train split)
python tools/check_examples.py     # runs the preloaded examples through the pipeline and lists the wrong ones
python tools/build_demo_folder.py  # demo_examples/ (10 upload-ready cases) then python tools/make_contact_sheet.py
```

**S1 zero-shot experiment** (needs `artifacts/emb/` from a training run; set `PYTHONPATH=.`):
```
python tools/s1_write_prompts.py   # writes results/s1_prompts.json (fixed before any result)
python s1_zeroshot.py              # dev selection, one test evaluation, results/s1_zeroshot.csv
python tools/s1_extras.py          # trivial baselines and image-only k-NN rows (appended to the same CSV)
python tools/s1_sanity.py          # feature sanity checks -> results/s1_sanity.txt
```

**Day 4 tables (model exploration; one command per table).** Use the Python 3.13 interpreter with CUDA PyTorch, `transformers`, `emoji`; run from `ablation/`. Every command reads or writes `results/ablation/`; the rules were committed in `ablation/SELECTION_RULE_DAY4.md` before the runs.
```
cd ablation
python d4_t1_classical.py                 # Task 1 TF-IDF + LR / NB / RF / SVM            -> D4/T1_*  (then d4_t1_step2.py: fair NB, majority, weighting, GloVe check, D4_T1_summary.{json,md})
python d4_t1_bertweet.py                  # Task 1 BERTweet, seeds 42-44 (add --noweight, or `42 --save` for the deployed checkpoint)
python d4_t1_bilstm.py                    # Task 1 BiLSTM + GloVe Twitter 200d (needs artifacts/glove/glove.twitter.27B.zip)
python d4_t1_extract_images.py resnet50 dinov2 clip_l14; python d4_t1_heads.py resnet50 dinov2 clip_l14    # Task 1 image heads
python d4_t1_fusion.py T1 bertweet img_clip_l14; python d4_t1_early_fusion.py resnet50  # late and early fusion (also: clip_l14)
python d4_t1_switch.py                    # the pre-registered switch rule             -> D4_switch_rule_T1.json
python d4_t1_fusion_gain.py               # Task 1 fusion gain, bootstrap, agreed/disagreed -> D4_T1_fusion.{json,md}
python d4_t2_classical.py; python d4_t2_bertweet_text.py matched; python d4_t2_bertweet_text.py rule8    # Task 2 text models
python d4_t2_reinfer.py                   # probabilities of the existing L4a / DistilBERT / image-only checkpoints (reproduces the stored results)
python d4_t2_early_fusion.py              # the guide's core model (add --clip_b32 for the post-hoc backbone swap)
python d4_t2_summary.py; python d4_t2_fusion_analysis.py   # Task 2 tables, comparisons, tie-break, subsets -> D4_T2_text_summary.*, D4_T2_fusion.*
python d4_fusion_both.py                  # fusion gain, both tasks, both baselines    -> D4_fusion_gain_both.*
python d4_t1_dups.py                      # Task 1 text and image duplicates           -> D4_T1_dups.*
python d4_t1_lodo.py                      # Task 1 leave-one-disaster-out with BERTweet (about 35 min)  -> D4_T1_lodo.*
python d4_t1_calib_ofli.py                # Task 1 calibration and the indicative Ofli row
python d4_error_analysis.py               # error analysis, contact sheets            -> docs/day4_error_analysis.*, report_v2/figures/contact_*
python d4_t1_latency.py; python d4_t1_http_latency.py   # latency (in-process, then through the Gradio server)
python d4_make_findings.py                # section 9.12 of docs/ablation_findings.md + docs/day4_justification_table.tex
python d4_make_figures.py; python d4_make_pipeline_fig.py                  # report_v2/figures
python d4_build_report_v2.py; python d4_check_tex.py; python d4_make_dossier.py   # report_v2/main.tex (+ static checks), docs/report_dossier.md
python verify_findings.py --write-log     # re-derives every number (docs section 9.12, report_v2 table cells, dossier tokens)
```
Training logs are archived in `results/ablation/d4_logs.tar.gz`. Run the new Task 1 scripts only in a process that has not imported a Task 2 module (and vice versa): `d4_t1_data.py` / `d4_t2_data.py` assert this.

**`artifacts/` is git-ignored** (the Task 2 checkpoint exceeds GitHub's 100 MB limit). Recreate it with the full retrain above, or
unzip a backup copy into the repository root.

## Repository layout

```
capstone-final.ipynb       original Kaggle notebook (unmodified, for reference)
capstone-v2.ipynb          corrected pipeline (TRAIN flag: False loads artifacts/, True retrains)
crisis_common.py           shared loaders, embedders, models and metrics (used by the notebook and the demo)
demo/pipeline.py           loads artifacts once; predict(image, text) -> dict
demo/app.py                Gradio UI
demo/examples.json         the six preloaded test examples
demo_examples/             ten upload-ready demo cases (text + MANIFEST.md; images are git-ignored, rebuild with the tool)
s1_zeroshot.py             S1: CLIP zero-shot damage severity (dev-selected prompts, test scored once)
tools/                     notebook builder, figure generator, example selection/check scripts, S1 helpers
results/                   before_after.csv, metrics.json, split_overlap.csv, test predictions, s1_* files, figures/
artifacts/                 models, tokenizer, FAISS index, metadata, cached embeddings (git-ignored)
RESULTS.md                 bug audit, before/after table, architecture, limitations, future work
```

## The audit in brief

Details, evidence and the reasoning behind each fix are in RESULTS.md.

1. **Path bug:** `lstrip('data_image/')` strips a character set and silently dropped the Iraq-Iran and Mexico earthquake events
   (about 11% of every split). Fixed with `removeprefix`; every image now resolves and this is asserted.
2. **Task 3 leakage:** the index was built from all annotations, including the test images. It is now built from the damage-train split
   only. A second, subtler overlap (tweets with several images split across train and test) is also excluded.
3. **Task 1 report labels** were in the wrong order; **undefined `T3_CLIP` objects**, a **`TRAIN` flag**, a **QuickGELU mismatch** in the
   old open_clip image features, and a non-deterministic **vote tie-break** were also fixed.

## Limitations and future work

* Task 3 retrieves labels of past cases; it is biased towards "Severe damage" and its vote share is not a probability.
* Task 2's cross-attention attends over a single pooled image vector, so it is effectively a projection.
* Small classes (for example "Missing or found people", 3 test samples) make Task 2 macro-F1 noisy.
* Future work: distance-weighted or class-balanced voting for Task 3 (tuned on dev, after excluding the 16 dev/test shared tweets),
  leave-one-event-out evaluation, a zero-shot VLM (LLaVA), patch-token cross-attention for Task 2, and satellite imagery (xBD).

## Data and references

* CrisisMMD v2.0: Alam, Ofli, Imran (2018), *CrisisMMD: Multimodal Twitter Datasets from Natural Disasters*.
* Ofli, Alam, Imran (2020), *Analysis of Social Media Data using Multimodal Deep Learning for Disaster Response*, arXiv:2004.11838.
* Models: OpenAI CLIP ViT-B/32 (`openai/clip-vit-base-patch32`), `distilbert-base-uncased`, `sentence-transformers/all-MiniLM-L6-v2`.
