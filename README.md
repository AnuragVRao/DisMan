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

Task 3 is lower than the old notebook's number on purpose: the old evaluation was inflated by test-set leakage and polluted by a
class that does not exist in the test split (details in RESULTS.md). The old and corrected rows are not on identical test sets
(the old path bug removed two events), and the comparison with the CrisisMMD authors' numbers is indicative only. Both caveats are
explained in RESULTS.md. The old-vs-corrected table is `results/before_after.csv`.

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
`artifacts/` must exist (see "Reproduce"); from the repository root:

```
set HF_HUB_OFFLINE=1
python demo/app.py
```

Then open **http://127.0.0.1:7860**. Everything loads from `artifacts/`, so no internet connection is needed. Loading takes a few
seconds and a prediction takes about 0.1-0.2 s on the GPU (about 1.8 s end to end through the web request).

### Use it
1. Upload an image and paste the tweet text, then press **Analyse**. Or click one of the six **Preloaded examples** (test-split tweets).
2. Read the four outputs:
   * **Informative?** A badge from the text-only Task 1 model. It is a badge, never a gate: Tasks 2 and 3 always run.
   * **Humanitarian category:** probability bars over 8 classes. This includes its own "Not humanitarian" class.
   * **Damage severity:** the predicted label and a bar titled *agreement among similar past cases (k=5 vote share)*. This is a vote
     share among retrieved neighbours, **not a calibrated probability**.
   * **Top-5 similar past cases:** images, tweets and labels from the train-only index.
3. If Task 1 and Task 2 disagree, both are shown with a note. Task 1 is trained on tweet-text labels and Task 2 on image labels, so a
   tweet can be informative as text while its image is not humanitarian, or vice versa.

### What to expect (be honest on stage)
* The preloaded examples are test-split tweets with no twin in the damage-train index, chosen by their true labels (not by whether
  the model gets them right). Four of the six examples have a wrong prediction in at least one task: Task 3 on the Iraq-Iran
  "I am alive" tweet and on the Irma "Restoration of Water Supply" tweet, and Task 2 on the Harvey "A picture worth a thousand words"
  tweet and on the Irma "Cuba announces program" tweet.
* Task 3 is biased towards "Severe damage" (Mild-damage recall is 18%); see the limitations in RESULTS.md.
* A tweet shown in the neighbour panel is never the query tweet for the preloaded examples (this is asserted in `tools/select_examples.py`).

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
```

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
tools/                     notebook builder, figure generator, example selection and check scripts
results/                   before_after.csv, metrics.json, split_overlap.csv, test predictions, figures/
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
