# Day 4 pre-registration: model exploration requested by the guide

Written and committed before any Day 4 run. Scope: the Task 1 and Task 2 candidate lists from the guide's model sheets, plus the metric set from the guide's metric sheet.

## Candidates
Task 1 (informative vs not informative):
- Text: TF-IDF+LR (existing), TF-IDF+Naive Bayes, TF-IDF+Random Forest, TF-IDF+Linear SVM, BiLSTM+GloVe Twitter 200d (fallback GloVe 6B 100d), BERTweet fine-tuned.
- Image (cached frozen features + head): ResNet50, DINOv2, CLIP image-only.
- Multimodal: calibrated late fusion (best text + best image), BERTweet (fine-tuned) + ResNet50 (frozen) concat + MLP, CLIP-feature fusion with a 2-class head.
Task 2 (8-class humanitarian):
- Text: TF-IDF+LR (class-weighted), Linear SVM, Naive Bayes, BERTweet with weighted CE.
- Multimodal: calibrated late fusion, BERTweet (fine-tuned) + ResNet50 (frozen) early fusion with weighted CE; BERTweet inside L4a cross-attention if time.
- Image heads and gated/cross-attention fusion: reused from D2 (§9.10), not rerun.
- Stretch only: ResNet50 fine-tuned, DeBERTa-v3 (bf16), ModernBERT, zero-shot VLM.

## Rules
1. **Task 1 label.** All Task 1 models (text, image, fusion) are trained and scored on the text informativeness label (`label_text`). The loader asserts this column.
2. **Splits and selection.** Same corrected splits, seeds 42/43/44. Select on dev (T1 headline weighted F1, T2 macro-F1). Test is scored once. Ties go to the simplest model.
3. **Deployed Task 1 model switch is decided on dev.** Switch from TF-IDF+LR to BERTweet only if the paired dev bootstrap CI (seed-averaged) of the gain excludes 0 AND the gain is positive at every seed. TF-IDF is deterministic, so each BERTweet seed is compared against it. Otherwise keep TF-IDF and justify on latency. Test is reported afterwards and not used to decide.
4. **Probabilities.** Every run saves dev and test probabilities. For the SVM, save `decision_function` scores; probability metrics use `CalibratedClassifierCV` fitted with CV on train (dev stays clean).
5. **Labelling.** The guide's model is reported as "BERTweet (fine-tuned) + ResNet50 (frozen)". Reason: frozen vs fine-tuned CLIP showed no difference (L3b); breadth was prioritised.
6. **Multiplicity.** Roughly 15-20 additional comparisons, no correction. Language stays "established / not distinguishable" (§9.10). Bootstrap CIs only for: best model vs the guide's core model, and fusion vs best unimodal.
7. **Metrics.** Task 1: accuracy, weighted F1 (headline), macro-F1, per-class P/R, MCC, ROC-AUC, confusion matrix. Task 2: macro-F1 (headline), weighted F1, accuracy, per-class P/R/F1, balanced accuracy, MCC, row-normalised 8x8 confusion matrix, top-2 accuracy. Top-2 is computed for more than 2 classes only. Both tasks: mean ± std over 3 seeds for trained neural models, fusion gain (multimodal minus best single modality, macro-F1 and headline), 10-15 error-analysis examples.
8. **BERTweet setup.** `use_fast=False`, `normalization=True`, max length 128, lr 2e-5, 4 epochs, select on dev. A sanity check must show `HTTPURL` and `@USER` in a tokenised example.
9. **Honesty.** Results from candidates added after seeing earlier Day 4 results are labelled post-hoc.
10. **Post-hoc arm (added after the first Day 4 results were seen).** BERTweet with plain unweighted cross-entropy, 3 seeds, everything else identical, to answer "why a weighted loss for Task 1 (71% informative / 29% not)?". Labelled post-hoc; it does not change the BERTweet headline arm (weighted), which was fixed before running.
11. **Task 2 provenance (pre-registered before any Task 2 Day 4 run).** Every Task 2 model initialises its text encoder from the hub (`vinai/bertweet-base`, `distilbert-base-uncased`, ...), never from a Task 1 checkpoint, head, logit file or fine-tuned weight, because T1 train contains 1,587 T2 test rows (D4_cross_task_overlap.json). Every Task 2 run JSON carries `task_name == "humanitarian"`, `label_col == "label_image"`, the T2 `train_split_sha256` and an `init_from` that is a hub id; `d4_prov.assert_init` (hub id only) is called by every Task 2 script before it builds a model, and `d4_prov.assert_run(run, "humanitarian")` by every script that consumes saved Task 2 runs. The mirror-image rule for Task 1 (rule 1 plus `assert_run(run, "informative")`) is enforced in `d4_t1_fusion.py`, `d4_t1_switch.py` and the loaders; the module-name guard in `d4_t1_data.data()` is kept as a secondary check.
12. **Provenance record.** Run JSONs carry `task` ("T1"/"T2", used in filenames) and `task_name` ("informative"/"humanitarian"). Task 1 runs made before `d4_prov.py` existed were back-filled with metadata only, with a parsed-JSON identity check (no pre-existing value changed), and every Task 1 run's stored `y_dev`/`y_test` was verified equal to the loader's labels (`D4_T1_audit.json`, 25/25).

## Deviations
- **Deployed Task 1 checkpoint is a retrain.** Original seed runs saved no weights; seed 42 was retrained with --save (dev wF1 0.8289 vs 0.8321 originally; 94.3% test-prediction agreement). Same-seed run-to-run variation (0.0032) exceeds the inter-seed dev spread (0.0030: 0.8321, 0.8292, 0.8300), so seed selection is within noise. Architecture claims use the 3-seed means; deployed-model numbers come from the bertweet_deploy run JSON (results/ablation/D4/T1_bertweet_deploy_seed42.json). No further seeds were retrained.
