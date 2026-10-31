# Day 4: what the Task 1 switch and the new results change in the report (corrected: written against `report/Capstone_report_Sem5.tex`, the report whose sections are numbered III-C ... VII-H)

**Correction.** The first version of this file was written against `report/main.tex`, an older and shorter draft in the same folder; its line numbers and section letters were wrong for the report you work from. This version maps every change to `report/Capstone_report_Sem5.tex` (653 lines; the file the dossier calls "the old report"). `report_v2/main.tex` is generated from it by `ablation/d4_build_report_v2.py`, which applies each change below with an exact-match assertion, so a change that no longer matches the source stops the build. Nothing in `report/` is modified.

Which Task 1 number to quote: the **deployed** checkpoint is a retrain of seed 42 (`bertweet_deploy`): test weighted F1 0.8443, macro-F1 0.8024, accuracy 0.8480. The **architecture** claim uses the 3-seed mean: weighted F1 0.8382 ± 0.0022, macro-F1 0.7970 ± 0.0009. TF-IDF + LR (v2): 0.7970 and 0.7513. The abstract uses the 3-seed means; the body gives both.

| Where in `Capstone_report_Sem5.tex` (lines) | Section | Change |
|---|---|---|
| 46-48 | Abstract | Task 1 is BERTweet: weighted F1 (the guide's headline) and macro-F1, 3-seed means. One sentence on the guide-requested model comparison, one on the unified agreed/disagreed finding. |
| 63-69 | Introduction, contributions | Add the model-exploration contribution; the third contribution already mentions the agreed/disagreed finding for Task 2, extend it to both tasks. |
| 78 | II-B | Ofli et al. comparison: point to the new indicative BERTweet row. |
| 80-81 | II-C | Add BERTweet (and RoBERTa) with citations; add the classical and recurrent baselines (NB, Random Forest, SVM, BiLSTM + GloVe). |
| 124-125 | III-C Tweet-level integrity | Add: tweet-disjoint within each of Tasks 1 and 2 (0 shared tweets) but NOT across tasks (1,603 of the 2,237 Task 1 test rows are in the Task 2 train split); handled by provenance asserts and no cross-task weights. The sentence "The informativeness and humanitarian splits are tweet-disjoint" must say "within each task". |
| 127-128 | III-D Duplicate images | Add the Task 1 results: text duplicates (1 exact, 2 near) and image duplicates (57 byte-identical, 142 flagged); re-scored models change by at most 0.005. |
| 133-162 | Fig. 1 (TikZ) | Task 1 box: BERTweet (fine-tuned) + temperature; caption sentence. Redrawn PNG: `report_v2/figures/fig_c6_pipeline.png`. |
| 166-167 | IV-A Task 1 | BERTweet (`vinai/bertweet-base`, normalisation on, max length 128, lr 2e-5, 4 epochs, weighted CE, epoch by dev weighted F1), temperature T = 1.825; TF-IDF + LR kept as the CPU fallback (`CAPSTONE_T1=tfidf`). |
| 181-182 | IV-D Calibration | Add the Task 1 temperature; the Task 1 badge is no longer a raw probability. |
| 184-185 | IV-E Interactive tool | "the raw logistic-regression probability for the Task 1 badge" becomes the calibrated BERTweet probability. Latency "A model call takes 0.1-0.2 s and an end-to-end request about 1.8 s" (no recorded source) is replaced by the measured handler-only and HTTP round-trip numbers for both backends, each with what it measures. |
| 190-195 | IV-F Design choices | Rewrite "A linear text model for Task 1" as "BERTweet replaced TF-IDF under a pre-registered dev rule (+0.0195 dev weighted F1, 95% CI [+0.0049, +0.0340])". Add: Task 1 stays text-only because fusion ties on dev (tie goes to the simpler model); L4a is kept over the guide's core model. |
| 200-201 | V-A Implementation | Add BERTweet fine-tuning and the GloVe BiLSTM to the training description. |
| 203-204 | V-B Metrics | Task 1 headline = weighted F1 (with macro-F1), Task 2 headline = macro-F1; add top-2 accuracy (Task 2), per-class precision/recall, confusion matrices, fusion gain, error analysis. |
| 210 | V-C Protocol | "Task 1 and Task 3 involve no stochastic training" is now false for Task 1 (BERTweet, seeds 42/43/44); fix it. Add GPU non-determinism (94.3% test-prediction agreement on the retrain) and that the deployed checkpoint is a retrain. |
| 214-216 | V-C Protocol | Rules fixed in advance now include the Day 4 rules (`SELECTION_RULE_DAY4.md`); the verification sentence covers the Day 4 section. |
| 222-240 | VI Table I | Task 1 row: deployed BERTweet and the 3-seed mean (weighted F1, macro-F1, accuracy). Paragraph "Task 1 reaches 0.751 macro-F1" updated. |
| 242-262 | VI Ofli table | Add an INDICATIVE row: BERTweet on the Task 1 test rows where text and image labels agree (n = 1,534; this is the agreed subset of datasplit_all, not Ofli's agreed-label split, so still not like-for-like). |
| 453-475 | VII-C LODO table | Task 1 column: BERTweet, one seed per fold, labelled single-seed, next to the earlier TF-IDF column. |
| 480-498 | VII-D Calibration table | Add Task 1 BERTweet rows (3 seeds + deployed): accuracy, mean confidence, ECE, T, ECE after. |
| 526-558 | VII-F Class imbalance | Add the Task 1 weighting check: weighted vs unweighted BERTweet (post-hoc), the precision/recall trade-off, not a safety claim. |
| (new) | VII-I Model exploration | New subsection: Task 1 and Task 2 comparison tables, pre-registered comparison table, weighting trade-off, fusion-gain table (both baselines), error analysis, justification table. |
| 594-595 | VII-H What is established | Add the Day 4 items, split into established / not established. |
| 598-610 | VIII Discussion | Add "Each modality helps where it describes the label being scored" for both tasks; update the Limitations paragraph (retrain non-determinism, Task 1 LODO is single-seed, optional models not run, ~20 extra comparisons without correction). |
| 613-624 | IX Conclusion / Future work | Update the conclusion; remove the future-work items that are now done. |
| 630-651 | Bibliography | Add BERTweet, GloVe, LSTM, bidirectional RNN, Random Forest, scikit-learn, RoBERTa, Complement Naive Bayes, LIBLINEAR, MCC; each entry verified against its source (see the dossier). |

## Not changed by the build (kept as in `Capstone_report_Sem5.tex`)
Section III-A/B, the Task 2 ladder (Fig. 2 and its table), Task 3 text, the robustness and scale subsections except the lines above, VII-G (architecture choices, frozen-feature study). Numbers in these unchanged parts are carried over from the old report and are **not re-derived by `verify_findings.py`** (the verifier covered `report/ablation_section.tex`, not `Capstone_report_Sem5.tex`); the dossier lists them as "carried over, unverified".

## Slides
No slide file is in the repository. `docs/report_dossier.md` contains the slide map (keep / update / replace per slide, plus new slides).
