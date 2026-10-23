# ABLATION_PLAN.md — CrisisMMD v1 → v2 comparison + ablation study

Read CLAUDE.md first. This file adds a two-day experiment plan on top of the existing v2 pipeline.
Do not change v2's reported headline numbers or default behaviour; all new behaviour goes behind flags.

## Ground rules (apply to every experiment)

- **One test set for everything:** the corrected official splits (all 7 events, every image resolving).
  T3 uses the tweet-level leakage-free split already used for the v2 headline.
- Model selection on **dev only**. Test is evaluated once per final config.
- Metrics: macro-F1 (headline), accuracy, weighted F1, plus per-class F1. Save confusion matrices.
- **Seeds:** T2 configs run with seeds {42, 43, 44}; report mean ± std. T1/T3 are deterministic: one run.
- Every run writes one JSON row to `results/ablation/<part>/<config>_seed<k>.json`:
  `{part, task, config, seed, dev_macro_f1, test_acc, test_macro_f1, test_weighted_f1, per_class_f1, epochs_trained, best_epoch, wall_clock_s, git_commit}`.
- Do not overwrite existing v2 artifacts. New checkpoints go under `artifacts/ablation/`.
- Before launching any batch, **time one L0 run and one L4 run** and report both times. Stop and ask
  if a single L0 run exceeds 30 minutes.

## Part A — v1 → v2 ladder

### Step 0: recover v1 exactly
Read the old notebook and write `docs/v1_config.md` listing v1's T2 settings as they were:
epochs, LR, batch size, optimiser, scheduler, CLIP activation/output used, whether CLIP was frozen,
checkpoint rule (last epoch vs best), loss, and how image paths were resolved.
Quote the notebook lines each setting comes from. If any setting is ambiguous, list it as ambiguous
and ask; do not guess.

### T2 ladder (flags on the training script; 3 seeds each)

| Rung | Data | CLIP activation | Selection | Caching/AMP |
|---|---|---|---|---|
| L0 | v1 buggy train/dev (earthquake events missing, i.e. old `lstrip` behaviour) | v1 | v1 rule | off |
| L1 | full corrected train/dev | v1 | v1 rule | off |
| L2 | full | fixed | v1 rule | off |
| L3 | full | fixed | 10-epoch cap, best dev macro-F1 | off |
| L4 | full | fixed | 10-epoch cap, best dev macro-F1 | on (= v2) |

All rungs are evaluated on the **full corrected test set**, including L0.
L0's buggy data must be reproduced by applying the old path logic to train/dev only, not to test.

Expected: L4 ≈ L3 (difference within seed std). If not, stop and report; do not continue to Part B.

### T3 ladder (no training; single run each)

| Rung | Index contents | Labels |
|---|---|---|
| L0 | v1 index (including test rows, as in v1) | v1 (with the irrelevant-class pollution) |
| L1 | test rows removed | v1 |
| L2 | test rows removed | pollution removed |
| L3 | tweet-level leakage-free split (= v2) | clean |

Evaluate every rung on the v2 T3 test set. Expected: scores decrease or stay flat from L0 → L3.
For L0, report how many test queries retrieve themselves as a nearest neighbour.

### T1
No ladder. Re-run v1 and v2 T1 and assert identical accuracy and weighted F1 (regression check).

## Part B — Component ablation of v2

### T2 (3 seeds each; the full model = L4 runs, reuse them)
- `text_only`: DistilBERT + head, no image stream
- `image_only`: cached CLIP embedding + MLP head
- `concat`: concatenate pooled text + image, MLP head, no attention blocks
- `no_t2i`: remove the text→image attention block, keep image→text

Keep every other hyperparameter identical to v2. Report parameter count per variant.

### T3 (deterministic)
- k ∈ {1, 3, 5, 10, 20}: report dev and test; headline stays k=5
- voting: majority (current) vs similarity-weighted vs class-balanced (divide votes by class frequency in index)
- embeddings: image-only, text-only, fused
- logistic regression probe (sklearn, class_weight="balanced", C tuned on dev) on the same embeddings

## Part C — Robustness

### C1 Missing modality (no retraining)
At test time for T2 and T3, evaluate with: (a) image replaced by the embedding of a blank grey image;
(b) text replaced by an empty string. Report the drop from the full-input score.

### C2 Leave-one-disaster-out (T1 and T3 only)
For each of the 7 events: train/build the index on the other 6 events' train split, test on the
held-out event's test split. Assert the held-out event contributes zero rows to the training data / index.
Report a 7-row table + macro average. T2 is out of scope (mark as future work).

## Deliverables

1. `results/ablation/summary.csv`: one row per (part, task, config) with mean ± std across seeds.
2. `results/ablation/tables.tex`: LaTeX tables for A (T2 ladder, T3 ladder), B (T2, T3), C1, C2,
   matching the IEEE report's table style.
3. `results/ablation/ladder_t2.pdf`: bar chart of T2 macro-F1 per rung with std error bars.
4. `docs/ablation_findings.md`: short factual notes, including any expectation above that failed.

## Stop-and-ask conditions
- v1 config cannot be recovered unambiguously
- L4 vs L3 difference exceeds one seed std
- any variant's test macro-F1 is below the majority-class baseline
- leave-one-disaster-out leakage assertion fails
