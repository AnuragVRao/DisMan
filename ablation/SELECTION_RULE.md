# Selection rule for the T2 imbalance treatment (fixed before any CLIP L/14 imbalance result existed)

Scope: image-only head on cached CLIP features, arms `none`, `inv`, `sqrt_inv`, `eff_num`, `focal1`, `focal2`, `focal2_inv`, `sampler`, `logit_adj`,
`none_posthoc`; seeds 42/43/44; backbones B/32 and L/14 (L2-normalised, the `L14_norm` condition of `b_t2_l14.py`).
Selection uses DEV only. The test set is scored once per run and never used to choose anything.

## Rule
Per backbone:
1. Mean dev macro-F1 (all 8 classes) per arm over the three seeds. The "best arm" is the one with the highest mean.
2. Paired bootstrap on dev for (best arm - each other arm): 2000 resamples of the dev rows. The same resampled row indices are used for every arm and
   every seed. For each resample the difference is computed per seed and then averaged over the three seeds, so the CI is for the seed-averaged difference.
   95% percentile CI.
3. An arm is "tied" with the best if the CI of (best - arm) includes 0. The best arm is always in its own tied set.
4. Among the tied set, pick the arm with the highest mean dev balanced accuracy.
5. Adopt a new headline loss only if the same arm is chosen on both backbones. Otherwise keep inverse-frequency (`inv`) and document the trade-off.

## Implementation notes (fixed now, to remove later discretion)
- Dev rows: the corrected humanitarian dev split (the same rows the checkpoint selection used).
- Dev predictions come from the dev logits saved next to each run (`artifacts/ablation/imbalance/<feat>_<arm>_seed<k>.npz`).
- `none_posthoc`: its temperature tau was chosen on dev, so its dev score is optimistically biased relative to the other arms. It stays in the comparison
  as a reported arm; if it is selected, that is stated alongside the selection-on-dev caveat.
- Bootstrap RNG seed 20241025 (as in `bootstrap_utils.py`).
- Test-set bootstraps (CI on test macro-F1 and on pairwise differences) are for REPORTING only, never selection, and are labelled as test-sampling
  uncertainty, separate from the seed standard deviation.

## Provenance note
The B/32 imbalance runs and their test numbers were already seen when this rule was written. The rule is therefore pre-registered only with respect
to the L/14 runs. For B/32 the rule is applied to the dev scores only.

---

# Addendum: Day 2 fused runs (written before any fused run that uses CLIP L/14 features)

Outcome of the rule above (see `results/ablation/G/g_imbalance_summary_seed42.json`): B/32 selected `inv`, L/14 selected `focal2_inv`. The two backbones
disagree, so **no new loss is adopted**. A post-hoc sensitivity analysis (`g_POSTHOC_selection_sensitivity_seed42.json`) was added afterwards; it is labelled
POST-HOC, was defined after the outcome was known, and does not change this decision.

Pre-registered for Day 2, on the fused model **L4a-L14** (the L4a recipe, i.e. pretrained DistilBERT, AMP, cached features, 10 epochs, batch 32, lr 1e-4, but with the
CLIP L/14 image vector (768-d, L2-normalised) in place of B/32). Note: L4a-L14 is a new model, not a re-run of L4a (the repository's L4a is B/32 + pretrained DistilBERT).
1. The headline loss is **inverse-frequency weighted cross-entropy** (`inv`), unchanged.
2. **One** extra arm is run on L4a-L14: **`none`** (unweighted cross-entropy), seeds 42, 43, 44, to test whether weighting matters in
   fusion. Same recipe, seeds, dev-based checkpoint selection and single test scoring as the `inv` runs (the `inv` runs of L4a-L14 are the comparator and
   are run with the same code path; they are the headline loss, not an extra arm).
3. **No other arm** (sqrt_inv, eff_num, focal, sampler, logit-adjusted, post-hoc shifts) is added to the fused configuration after seeing fused test numbers.
4. The outcome is reported whichever way it goes: seed means and std, the paired test bootstrap of `none` against `inv` (reporting only, labelled as a
   test-sampling CI, separate from the seed std), balanced accuracy, minority recall and per-class F1. "Better" or "worse" is used only where the paired CI
   excludes 0; otherwise "not distinguishable".
5. Nothing about the headline loss is changed on the basis of this one comparison; if `none` looks better, that is reported as an observation, and any change
   would require a new pre-registered decision.

## Run log for the L4a-L14 check (added after the runs; the pre-registration above is unchanged)
- The first launch of the six L4a-L14 runs (`inv` and `none`, seeds 42, 43, 44) failed on argument parsing (`--tag -L14_inv`: a value that starts with a dash is read as an
  option) before producing any result. The relaunch used `--tag=-L14_inv` and `--tag=-L14_none`. No results were seen or discarded.
- One validation run (B/32, `inv`, seed 42, saved as `results/ablation/G/t2_L4a_validate_seed42.json`) was made first to check that the new `--feat`, `--loss` and `--part`
  flags of `ablation/t2_train.py` leave the old behaviour unchanged; it reproduced the stored L4a seed-42 result exactly (dev and test macro-F1, accuracy, per-class F1, confusion
  matrix) and is kept.
- While the six runs were going, the session reported the background wrapper as stopped for low memory; the training processes continued and all six completed. None was re-run or discarded.
- Labelling, one rule for every comparison (the image-only study and L4a-L14): "better"/"worse" where the paired 95% CI excludes 0 by more than 0.002, "borderline" within 0.002 of
  zero, otherwise "not distinguishable"; for the injured-or-dead recall (8 dev / 10 test rows) a CI that includes 0 is "not resolvable". Every recall comparison states its row count.

---

# Day 2 addendum: architecture comparison (written and committed before any Day 2 run; no Day 2 result existed when this was written)

Scope: Task 2 only. Frozen cached features and small trainable heads; **no fine-tuned fusion runs and no documentation edits until the results are confirmed.**
Per-run JSONs go to `results/ablation/H/`. Seeds 42, 43, 44 for every arm. Selection uses DEV only; the test split is scored once per run and is never used to choose.

## Candidates (fixed here; none is added after any Day 2 test number is seen)
- **Image backbones** (frozen; features cached for the humanitarian train/dev/test rows in v2's row order):
  1. CLIP ViT-B/32 (v2's cached 512-d raw features; image tower + projection 87.8M parameters).
  2. CLIP ViT-L/14 (`openai/clip-vit-large-patch14`; the cached 768-d L2-normalised features, as the §9.7 / §9.9 L/14 condition; 304.0M).
  3. **DINOv2 ViT-B/14** (`facebook/dinov2-base`, one size only; 768-d `pooler_output` = CLS token after the final layer norm, raw; 86.6M).
  4. **ResNet-50** (`microsoft/resnet-50`, ImageNet-1k supervised; 2048-d global-average-pooled feature, raw; 23.5M).
  New backbones use their own HF image processor's deterministic eval preprocessing, fp16 forward, the L/14 extraction path (`ablation/b_clip_l14_extract.py`: each unique file embedded once, same row order and ids).
- **Text encoders** (frozen, sentence embeddings of the same cleaned tweet text the fine-tuned models read, L2-normalised): **all-MiniLM-L6-v2** (`sentence-transformers/all-MiniLM-L6-v2`, 384-d, 22.7M) and
  the stronger **all-mpnet-base-v2** (`sentence-transformers/all-mpnet-base-v2`, 768-d, 109.5M).
- **Heads** (d_model 256, dropout 0.2, 4 attention heads, the v2 optimiser recipe: AdamW wd 1e-4, lr 1e-4, 3 warm-up epochs then cosine, 10 epochs, batch 32, fp16 autocast, grad clip 1.0, inverse-frequency class weights, checkpoint = best dev macro-F1):
  `image_only`, `text_only`, `concat` (projected image and text vectors concatenated), `gated_late` (a separate classifier per modality; the final logits are g x image logits + (1 - g) x text logits, with a per-example scalar gate g = sigmoid of a linear map of the concatenated projections),
  and `cross_attention` (v2's two cross-attention blocks, with the DistilBERT stream replaced by the chosen encoder's frozen token features: the first 64 tokens, the first token as the text query; token features are extracted for the chosen encoder only, and the truncated fraction is recorded).
- Headline loss stays **inverse-frequency** in every arm.

## Decisions and the selection rule (the same rule for every decision)
Decisions, in order: D1 image backbone (image-only head); D2 text encoder (text-only head on the sentence embedding); D3 head (with the D1 and D2 winners); D4 hyperparameters (on the D3 winner).
For each decision: (1) mean dev macro-F1 over the three seeds; the best arm is the one with the highest mean. (2) Paired bootstrap on dev (2000 resamples of the dev rows, seed 20241025, the same rows for every arm and seed; per resample the difference is computed per seed and averaged over the seeds; 95% percentile CI) of (best arm - each other arm).
(3) An arm is **tied** with the best if the CI of the difference includes 0; the best arm is always tied. (4) Among the tied arms pick the **simplest**: the fewest trainable parameters (counted from the instantiated head, including its input projection), then the smaller frozen backbone(s) it consumes
(parameter counts above; a head consuming two backbones counts both). The highest dev balanced accuracy is NOT used. For D4 (equal-size configurations): fewer parameters = smaller d_model, then the configuration closest to the v2 defaults (lr 1e-4, dropout 0.2) in grid steps, then the lower lr.
Note the literal reading: "fewest trainable parameters" is applied first, so a 512-d head (CLIP B/32) is cheaper than a 2048-d head (ResNet-50) even though ResNet-50 is the smaller backbone; the backbone size only breaks exact ties.

## D4 grid (fixed now)
lr in {3e-5, 1e-4, 3e-4} x dropout in {0.1, 0.2, 0.4} x d_model in {128, 256, 512}: 27 configurations x 3 seeds, everything else as above. Dev only.

## Validation before anything new is added
The existing image-only B/32 (raw) and L/14 (normalised) runs must reproduce the stored results exactly under the Day 2 protocol; the Day 2 multi-input trainer used for D3 must reproduce the stored image-only run exactly for `image_only` before any new head is run.

## Reporting and labels
One table: dev macro-F1, test macro-F1 (seed mean +- seed std), balanced accuracy, recall of injured-or-dead (test n = 10) and vehicle damage (test n = 36; dev n = 8 and 38), trainable parameters. Paired dev bootstraps and the rule's pick per decision.
Test intervals are test-sampling intervals, for reporting only, separate from the seed std; they are never used to select. Labels as in findings 9.9: "better"/"worse" only where the paired CI excludes 0 (by more than 0.002; a bound within 0.002 is "borderline"), otherwise "not distinguishable"; the injured-or-dead recall is "not resolvable" only where its CI includes 0. Every recall comparison states its test n.
No multiplicity correction is applied, and the report says so.
