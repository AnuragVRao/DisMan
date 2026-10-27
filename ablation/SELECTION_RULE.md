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
