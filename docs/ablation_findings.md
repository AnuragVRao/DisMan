# Ablation findings (running notes)

Facts only; numbers come from `results/ablation/` (not committed yet). Status: Part A T3 ladder done, T2 ladder not started.

## v1 specifics that the ladders must respect

* **v1 applied `RandomResizedCrop` (scale 0.9-1.0) to dev and test images too**, not only train: `capstone-final.ipynb` cell 15 keeps open_clip's
  *train* transform (`clip_preprocess_t2`) and cell 18 uses it for all three splits. v1's logged dev and test T2 scores are therefore measured on randomly cropped images.
  Consequence: CLIP features cannot be cached for L0-L2; L0-L2 run CLIP online. v2's cached features removed the augmentation as a side effect.
* **v2's Task 2 text encoder starts from random initialisation, not pretrained DistilBERT.** `crisis_common.DualStreamClassifier` builds
  `DistilBertModel(bert_config)` whenever a config is passed, and the v2 training cell passes one (to avoid hub access in the demo).
  Evidence: in `artifacts/task2_model.pt` the text encoder's word-embedding std is 0.0200 (the random-init value; pretrained is 0.0470) and its relative L2 distance to
  pretrained DistilBERT (1.06) equals that between a random init and pretrained. v1 used `DistilBertModel.from_pretrained('distilbert-base-uncased')`.
  This is not listed in RESULTS.md's audit. v2's reported numbers are unchanged; the ablation presets expose it as `bert=pretrained|random` (L0-L3b pretrained, L4 random = v2).

## Part A, T3 ladder (deterministic, v2 vote rule, HF CLIP verified as v1's)

| Rung | Index rows | Acc | Macro-F1 | Weighted-F1 | Test queries with self in top-5 |
|---|---|---|---|---|---|
| L0 v1 index (incl. test, "Don't know" pollution) | 18,082 | 0.7221 | 0.6880 | 0.7630 | 529/529 (top-1: 528) |
| L1 test rows removed | 17,553 | 0.5028 | 0.3777 | 0.5328 | 0 |
| L2 "Don't know" rows removed (train+dev) | 2,997 | 0.6767 | 0.5037 | 0.6318 | 0 |
| L3 dev removed (train only) | 2,468 | 0.6749 | 0.5091 | 0.6309 | 0 |
| L4 tweet siblings removed (= v2) | 2,361 | 0.6673 | 0.4914 | 0.6202 | 0 |

* The pollution removed at L2 is 14,556 rows: 14,455 images with no damage annotation (NaN, mapped to "Don't know" by v1) + 101 explicit "dont_know_or_cant_judge".
* **L1 -> L2 (+0.126 macro-F1): expected effect** of removing the "Don't know" rows. With them in the index 786 of 2,645 neighbour votes (30%) went to a class that cannot occur in test, and 143 test queries were predicted "Don't know".
* **L2 -> L3 (dev rows removed): no measurable effect** (macro-F1 0.5037 -> 0.5091, accuracy 0.6767 -> 0.6749; 529 test rows, differences are within noise).
* L3 -> L4 (tweet siblings removed): macro-F1 0.5091 -> 0.4914. 67 test queries had a same-tweet sibling in their top-5 at L3; none at L4.
* L0 is higher than L1-L4 mainly because every test query retrieves itself (529/529 in the top-5); L0 -> L1 removes that.
* L4 reproduces v2's headline exactly (0.6673 / 0.4914 / 0.6202).
