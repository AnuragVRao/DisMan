# Task 2 results, text / image / fusion (image label, 8 classes; mean ± std over seeds)

| Model | dev macro-F1 | test macro-F1 | test wF1 | test acc | test bal-acc | test MCC | test top-2 | test ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| majority class ('Not humanitarian') | 0.0818 | 0.0801 | 0.3023 | 0.4716 | 0.1250 | 0.0000 | 0.5110 | 0.5000 |
| TF-IDF + LR (class-weighted) | 0.2700 | 0.2779 | 0.4754 | 0.4573 | 0.3570 | 0.2880 | 0.6866 | 0.7115 |
| BERTweet text-only (rule-8 recipe; rule-13 pick) | 0.3083 ± 0.0025 | 0.3010 ± 0.0047 | 0.5067 ± 0.0046 | 0.4762 ± 0.0036 | 0.3903 ± 0.0053 | 0.3251 ± 0.0052 | 0.6844 ± 0.0052 | 0.7873 ± 0.0045 |
| BERTweet text-only (DistilBERT-matched recipe) | 0.3113 ± 0.0010 | 0.3076 ± 0.0177 | 0.5262 ± 0.0124 | 0.5083 ± 0.0131 | 0.3602 ± 0.0066 | 0.3435 ± 0.0138 | 0.7310 ± 0.0219 | 0.7914 ± 0.0134 |
| DistilBERT text-only (re-inferred) | 0.2948 ± 0.0048 | 0.2937 ± 0.0140 | 0.4993 ± 0.0083 | 0.4861 ± 0.0100 | 0.3198 ± 0.0198 | 0.3081 ± 0.0102 | 0.7257 ± 0.0047 | 0.7614 ± 0.0034 |
| image-only head, CLIP B/32 | 0.5679 ± 0.0083 | 0.6014 ± 0.0053 | 0.7931 ± 0.0016 | 0.7878 ± 0.0021 | 0.6511 ± 0.0021 | 0.7082 ± 0.0020 | 0.9435 ± 0.0009 | 0.9502 ± 0.0057 |
| image-only head, CLIP L/14 | 0.6124 ± 0.0024 | 0.6438 ± 0.0089 | 0.8163 ± 0.0036 | 0.8124 ± 0.0039 | 0.6928 ± 0.0073 | 0.7398 ± 0.0053 | 0.9444 ± 0.0022 | 0.9604 ± 0.0042 |
| L4a: CLIP B/32 + DistilBERT cross-attention (demo model) | 0.5849 ± 0.0080 | 0.5984 ± 0.0101 | 0.8004 ± 0.0057 | 0.7994 ± 0.0063 | 0.6104 ± 0.0076 | 0.7172 ± 0.0091 | 0.9395 ± 0.0030 | 0.9527 ± 0.0041 |
| late fusion (calibrated avg): BERTweet + CLIP L/14 head | 0.5974 ± 0.0033 | 0.6223 ± 0.0068 | 0.8044 ± 0.0031 | 0.7999 ± 0.0034 | 0.6905 ± 0.0076 | 0.7230 ± 0.0047 | 0.9200 ± 0.0019 | 0.9355 ± 0.0031 |
| early fusion BERTweet (fine-tuned) + ResNet50 (frozen) [guide's core model] | 0.5605 ± 0.0097 | 0.5535 ± 0.0040 | 0.7510 ± 0.0151 | 0.7485 ± 0.0187 | 0.5873 ± 0.0177 | 0.6532 ± 0.0127 | 0.9140 ± 0.0076 | 0.9430 ± 0.0042 |
| post-hoc: early fusion BERTweet (fine-tuned) + CLIP B/32 (frozen) | 0.5709 ± 0.0021 | 0.6063 ± 0.0101 | 0.7981 ± 0.0057 | 0.7933 ± 0.0060 | 0.6458 ± 0.0118 | 0.7139 ± 0.0075 | 0.9397 ± 0.0012 | 0.9489 ± 0.0011 |

## Pre-registered comparisons (dev paired bootstrap, seed-averaged, 10,000 resamples; dev decides, test is reporting only)

| Comparison | split | metric | point | 95% CI | verdict |
|---|---|---|---|---|---|
| t2_early_bertweet_resnet50 − t2_L4a (core_vs_L4a) | dev | macro_f1 | -0.0243 | [-0.0579, +0.0064] | not distinguishable |
| t2_early_bertweet_resnet50 − t2_L4a (core_vs_L4a) | dev | weighted_f1 | -0.0341 | [-0.0484, -0.0200] | worse |
| t2_early_bertweet_resnet50 − t2_L4a (core_vs_L4a) | test | macro_f1 | -0.0450 | [-0.0735, -0.0156] | worse |
| t2_early_bertweet_resnet50 − t2_L4a (core_vs_L4a) | test | weighted_f1 | -0.0494 | [-0.0627, -0.0360] | worse |
| t2_early_bertweet_resnet50 − t2_img_clip_b32 (core_vs_image_only_clip_b32) | dev | macro_f1 | -0.0074 | [-0.0428, +0.0246] | not distinguishable |
| t2_early_bertweet_resnet50 − t2_img_clip_b32 (core_vs_image_only_clip_b32) | dev | weighted_f1 | -0.0260 | [-0.0413, -0.0105] | worse |
| t2_early_bertweet_resnet50 − t2_img_clip_b32 (core_vs_image_only_clip_b32) | test | macro_f1 | -0.0479 | [-0.0795, -0.0152] | worse |
| t2_early_bertweet_resnet50 − t2_img_clip_b32 (core_vs_image_only_clip_b32) | test | weighted_f1 | -0.0422 | [-0.0566, -0.0277] | worse |
| bertweet_text_matched − t2_distilbert_text (bertweet_matched_vs_distilbert_text, post-hoc) | dev | macro_f1 | +0.0165 | [-0.0004, +0.0334] | not distinguishable |
| bertweet_text_matched − t2_distilbert_text (bertweet_matched_vs_distilbert_text, post-hoc) | dev | weighted_f1 | +0.0195 | [+0.0052, +0.0336] | better |
| bertweet_text_matched − t2_distilbert_text (bertweet_matched_vs_distilbert_text, post-hoc) | test | macro_f1 | +0.0138 | [-0.0071, +0.0353] | not distinguishable |
| bertweet_text_matched − t2_distilbert_text (bertweet_matched_vs_distilbert_text, post-hoc) | test | weighted_f1 | +0.0269 | [+0.0123, +0.0410] | better |
| t2_early_bertweet_clip_b32 − t2_L4a (posthoc_clip_b32_vs_L4a, post-hoc) | dev | macro_f1 | -0.0139 | [-0.0317, +0.0025] | not distinguishable |
| t2_early_bertweet_clip_b32 − t2_L4a (posthoc_clip_b32_vs_L4a, post-hoc) | dev | weighted_f1 | -0.0047 | [-0.0122, +0.0031] | not distinguishable |
| t2_early_bertweet_clip_b32 − t2_L4a (posthoc_clip_b32_vs_L4a, post-hoc) | test | macro_f1 | +0.0078 | [-0.0178, +0.0337] | not distinguishable |
| t2_early_bertweet_clip_b32 − t2_L4a (posthoc_clip_b32_vs_L4a, post-hoc) | test | weighted_f1 | -0.0023 | [-0.0102, +0.0059] | not distinguishable |
| t2_early_bertweet_clip_b32 − t2_early_bertweet_resnet50 (posthoc_clip_b32_vs_core, post-hoc) | dev | macro_f1 | +0.0104 | [-0.0192, +0.0434] | not distinguishable |
| t2_early_bertweet_clip_b32 − t2_early_bertweet_resnet50 (posthoc_clip_b32_vs_core, post-hoc) | dev | weighted_f1 | +0.0294 | [+0.0155, +0.0437] | better |
| t2_early_bertweet_clip_b32 − t2_early_bertweet_resnet50 (posthoc_clip_b32_vs_core, post-hoc) | test | macro_f1 | +0.0528 | [+0.0269, +0.0785] | better |
| t2_early_bertweet_clip_b32 − t2_early_bertweet_resnet50 (posthoc_clip_b32_vs_core, post-hoc) | test | weighted_f1 | +0.0471 | [+0.0342, +0.0599] | better |

## Fusion gain = fused − best single modality (per seed, best on dev macro-F1 among BERTweet text-only and the CLIP L/14 image head)

| Fused model | dev gain macro-F1 | dev gain wF1 | test gain macro-F1 | test gain wF1 |
|---|---|---|---|---|
| late_bertweet_text_rule8+t2_img_clip_l14_inv | -0.0150 ± 0.0013 | -0.0096 ± 0.0031 | -0.0214 ± 0.0093 | -0.0120 ± 0.0029 |
| t2_early_bertweet_resnet50 | -0.0519 ± 0.0103 | -0.0572 ± 0.0199 | -0.0903 ± 0.0122 | -0.0653 ± 0.0186 |
| t2_early_bertweet_clip_b32 | -0.0415 ± 0.0034 | -0.0278 ± 0.0011 | -0.0375 ± 0.0158 | -0.0182 ± 0.0091 |

### Fusion gain vs own components = fused − the better (per seed, on dev macro-F1) of the two single-modality models it is built from; mean ± std over seeds

| Fused model | own components | dev gain macro-F1 | dev gain wF1 | test gain macro-F1 | test gain wF1 |
|---|---|---|---|---|---|
| t2_early_bertweet_resnet50 | bertweet_text_matched + d1_resnet50 | +0.0187 ± 0.0106 | +0.0231 ± 0.0216 | +0.0270 ± 0.0026 | +0.0158 ± 0.0137 |
| late_bertweet_text_rule8+t2_img_clip_l14_inv | bertweet_text_rule8 + t2_img_clip_l14_inv | -0.0150 ± 0.0013 | -0.0096 ± 0.0031 | -0.0214 ± 0.0093 | -0.0120 ± 0.0029 |
| t2_L4a | t2_distilbert_text + t2_img_clip_b32 | +0.0170 ± 0.0161 | +0.0081 ± 0.0048 | -0.0029 ± 0.0088 | +0.0072 ± 0.0070 |
| t2_early_bertweet_clip_b32 | bertweet_text_matched + t2_img_clip_b32 | +0.0030 ± 0.0073 | +0.0034 ± 0.0033 | +0.0049 ± 0.0119 | +0.0049 ± 0.0073 |

(The late-fusion row equals the 'best single modality' table above, since its components are the same two models; the core model's components are BERTweet text-only (matched recipe, its own) and the stored frozen-ResNet50 image-only head.)

## Test set by label agreement (normalised label_text == label_image): agreed n=955, disagreed n=1282; 'Not humanitarian' share 0.53 (agreed) vs 0.43 (disagreed). Reported macro-F1 averages the classes present in the subset's true labels or predictions (the report's Table VIII convention); the fixed 8-class average (absent classes score 0) is in the `all8` columns. Mean ± std over 3 seeds, descriptive only.

| Model | agreed macro-F1 (present) | agreed macro-F1 (all8) | agreed wF1 | agreed acc | disagreed macro-F1 (present) | disagreed macro-F1 (all8) | disagreed wF1 | disagreed acc |
|---|---|---|---|---|---|---|---|---|
| bertweet_text_rule8 | 0.479 ± 0.035 | 0.419 ± 0.030 | 0.744 ± 0.012 | 0.713 ± 0.018 | 0.194 ± 0.004 | 0.194 ± 0.004 | 0.335 ± 0.017 | 0.300 ± 0.014 |
| bertweet_text_matched | 0.504 ± 0.023 | 0.441 ± 0.021 | 0.737 ± 0.008 | 0.716 ± 0.010 | 0.201 ± 0.020 | 0.201 ± 0.020 | 0.375 ± 0.021 | 0.354 ± 0.020 |
| t2_distilbert_text | 0.482 ± 0.061 | 0.422 ± 0.053 | 0.703 ± 0.017 | 0.685 ± 0.019 | 0.198 ± 0.010 | 0.198 ± 0.010 | 0.359 ± 0.002 | 0.338 ± 0.005 |
| t2_img_clip_l14_inv | 0.708 ± 0.024 | 0.620 ± 0.021 | 0.860 ± 0.005 | 0.856 ± 0.005 | 0.614 ± 0.011 | 0.614 ± 0.011 | 0.787 ± 0.003 | 0.780 ± 0.003 |
| t2_img_clip_b32 | 0.696 ± 0.031 | 0.609 ± 0.027 | 0.850 ± 0.003 | 0.845 ± 0.004 | 0.563 ± 0.004 | 0.563 ± 0.004 | 0.753 ± 0.005 | 0.745 ± 0.005 |
| t2_L4a | 0.745 ± 0.026 | 0.652 ± 0.023 | 0.867 ± 0.008 | 0.866 ± 0.008 | 0.549 ± 0.009 | 0.549 ± 0.009 | 0.753 ± 0.004 | 0.750 ± 0.005 |
| late_bertweet_text_rule8+t2_img_clip_l14_inv | 0.729 ± 0.019 | 0.638 ± 0.017 | 0.880 ± 0.001 | 0.877 ± 0.001 | 0.573 ± 0.012 | 0.573 ± 0.012 | 0.753 ± 0.006 | 0.743 ± 0.006 |
| t2_early_bertweet_resnet50 | 0.670 ± 0.022 | 0.586 ± 0.020 | 0.836 ± 0.014 | 0.832 ± 0.016 | 0.503 ± 0.008 | 0.503 ± 0.008 | 0.691 ± 0.016 | 0.686 ± 0.021 |
| t2_early_bertweet_clip_b32 | 0.713 ± 0.041 | 0.624 ± 0.036 | 0.860 ± 0.011 | 0.858 ± 0.013 | 0.565 ± 0.009 | 0.565 ± 0.009 | 0.755 ± 0.009 | 0.745 ± 0.009 |

Table VIII check: L4a reproduces the report's subset numbers with the present-classes definition (agreed 0.745 vs stored 0.745, disagreed 0.549 vs stored 0.549); the all-8 definition gives 0.652 / 0.549.

### Fusion gain on the subsets (test; fused − best single modality of that seed)

| Fused model | agreed macro-F1 | agreed wF1 | disagreed macro-F1 | disagreed wF1 |
|---|---|---|---|---|
| late_bertweet_text_rule8+t2_img_clip_l14_inv | +0.0209 | +0.0201 | -0.0409 | -0.0343 |
| t2_early_bertweet_resnet50 | -0.0381 | -0.0242 | -0.1114 | -0.0955 |
| t2_early_bertweet_clip_b32 | +0.0053 | +0.0005 | -0.0487 | -0.0320 |

## Tie-break applied to the two pre-registered comparisons (dev macro-F1)

Governing text: SELECTION_RULE_DAY4.md rule 2, "Ties go to the simplest model." Rules 14-19 contain no tie wording of their own (rule 17 only defines the label "not distinguishable": the 95% CI includes 0). "Simplest" is taken as defined in the Day 2 addendum of SELECTION_RULE.md and used in D2/§VII-G: the fewest trainable parameters (counted from the instantiated model, frozen backbones excluded), then the smaller frozen backbone.

| Comparison | dev macro-F1 difference [95% CI] | tied? | trainable parameters (core / other) | the rule picks | dev weighted F1 (corroboration only) |
|---|---|---|---|---|---|
| core − t2_L4a | -0.0243 [-0.0579, +0.0064] | yes | 135,755,784 / 67,352,840 | **t2_L4a** (fewest trainable parameters) | -0.0341 [-0.0484, -0.0200] (worse) |
| core − t2_img_clip_b32 | -0.0074 [-0.0428, +0.0246] | yes | 135,755,784 / 199,688 | **t2_img_clip_b32** (fewest trainable parameters) | -0.0260 [-0.0413, -0.0105] (worse) |

The pick rests on the tie-break (simplicity), not on a resolved difference: neither comparison shows the core model better, and dev weighted F1 (worse for the core model in both) is mentioned as corroboration only, not as a criterion.
