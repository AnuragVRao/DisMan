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

## Fusion gain = fused − best single modality (per seed, best on dev macro-F1 among BERTweet text-only and the CLIP L/14 image head)

| Fused model | dev gain macro-F1 | dev gain wF1 | test gain macro-F1 | test gain wF1 |
|---|---|---|---|---|
| late_bertweet_text_rule8+t2_img_clip_l14_inv | -0.0150 ± 0.0013 | -0.0096 ± 0.0031 | -0.0214 ± 0.0093 | -0.0120 ± 0.0029 |
| t2_early_bertweet_resnet50 | -0.0519 ± 0.0103 | -0.0572 ± 0.0199 | -0.0903 ± 0.0122 | -0.0653 ± 0.0186 |

## Test set by label agreement (normalised label_text == label_image): agreed n=955, disagreed n=1282; 'Not humanitarian' share 0.53 (agreed) vs 0.43 (disagreed). Macro-F1 over all 8 classes (classes absent from a subset score 0), descriptive only.

| Model | agreed macro-F1 | agreed wF1 | agreed acc | disagreed macro-F1 | disagreed wF1 | disagreed acc |
|---|---|---|---|---|---|---|
| bertweet_text_rule8 | 0.4191 | 0.7445 | 0.7127 | 0.1942 | 0.3346 | 0.3001 |
| t2_img_clip_l14_inv | 0.6195 | 0.8599 | 0.8558 | 0.6140 | 0.7868 | 0.7800 |
| t2_img_clip_b32 | 0.6089 | 0.8496 | 0.8454 | 0.5634 | 0.7534 | 0.7449 |
| t2_L4a | 0.6520 | 0.8672 | 0.8660 | 0.5490 | 0.7532 | 0.7499 |
| late_bertweet_text_rule8+t2_img_clip_l14_inv | 0.6378 | 0.8800 | 0.8768 | 0.5731 | 0.7525 | 0.7426 |
| t2_early_bertweet_resnet50 | 0.5862 | 0.8357 | 0.8321 | 0.5026 | 0.6914 | 0.6862 |

### Fusion gain on the subsets (test; fused − best single modality of that seed)

| Fused model | agreed macro-F1 | agreed wF1 | disagreed macro-F1 | disagreed wF1 |
|---|---|---|---|---|
| late_bertweet_text_rule8+t2_img_clip_l14_inv | +0.0183 | +0.0201 | -0.0409 | -0.0343 |
| t2_early_bertweet_resnet50 | -0.0333 | -0.0242 | -0.1114 | -0.0955 |
