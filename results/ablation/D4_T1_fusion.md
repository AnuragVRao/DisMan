# Task 1 fusion (text label; mean over 3 seeds)

## Fusion gain = fused - best unimodal (best unimodal chosen per seed on dev weighted F1)

| Fused model | dev gain wF1 | dev gain macro-F1 | test gain wF1 | test gain macro-F1 | positive every seed (dev wF1 / test wF1) |
|---|---|---|---|---|---|
| late fusion (calibrated avg) BERTweet + CLIP L/14 head | +0.0041 ± 0.0004 | +0.0057 ± 0.0003 | +0.0064 ± 0.0017 | +0.0094 ± 0.0024 | True / True |
| early fusion BERTweet (fine-tuned) + ResNet50 (frozen)  [guide's core model] | +0.0040 ± 0.0035 | +0.0047 ± 0.0061 | +0.0062 ± 0.0014 | +0.0081 ± 0.0036 | True / True |
| early fusion BERTweet (fine-tuned) + CLIP L/14 (frozen)  [optional] | +0.0043 ± 0.0012 | +0.0043 ± 0.0016 | +0.0067 ± 0.0045 | +0.0078 ± 0.0055 | True / True |

All fusion variants tie with each other on dev (mean dev weighted F1 spread 0.0003: late_bertweet+img_clip_l14 0.8346, early_bertweet_resnet50 0.8345, early_bertweet_clip_l14 0.8348); none is 'best', and the variant used in comparison (i) below was only the nominal dev maximum.

Late fusion has a lower test ROC-AUC (0.8841) than BERTweet alone (0.8972) although its F1 is higher: equal-weight averaging with a weaker image model (CLIP head test ROC-AUC 0.7967) plausibly degrades the ranking while leaving the thresholded labels slightly better. This explanation was not tested, and the weights were not tuned (tuning them would be a post-hoc arm).

## Paired bootstrap, seed-averaged, 10,000 resamples (dev decides; test is reporting only)

| Comparison | split | metric | point | 95% CI | verdict |
|---|---|---|---|---|---|
| early_bertweet_clip_l14 − bertweet (best_fused_vs_text_only) | dev | weighted_f1 | +0.0043 | [-0.0018, +0.0102] | not distinguishable |
| early_bertweet_clip_l14 − bertweet (best_fused_vs_text_only) | dev | macro_f1 | +0.0043 | [-0.0035, +0.0119] | not distinguishable |
| early_bertweet_clip_l14 − bertweet (best_fused_vs_text_only) | test | weighted_f1 | +0.0067 | [+0.0010, +0.0124] | better |
| early_bertweet_clip_l14 − bertweet (best_fused_vs_text_only) | test | macro_f1 | +0.0078 | [+0.0007, +0.0149] | better |
| early_bertweet_resnet50 − bertweet (guides_core_vs_deployed) | dev | weighted_f1 | +0.0040 | [-0.0014, +0.0095] | not distinguishable |
| early_bertweet_resnet50 − bertweet (guides_core_vs_deployed) | dev | macro_f1 | +0.0047 | [-0.0021, +0.0115] | not distinguishable |
| early_bertweet_resnet50 − bertweet (guides_core_vs_deployed) | test | weighted_f1 | +0.0062 | [+0.0009, +0.0115] | better |
| early_bertweet_resnet50 − bertweet (guides_core_vs_deployed) | test | macro_f1 | +0.0081 | [+0.0014, +0.0147] | better |

## Test split by label agreement (label_text == label_image): agreed n=1534, disagreed n=703; informative share 0.67 (agreed) vs 0.83 (disagreed)

| Model | agreed wF1 | agreed macro-F1 | agreed acc | disagreed wF1 | disagreed macro-F1 | disagreed acc |
|---|---|---|---|---|---|---|
| bertweet | 0.8608 | 0.8397 | 0.8635 | 0.7955 | 0.6538 | 0.7890 |
| img_clip_l14 | 0.8832 | 0.8662 | 0.8846 | 0.5346 | 0.3607 | 0.4708 |
| late_bertweet+img_clip_l14 | 0.9016 | 0.8868 | 0.9033 | 0.7347 | 0.5670 | 0.7183 |
| early_bertweet_resnet50 | 0.8742 | 0.8550 | 0.8766 | 0.7880 | 0.6447 | 0.7790 |
| early_bertweet_clip_l14 | 0.8720 | 0.8521 | 0.8748 | 0.7930 | 0.6481 | 0.7871 |

## Dev split by label agreement (label_text == label_image): agreed n=1573, disagreed n=664; informative share 0.67 (agreed) vs 0.84 (disagreed)

| Model | agreed wF1 | agreed macro-F1 | agreed acc | disagreed wF1 | disagreed macro-F1 | disagreed acc |
|---|---|---|---|---|---|---|
| bertweet | 0.8497 | 0.8267 | 0.8527 | 0.7938 | 0.6419 | 0.7836 |
| img_clip_l14 | 0.8759 | 0.8564 | 0.8790 | 0.5411 | 0.3637 | 0.4724 |
| late_bertweet+img_clip_l14 | 0.8843 | 0.8663 | 0.8871 | 0.7347 | 0.5556 | 0.7144 |
| early_bertweet_resnet50 | 0.8588 | 0.8370 | 0.8618 | 0.7867 | 0.6294 | 0.7761 |
| early_bertweet_clip_l14 | 0.8568 | 0.8345 | 0.8601 | 0.7908 | 0.6300 | 0.7836 |
