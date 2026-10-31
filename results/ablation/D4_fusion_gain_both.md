# Fusion gain, both tasks, both baselines (mean ± std over 3 seeds; baseline picked per seed on dev headline; test reported afterwards)

## T1 (headline: weighted_f1; the other metric is in the right-hand columns)

| Fused model | own components | fused weighted_f1 (dev / test) | gain vs best unimodal weighted_f1 (dev / test) | gain vs own components weighted_f1 (dev / test) | gain vs best unimodal macro_f1 (dev / test) | gain vs own components macro_f1 (dev / test) |
|---|---|---|---|---|---|---|
| late fusion: BERTweet + CLIP L/14 head | bertweet + img_clip_l14 | 0.8346 / 0.8446 | +0.0041 ± 0.0004 / +0.0064 ± 0.0017 | +0.0041 ± 0.0004 / +0.0064 ± 0.0017 | +0.0057 ± 0.0003 / +0.0094 ± 0.0024 | +0.0057 ± 0.0003 / +0.0094 ± 0.0024 |
| core model: BERTweet (fine-tuned) + ResNet50 (frozen) | bertweet + img_resnet50 | 0.8345 / 0.8445 | +0.0040 ± 0.0035 / +0.0062 ± 0.0014 | +0.0040 ± 0.0035 / +0.0062 ± 0.0014 | +0.0047 ± 0.0061 / +0.0081 ± 0.0036 | +0.0047 ± 0.0061 / +0.0081 ± 0.0036 |
| BERTweet (fine-tuned) + CLIP L/14 (frozen) | bertweet + img_clip_l14 | 0.8348 / 0.8449 | +0.0043 ± 0.0012 / +0.0067 ± 0.0045 | +0.0043 ± 0.0012 / +0.0067 ± 0.0045 | +0.0043 ± 0.0016 / +0.0078 ± 0.0055 | +0.0043 ± 0.0016 / +0.0078 ± 0.0055 |

## T2 (headline: macro_f1; the other metric is in the right-hand columns)

| Fused model | own components | fused macro_f1 (dev / test) | gain vs best unimodal macro_f1 (dev / test) | gain vs own components macro_f1 (dev / test) | gain vs best unimodal weighted_f1 (dev / test) | gain vs own components weighted_f1 (dev / test) |
|---|---|---|---|---|---|---|
| late fusion: BERTweet + CLIP L/14 head | bertweet_text_rule8 + t2_img_clip_l14_inv | 0.5974 / 0.6223 | -0.0150 ± 0.0013 / -0.0214 ± 0.0093 | -0.0150 ± 0.0013 / -0.0214 ± 0.0093 | -0.0096 ± 0.0031 / -0.0120 ± 0.0029 | -0.0096 ± 0.0031 / -0.0120 ± 0.0029 |
| core model: BERTweet (fine-tuned) + ResNet50 (frozen) | bertweet_text_matched + H:d1_resnet50 | 0.5605 / 0.5535 | -0.0519 ± 0.0103 / -0.0903 ± 0.0122 | +0.0187 ± 0.0106 / +0.0270 ± 0.0026 | -0.0572 ± 0.0199 / -0.0653 ± 0.0186 | +0.0231 ± 0.0216 / +0.0158 ± 0.0137 |
| post-hoc: BERTweet (fine-tuned) + CLIP B/32 (frozen) | bertweet_text_matched + t2_img_clip_b32 | 0.5709 / 0.6063 | -0.0415 ± 0.0034 / -0.0375 ± 0.0158 | +0.0030 ± 0.0073 / +0.0049 ± 0.0119 | -0.0278 ± 0.0011 / -0.0182 ± 0.0091 | +0.0034 ± 0.0033 / +0.0049 ± 0.0073 |
| L4a: CLIP B/32 + DistilBERT cross-attention | t2_distilbert_text + t2_img_clip_b32 | 0.5849 / 0.5984 | -0.0275 ± 0.0094 / -0.0453 ± 0.0138 | +0.0170 ± 0.0161 / -0.0029 ± 0.0088 | -0.0231 ± 0.0032 / -0.0160 ± 0.0091 | +0.0081 ± 0.0048 / +0.0072 ± 0.0070 |
