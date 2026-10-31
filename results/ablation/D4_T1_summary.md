# Task 1 summary (text label; sorted by dev weighted F1; mean ± std over seeds, deterministic models have 1 seed)

| Model | seeds | dev wF1 | dev bal-acc | test acc | test wF1 | test macro-F1 | test bal-acc | test MCC | test ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| BERTweet fine-tuned (weighted CE) | 3 | 0.8304 ± 0.0015 | 0.7827 ± 0.0143 | 0.8401 ± 0.0054 | 0.8382 ± 0.0022 | 0.7970 ± 0.0009 | 0.7913 ± 0.0110 | 0.5968 ± 0.0027 | 0.8972 ± 0.0014 |
| BERTweet fine-tuned, unweighted CE (post-hoc) | 3 | 0.8298 ± 0.0039 | 0.7652 ± 0.0138 | 0.8473 ± 0.0030 | 0.8414 ± 0.0048 | 0.7967 ± 0.0079 | 0.7792 ± 0.0121 | 0.6027 ± 0.0105 | 0.8947 ± 0.0024 |
| BiLSTM + GloVe Twitter 200d | 3 | 0.8114 ± 0.0034 | 0.7680 ± 0.0112 | 0.8103 ± 0.0130 | 0.8109 ± 0.0098 | 0.7662 ± 0.0083 | 0.7681 ± 0.0028 | 0.5338 ± 0.0161 | 0.8580 ± 0.0072 |
| TF-IDF + LR (balanced) | 1 | 0.8109 | 0.7730 | 0.7944 | 0.7970 | 0.7513 | 0.7584 | 0.5039 | 0.8407 |
| TF-IDF + NB, balanced sample weights (post-hoc) | 1 | 0.7987 | 0.7547 | 0.7872 | 0.7875 | 0.7366 | 0.7372 | 0.4731 | 0.8386 |
| TF-IDF + Random Forest (balanced) | 1 | 0.7983 | 0.7166 | 0.8176 | 0.8078 | 0.7510 | 0.7319 | 0.5170 | 0.8347 |
| TF-IDF + Linear SVM (CV-calibrated) | 1 | 0.7940 | 0.7107 | 0.8020 | 0.7909 | 0.7288 | 0.7112 | 0.4729 | 0.8254 |
| TF-IDF + ComplementNB (post-hoc) | 1 | 0.7856 | 0.6927 | 0.8078 | 0.7891 | 0.7199 | 0.6962 | 0.4796 | 0.8400 |
| CLIP L/14 head (frozen) [image-only] | 3 | 0.7629 ± 0.0056 | 0.7216 ± 0.0079 | 0.7546 ± 0.0066 | 0.7608 ± 0.0056 | 0.7119 ± 0.0054 | 0.7262 ± 0.0040 | 0.4299 ± 0.0092 | 0.7967 ± 0.0004 |
| DINOv2 head (frozen) [image-only] | 3 | 0.7189 ± 0.0050 | 0.6836 ± 0.0052 | 0.7069 ± 0.0072 | 0.7181 ± 0.0063 | 0.6680 ± 0.0058 | 0.6902 ± 0.0039 | 0.3520 ± 0.0089 | 0.7433 ± 0.0016 |
| ResNet50 head (frozen) [image-only] | 3 | 0.7106 ± 0.0047 | 0.6596 ± 0.0104 | 0.6923 ± 0.0126 | 0.7016 ± 0.0092 | 0.6436 ± 0.0057 | 0.6575 ± 0.0043 | 0.2965 ± 0.0070 | 0.7210 ± 0.0031 |
| TF-IDF + NB, unweighted (reference) | 1 | 0.6979 | 0.5813 | 0.7671 | 0.7076 | 0.5875 | 0.5910 | 0.3419 | 0.8400 |
| majority class (always informative) (post-hoc) | 1 | 0.6036 | 0.5000 | 0.7206 | 0.6036 | 0.4188 | 0.5000 | 0.0000 | 0.5000 |

## Notes

- **BiLSTM + GloVe vs TF-IDF + LR: tied / not resolvable.** Dev weighted F1 0.8114 (BiLSTM, mean of 3 seeds) vs 0.8109 (TF-IDF+LR), so the BiLSTM is nominally *above* by +0.0005; the paired dev bootstrap CI is [-0.0129, +0.0133], which includes 0. Post-hoc comparison, no multiplicity correction.
- **Loss weighting for BERTweet (post-hoc arm; weighting is retained because it was fixed by pre-registration).** Weighting raises not-informative recall at the cost of informative recall (and lowers not-informative precision). Mean of 3 seeds, dev / test: not-informative recall 0.670 / 0.681 (weighted) vs 0.604 / 0.625 (unweighted); informative recall 0.895 / 0.902 vs 0.926 / 0.933; not-informative precision 0.714 / 0.732 vs 0.763 / 0.785. Dev balanced accuracy gain +0.0175, 95% CI [+0.0095, +0.0258]. Weighted F1 and macro-F1 were not tested with a bootstrap; the arm means differ by less than the seed spread (see the table). This is a different operating point (more not-informative predictions), not a safety improvement; whether it is preferable depends on the cost of missed informative tweets.
