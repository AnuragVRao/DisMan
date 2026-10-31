# Task 2 text-only models (image label; mean ± std over seeds, deterministic models have 1 seed)

| Model | dev macro-F1 | test macro-F1 | test wF1 | test acc | test bal-acc | test MCC | test top-2 | test ROC-AUC | test ECE |
|---|---|---|---|---|---|---|---|---|---|
| majority class ('Not humanitarian') (post-hoc) | 0.0818 | 0.0801 | 0.3023 | 0.4716 | 0.1250 | 0.0000 | 0.5110 | 0.5000 | 0.5284 |
| TF-IDF + LR (class-weighted) | 0.2700 | 0.2779 | 0.4754 | 0.4573 | 0.3570 | 0.2880 | 0.6866 | 0.7115 | 0.0741 |
| TF-IDF + LinearSVC (class-weighted, CV-calibrated on train) | 0.2272 | 0.2273 | 0.5028 | 0.5489 | 0.2228 | 0.2927 | 0.7555 | 0.7171 | 0.0351 |
| TF-IDF + ComplementNB, unweighted (reference) | 0.2385 | 0.2480 | 0.5218 | 0.5458 | 0.2458 | 0.3155 | 0.7363 | 0.7323 | 0.3269 |
| TF-IDF + ComplementNB, balanced sample weights (post-hoc) | 0.2165 | 0.2135 | 0.3637 | 0.3183 | 0.3226 | 0.1958 | 0.5525 | 0.6805 | 0.1226 |
| BERTweet text-only, DistilBERT-matched recipe (10 ep) [primary] | 0.3113 ± 0.0010 | 0.3076 ± 0.0177 | 0.5262 ± 0.0124 | 0.5083 ± 0.0131 | 0.3602 ± 0.0066 | 0.3435 ± 0.0138 | 0.7310 ± 0.0219 | 0.7914 ± 0.0134 | 0.1154 ± 0.0710 |
| BERTweet text-only, rule-8 recipe (4 ep, lr 2e-5) | 0.3083 ± 0.0025 | 0.3010 ± 0.0047 | 0.5067 ± 0.0046 | 0.4762 ± 0.0036 | 0.3903 ± 0.0053 | 0.3251 ± 0.0052 | 0.6844 ± 0.0052 | 0.7873 ± 0.0045 | 0.0807 ± 0.0173 |
| DistilBERT text-only (existing, Part B) | 0.2948 ± 0.0048 | 0.2937 ± 0.0140 | 0.4993 | 0.4861 | n/a | n/a | n/a | n/a | n/a |

Arm selection (rule 13): matched − rule8, dev macro-F1 +0.0030, 95% CI [-0.0097, +0.0159] -> tied; pick: bertweet_text_rule8 (tied on dev -> simpler, rule 13).

Class-weighted classical models, selected on dev macro-F1: t2_tfidf_lr 0.2700, t2_tfidf_svm 0.2272, t2_tfidf_cnb_bal 0.2165 -> best: t2_tfidf_lr. The unweighted ComplementNB (dev 0.2385) is kept as a reference row; weighting did not help it.

BERTweet (matched recipe) vs DistilBERT text-only, per-seed dev macro-F1: BERTweet 0.3108, 0.3125, 0.3106; DistilBERT 0.2984, 0.2967, 0.2893. Consistent at every seed on dev, but the paired dev interval marginally includes zero [-0.0004, +0.0334]: small, not robust. Detail of the paired bootstrap (post-hoc; DistilBERT predictions re-inferred from its checkpoints, d4_t2_reinfer.py; seed-averaged, 10,000 resamples), for macro-F1, dev +0.0165 [-0.0004, +0.0334] (not distinguishable) and test +0.0138 [-0.0071, +0.0353] (not distinguishable); for weighted F1, dev +0.0195 (better) and test +0.0269 (better). Confound: BERTweet reads the raw tweet text with its own normalisation, DistilBERT read the cleaned text, and the encoders differ in size, so the gap is not attributable to the encoder alone.

## Test per-class F1 (mean over seeds)

| Class | TF-IDF LR | BERTweet matched | BERTweet rule8 | DistilBERT |
|---|---|---|---|---|
| Affected individuals | 0.144 | 0.116 | 0.165 | 0.172 |
| Infrastructure and utility damage | 0.394 | 0.438 | 0.352 | 0.421 |
| Injured or dead people | 0.192 | 0.202 | 0.180 | 0.218 |
| Missing or found people | 0.000 | 0.000 | 0.000 | 0.000 |
| Not humanitarian | 0.580 | 0.645 | 0.634 | 0.615 |
| Other relevant information | 0.483 | 0.545 | 0.547 | 0.488 |
| Rescue, volunteering or donation effort | 0.369 | 0.384 | 0.396 | 0.368 |
| Vehicle damage | 0.061 | 0.130 | 0.134 | 0.067 |
