# Day 5: preprocessing A/B on Task 1 BERTweet (test split, mean ± std over seeds)

Recipe for every arm: vinai/bertweet-base, max_len 128, lr 2e-5, 4 epochs, batch 32, weighted CE, epoch chosen on dev weighted F1, test scored once. Only the input text differs.

| Arm | Seeds | Weighted F1 | Macro-F1 | Accuracy | Dev weighted F1 |
|---|---|---|---|---|---|
| Baseline: raw tweet_text, BERTweet normalization on | 8 | 0.8413 ± 0.0042 | 0.8008 ± 0.0058 | 0.8433 ± 0.0047 | 0.8322 ± 0.0030 |
| A: preprocessing/ branch text, normalization on | 3 | 0.8378 ± 0.0086 | 0.7960 ± 0.0122 | 0.8401 ± 0.0081 | 0.8336 ± 0.0011 |
| B: preprocessing/ branch text, normalization off | 3 | 0.8364 ± 0.0062 | 0.7951 ± 0.0086 | 0.8377 ± 0.0054 | 0.8306 ± 0.0061 |
| C: minimal cleaning (ftfy + html.unescape + truncated-tail drop), normalization on | 8 | 0.8423 ± 0.0058 | 0.8024 ± 0.0077 | 0.8439 ± 0.0059 | 0.8309 ± 0.0017 |

## Difference from the baseline on the same seeds

| Arm | Test Δ weighted F1 | Test Δ macro-F1 | Test Δ accuracy | Dev bootstrap Δ weighted F1 (mean [95% CI]) |
|---|---|---|---|---|
| A | -0.0004 | -0.0010 | +0.0000 | +0.0021 [-0.0059, +0.0099] |
| B | -0.0019 | -0.0019 | -0.0024 | -0.0024 [-0.0114, +0.0066] |
| C | +0.0010 | +0.0016 | +0.0006 | -0.0045 [-0.0112, +0.0014] |

Arm C paired over 8 seeds: Δ weighted F1 +0.0010, SE 0.0028, better on 5 seeds, worse on 3.

Pre-registered rule for 'helps': mean test weighted-F1 gain > 0.004 AND the dev bootstrap 95% CI excludes 0. No arm meets it.

## Test weighted F1 per seed

| Arm | s42 | s43 | s44 | s45 | s46 | s47 | s48 | s49 |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.8362 | 0.8406 | 0.8379 | 0.8439 | 0.8491 | 0.8432 | 0.8376 | 0.8423 |
| A | 0.8346 | 0.8313 | 0.8476 | | | | | |
| B | 0.8322 | 0.8435 | 0.8334 | | | | | |
| C | 0.8390 | 0.8527 | 0.8419 | 0.8379 | 0.8351 | 0.8415 | 0.8416 | 0.8489 |
