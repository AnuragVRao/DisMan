# Task 1 calibration (BERTweet; temperature fitted on dev; ECE with 15 equal-width bins)

| Model | split | Acc. | Mean conf. | ECE | T | ECE after |
|---|---|---|---|---|---|---|
| BERTweet, seed 42 (original run) | dev | 0.8310 | 0.8773 | 0.0499 | 1.283 | 0.0234 |
| BERTweet, seed 42 (original run) | test | 0.8350 | 0.8789 | 0.0446 | 1.283 | 0.0262 |
| BERTweet, seed 43 | dev | 0.8342 | 0.8786 | 0.0527 | 1.194 | 0.0413 |
| BERTweet, seed 43 | test | 0.8458 | 0.8806 | 0.0393 | 1.194 | 0.0324 |
| BERTweet, seed 44 | dev | 0.8315 | 0.8854 | 0.0572 | 1.343 | 0.0252 |
| BERTweet, seed 44 | test | 0.8395 | 0.8885 | 0.0492 | 1.343 | 0.0223 |
| BERTweet, deployed checkpoint (retrain of seed 42) | dev | 0.8328 | 0.9322 | 0.1022 | 1.825 | 0.0418 |
| BERTweet, deployed checkpoint (retrain of seed 42) | test | 0.8480 | 0.9360 | 0.0882 | 1.825 | 0.0443 |
