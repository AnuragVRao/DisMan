# Task 1 duplicate checks (test rows against the Task 1 training split; rule 21)

**Text.** 1 of 2237 test tweets have the same normalised text as a training tweet (exact), 2 are exact or near-duplicates (TF-IDF cosine >= 0.9; 1 are near but not exact); 0 exact duplicates have a training copy with a different or mixed label; 1 flagged rows have agreeing text/image labels, 1 disagreeing.

| Model | n before -> after | weighted F1 / macro-F1 / acc before | after | change (wF1, macro-F1) |
|---|---|---|---|---|
| TF-IDF + LR | 2237 -> 2235 | 0.7970 / 0.7513 / 0.7944 | 0.7972 / 0.7516 / 0.7946 | +0.0002, +0.0003 |
| BERTweet (3 seeds) | 2237 -> 2235 | 0.8382 / 0.7970 / 0.8401 | 0.8384 / 0.7973 / 0.8403 | +0.0001, +0.0002 |
| BERTweet, deployed checkpoint | 2237 -> 2235 | 0.8443 / 0.8024 / 0.8480 | 0.8445 / 0.8028 / 0.8483 | +0.0003, +0.0004 |

**Images.** 57 of 2237 test images have a byte-identical file in the training split (0 with the same tweet id), 142 are exact or near-duplicates (CLIP B/32 cosine >= 0.98; 85 near but not byte-identical).

| Model | n before -> after | weighted F1 / macro-F1 / acc before | after | change (wF1, macro-F1) |
|---|---|---|---|---|
| ResNet50 head | 2237 -> 2095 | 0.7016 / 0.6436 / 0.6923 | 0.6994 / 0.6456 / 0.6907 | -0.0022, +0.0020 |
| DINOv2 head | 2237 -> 2095 | 0.7181 / 0.6680 / 0.7069 | 0.7153 / 0.6692 / 0.7044 | -0.0029, +0.0012 |
| CLIP L/14 head | 2237 -> 2095 | 0.7608 / 0.7119 / 0.7546 | 0.7595 / 0.7148 / 0.7532 | -0.0013, +0.0028 |
| BERTweet text-only (reference) | 2237 -> 2095 | 0.8382 / 0.7970 / 0.8401 | 0.8335 / 0.7943 / 0.8353 | -0.0047, -0.0027 |
| late fusion | 2237 -> 2095 | 0.8446 / 0.8064 / 0.8452 | 0.8398 / 0.8036 / 0.8403 | -0.0048, -0.0028 |
| core model BERTweet + ResNet50 | 2237 -> 2095 | 0.8445 / 0.8051 / 0.8459 | 0.8395 / 0.8020 / 0.8409 | -0.0050, -0.0031 |
| BERTweet + CLIP L/14 | 2237 -> 2095 | 0.8449 / 0.8048 / 0.8473 | 0.8398 / 0.8014 / 0.8422 | -0.0051, -0.0034 |
