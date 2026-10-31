# Task 1 leave-one-disaster-out (macro-F1 held out / in-distribution; single seed 42 for BERTweet)

| Held-out event | T1 test rows | BERTweet | TF-IDF + LR |
|---|---|---|---|
| california_wildfires | 240 | 0.662 / 0.637 | 0.615 / 0.606 |
| hurricane_harvey | 568 | 0.809 / 0.824 | 0.735 / 0.758 |
| hurricane_irma | 548 | 0.743 / 0.738 | 0.675 / 0.697 |
| hurricane_maria | 542 | 0.793 / 0.808 | 0.692 / 0.742 |
| iraq_iran_earthquake | 73 | 0.665 / 0.716 | 0.622 / 0.625 |
| mexico_earthquake | 159 | 0.665 / 0.749 | 0.628 / 0.638 |
| srilanka_floods | 107 | 0.978 / 0.955 | 0.793 / 0.932 |
| Average over events | 2237 | 0.759 / 0.775 | 0.680 / 0.714 |
