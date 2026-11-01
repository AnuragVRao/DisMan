# D5_prep: does the `preprocessing/` branch (or a lighter cleaning) help Task 1 BERTweet?

Result: no. See `summary.md` (tables) and `summary.json` (same numbers, plus per-seed values).

| File | What |
|---|---|
| `T1_bertweet_shek_norm_seed{42-44}.json` | Arm A: `preprocessing/` `clean_text_bertweet`, BERTweet normalization on |
| `T1_bertweet_shek_nonorm_seed{42-44}.json` | Arm B: same text, normalization off |
| `T1_bertweet_prepC_seed{42-49}.json` | Arm C: minimal cleaning (`ablation/d5_prep_c.py`), normalization on |
| `T1_bertweet_raw_seed{45-49}.json` | Baseline recipe re-run on extra seeds (seeds 42-44 of the baseline are `../D4/T1_bertweet_seed*.json`) |
| `task1_text_C_{train,dev,test}.tsv` | Arm C text, keyed by `image_id` in `crisis_common.load_split` order |

Reproduce (from `ablation/`, Python 3.13 CUDA interpreter, `PYTHONIOENCODING=utf-8`):
```
python d5_prep_c.py                              # writes task1_text_C_*.tsv
python d5_t1_bertweet_prep.py --norm 42 43 44    # arm A   (also --nonorm for B)
python d5_t1_bertweet_prep.py --c 42 43 44 45 46 47 48 49
python d5_t1_bertweet_prep.py --raw 45 46 47 48 49
python d5_summary.py                             # writes summary.json / summary.md
```
No D4 result file is read-modified or overwritten (the baseline seeds 42-44 are only read).
