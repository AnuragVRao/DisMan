# Resume notes (written 2026-10-22, mid Block 1)

**Python:** always `C:\Users\anura\AppData\Local\Programs\Python\Python313\python.exe` (CUDA torch, gradio installed,
huggingface-hub pinned <1.0 for transformers 4.57.1). The default `python` is 3.14 CPU-only: do not use.

**Rules:** don't modify `capstone-final.ipynb`; dev only for selection; report acc / macro-F1 / weighted-F1; seed 42;
`num_workers=0`; never fabricate numbers; STOP after Block 1 and show the before/after table. Plan: `~/.claude/plans/task-capstone-progress-vivid-hanrahan.md`.
User amendments: Task 1 must reproduce acc 0.7944 / wF1 0.7970 (else stop and explain); record HF CLIP switch as a QuickGELU BUG FIX;
label T2 "after" as "corrected pipeline" listing all changes; never commit weights (`artifacts/` ignored; after Block 2 zip it to
`../capstone_artifacts_backup.zip`); tokenizer via save_pretrained; demo runs with HF_HUB_OFFLINE=1; demo T1/T2 disagreement note
explains text-label vs image-label cause; after choosing preloaded examples, run them and list which the model gets wrong;
Block 1 time-box ~2 h (started ~20:35 on 2026-10-22).

**Where things are**
- `crisis_common.py` shared code; `tools/build_notebook.py` generates `capstone-v2.ipynb`; `demo/pipeline.py`, `demo/app.py` written, untested.
- `artifacts/emb/*.npy` CLIP embeddings for humanitarian+damage splits are done (reused automatically, ids checked).
- Finding: 62 tweet_ids in both damage train and test (multi-image tweets). Headline T3 index excludes the 107 train rows
  sharing a tweet with test; secondary row = official split, image-disjoint only.

**To resume Block 1**
```
cd C:\Users\anura\Desktop\Capstone\DisMan
python tools/build_notebook.py            # (Python313)
set CAPSTONE_TRAIN=1 & python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 capstone-v2.ipynb
```
Takes ~10-15 min (embeddings cached; Task 2 retrains 10 epochs). If it succeeds: check results/before_after.csv, commit, show table, STOP.
Then Block 2 (verify TRAIN=False run, zip artifacts), Block 3 (demo), Block 4 (RESULTS.md, figures).
