"""Day 4: Task 1 data for every model. Text label only (SELECTION_RULE_DAY4.md rule 1), row order = crisis_common.load_split (the order of the cached features)."""
import sys

import numpy as np
import pandas as pd

from common import DATA, SPLIT_DIR, cc

LABELS = cc.T1_LABELS
_D = {}


FORBIDDEN = ("d2_common", "t2_models", "d_lodo_t2", "t2_train", "g_imbalance_t2")      # modules that own Task 2 heads / checkpoints / logits


def data():
    """Task 1 rows. Guards (step 1 of the Day 4 plan, cross-task overlap): training rows come only from the official T1 train file, no Task 2-trained module is loaded
    in this process, and the T1 splits are tweet-disjoint within the task. Cross-task overlap exists (results/ablation/D4_cross_task_overlap.json), so Task 1 may never use
    weights, logits or heads trained on Task 2 data; frozen pretrained encoders are fine."""
    assert not [m for m in FORBIDDEN if m in sys.modules], "a Task 2-trained module is loaded in a Task 1 process"
    if not _D:
        assert cc.LABEL_COL["informative"] == "label_text", "Task 1 must use the text label"
        S = {s: cc.load_split("informative", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "dev", "test")}
        for s, df in S.items():
            assert set(df.label.unique()) <= set(LABELS), (s, df.label.unique())
            raw = pd.read_csv(SPLIT_DIR / f"task_informative_text_img_{s}.tsv", sep="\t", dtype=str).drop_duplicates(subset=["image_id"]).set_index("image_id")
            assert (raw.loc[df.image_id, "label_text"].str.strip().values == df.label.values).all(), "label is not the raw label_text column"
        raw_train = pd.read_csv(SPLIT_DIR / "task_informative_text_img_train.tsv", sep="	", dtype=str)
        assert set(S["train"].image_id) <= set(raw_train.image_id), "T1 training rows must come from the T1 train file only"
        assert not (set(S["train"].tweet_id) & (set(S["dev"].tweet_id) | set(S["test"].tweet_id))), "T1 train shares tweets with dev/test"
        _D.update(S=S, y={s: np.array([LABELS.index(l) for l in S[s].label]) for s in S}, ids={s: S[s].image_id.tolist() for s in S})
    return _D
