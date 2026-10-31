"""Day 4: Task 2 (humanitarian, 8 classes) data for every Task 2 model. Image-label task: `label_image`. Row order = crisis_common.load_split (the order of the cached features).
Mirror-image of d4_t1_data: no Task 1 trained module may be loaded in the process (SELECTION_RULE_DAY4.md rule 11)."""
import sys

import numpy as np
import pandas as pd

from common import DATA, SPLIT_DIR, cc

FORBIDDEN = ("d4_t1_data", "d4_t1_bertweet", "d4_t1_early_fusion", "d4_t1_heads", "d4_t1_bilstm", "d4_t1_classical", "t1_bertweet")   # Task 1 trained modules / loaders
_D = {}


def data():
    assert not [m for m in FORBIDDEN if m in sys.modules], "a Task 1 module is loaded in a Task 2 process"
    if not _D:
        assert cc.LABEL_COL["humanitarian"] == "label_image", "Task 2 must use the image label"
        S = {s: cc.load_split("humanitarian", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "dev", "test")}
        labels = sorted(S["train"].label.unique()); l2i = {l: i for i, l in enumerate(labels)}
        for s, df in S.items():
            assert set(df.label.unique()) <= set(labels), s
            raw = pd.read_csv(SPLIT_DIR / f"task_humanitarian_text_img_{s}.tsv", sep="\t", dtype=str).drop_duplicates(subset=["image_id"]).set_index("image_id")
            assert (raw.loc[df.image_id, "label_image"].map(cc.t2_normalise).values == df.label.values).all(), "label is not the raw label_image column"
        assert not (set(S["train"].tweet_id) & (set(S["dev"].tweet_id) | set(S["test"].tweet_id))), "T2 train shares tweets with dev/test"
        _D.update(S=S, labels=labels, y={s: np.array([l2i[l] for l in S[s].label]) for s in S}, ids={s: S[s].image_id.tolist() for s in S})
    return _D
