"""Day 4: Task 1 data for every model. Text label only (SELECTION_RULE_DAY4.md rule 1), row order = crisis_common.load_split (the order of the cached features)."""
import numpy as np
import pandas as pd

from common import DATA, SPLIT_DIR, cc

LABELS = cc.T1_LABELS
_D = {}


def data():
    if not _D:
        assert cc.LABEL_COL["informative"] == "label_text", "Task 1 must use the text label"
        S = {s: cc.load_split("informative", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "dev", "test")}
        for s, df in S.items():
            assert set(df.label.unique()) <= set(LABELS), (s, df.label.unique())
            raw = pd.read_csv(SPLIT_DIR / f"task_informative_text_img_{s}.tsv", sep="\t", dtype=str).drop_duplicates(subset=["image_id"]).set_index("image_id")
            assert (raw.loc[df.image_id, "label_text"].str.strip().values == df.label.values).all(), "label is not the raw label_text column"
        _D.update(S=S, y={s: np.array([LABELS.index(l) for l in S[s].label]) for s in S}, ids={s: S[s].image_id.tolist() for s in S})
    return _D
