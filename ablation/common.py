"""Shared helpers for the ablation study (see ABLATION_PLAN.md). Nothing here changes v2 behaviour."""
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import crisis_common as cc  # noqa: E402

DATA = ROOT / "data" / "CrisisMMD_v2.0"
SPLIT_DIR = DATA / "crisismmd_datasplit_all" / "crisismmd_datasplit_all"
ART = ROOT / "artifacts"
ABL_ART = ART / "ablation"            # new checkpoints / caches only; v2 artifacts are never overwritten
ABL_EMB = ABL_ART / "emb"
RES = ROOT / "results" / "ablation"

DONT_KNOW = "Don't know or can't judge"
T3_LABELS_V1 = cc.T3_LABELS + [DONT_KNOW]


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def write_row(part, config, seed, row):
    """One JSON row per run: results/ablation/<part>/<config>_seed<k>.json (never overwrites silently)."""
    out = RES / part / f"{config}_seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    row = dict(row, part=part, config=config, seed=seed, git_commit=git_commit())
    out.write_text(json.dumps(row, indent=1))
    return out


# ── v1 Task 3 annotation table (the 18,082 rows the v1 FAISS index was built from) ──
def load_annotations():
    """Replicates capstone-final.ipynb cell 23 (t3_load_annotations): every annotation TSV, no split filter.

    v1 label = image_damage with NaN -> 'dont_know_or_cant_judge' -> "Don't know or can't judge".
    Adds split membership (damage train/dev/test by image_id) and the NaN-vs-explicit origin of "Don't know".
    """
    raw = pd.concat([pd.read_csv(f, sep="\t") for f in sorted(glob.glob(str(DATA / "annotations" / "*.tsv")))],
                    ignore_index=True)
    df = pd.DataFrame({
        "event_name": raw["image_path"].str.split("/").str[1], "tweet_id": raw["tweet_id"].astype(str), "image_id": raw["image_id"].astype(str),
        "image_path": raw["image_path"].map(lambda p: str(DATA / p)),
        "tweet_text": raw["tweet_text"].fillna(""),
        "damage_raw": raw["image_damage"],
    })
    df["label_v1"] = raw["image_damage"].fillna("dont_know_or_cant_judge").map(cc.t3_normalise)
    df["dk_origin"] = np.where(raw["image_damage"].isna(), "nan", np.where(df["label_v1"] == DONT_KNOW, "explicit", ""))
    df["text"] = df["tweet_text"].map(cc.clean_tweet)
    split = {}
    for s in ("train", "dev", "test"):
        d = pd.read_csv(SPLIT_DIR / f"task_damage_text_img_{s}.tsv", sep="\t", dtype=str)
        split.update({i: s for i in d["image_id"]})
    df["split"] = df["image_id"].map(split).fillna("none")
    return df


def timer():
    t0 = time.time()
    return lambda: time.time() - t0
