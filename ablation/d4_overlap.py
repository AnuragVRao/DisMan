"""Day 4 step 1: cross-task split overlap. Rows = Task 1 train/dev/test, columns = Task 2 train/dev/test and Task 3 (damage) train/dev/test, counted by tweet_id and by image path.
Within each task the splits are disjoint (official CrisisMMD splits); across tasks they are not. Writes results/ablation/D4_cross_task_overlap.json."""
import json

import d4_t1_data as d4
from common import DATA, RES, SPLIT_DIR, cc

SP = ("train", "dev", "test")


def main():
    S = {"T1": d4.data()["S"]}
    S["T2"] = {s: cc.load_split("humanitarian", s, DATA, SPLIT_DIR, verbose=False)[0] for s in SP}
    S["T3"] = {s: cc.load_split("damage", s, DATA, SPLIT_DIR, verbose=False)[0] for s in SP}
    out = dict(sizes={t: {s: len(S[t][s]) for s in SP} for t in S})
    for key, col in (("tweet_id", "tweet_id"), ("image_path", "rel_image")):
        sets = {t: {s: set(S[t][s][col].astype(str)) for s in SP} for t in S}
        out[f"by_{key}"] = {f"T1_{a}": {f"{t}_{b}": len(sets["T1"][a] & sets[t][b]) for t in ("T2", "T3") for b in SP} for a in SP}
        out[f"unique_{key}"] = {t: {s: len(sets[t][s]) for s in SP} for t in S}
        # within-task disjointness check
        out[f"within_task_overlap_{key}"] = {t: {f"{a}&{b}": len(sets[t][a] & sets[t][b]) for i, a in enumerate(SP) for b in SP[i + 1:]} for t in S}
    # exact image_id identity too (the cache key)
    ids = {t: {s: set(S[t][s].image_id) for s in SP} for t in S}
    out["by_image_id"] = {f"T1_{a}": {f"{t}_{b}": len(ids["T1"][a] & ids[t][b]) for t in ("T2", "T3") for b in SP} for a in SP}
    (RES / "D4_cross_task_overlap.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    o = main()
    for k in ("sizes", "by_tweet_id", "by_image_path", "by_image_id", "within_task_overlap_tweet_id"):
        print(k); print(json.dumps(o[k], indent=1))
