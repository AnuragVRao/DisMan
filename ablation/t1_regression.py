"""Part A, T1 regression check: v1's Task 1 code (capstone-final.ipynb cells 10-13, verbatim logic) vs v2's pipeline
(crisis_common.load_split + the same TF-IDF/LR) must give identical test accuracy and weighted-F1. CPU only."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from common import ART, DATA, SPLIT_DIR, cc, write_row


def model(**kw):
    return Pipeline([("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
                     ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", **kw))])


# v1: cells 10-13
v1 = {s: pd.read_csv(SPLIT_DIR / f"task_informative_text_img_{s}.tsv", sep="\t").drop_duplicates(subset=["image_id"]) for s in ("train", "test")}
for d in v1.values():
    d["tweet_text"] = d["tweet_text"].fillna("").apply(cc.clean_tweet)
m1 = model().fit(v1["train"]["tweet_text"], v1["train"]["label_text"])
p1 = m1.predict(v1["test"]["tweet_text"]); y1 = v1["test"]["label_text"]
r1 = dict(acc=accuracy_score(y1, p1), weighted_f1=f1_score(y1, p1, average="weighted"), n=len(y1))

# v2: freshly trained with its loader, and the saved v2 artifact
v2 = {s: cc.load_split("informative", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "test")}
m2 = model(random_state=cc.SEED).fit(v2["train"].text, v2["train"].label)
p2 = m2.predict(v2["test"].text); y2 = v2["test"].label
r2 = dict(acc=accuracy_score(y2, p2), weighted_f1=f1_score(y2, p2, average="weighted"), n=len(y2))
p3 = joblib.load(ART / "task1_pipeline.joblib").predict(v2["test"].text)
r3 = dict(acc=accuracy_score(y2, p3), weighted_f1=f1_score(y2, p3, average="weighted"), n=len(y2))
print("v1 code      :", {k: round(v, 6) for k, v in r1.items()})
print("v2 retrained :", {k: round(v, 6) for k, v in r2.items()})
print("v2 artifact  :", {k: round(v, 6) for k, v in r3.items()})
assert r1["n"] == r2["n"] == 2237 and (p1 == p2).all() and (p1 == p3).all(), "T1 predictions differ"
assert r1["acc"] == r2["acc"] == r3["acc"] and r1["weighted_f1"] == r2["weighted_f1"] == r3["weighted_f1"]
print("OK: identical predictions, accuracy and weighted-F1")
write_row("A", "t1_regression", 42, dict(task="T1", tier="cpu", test_acc=r1["acc"], test_weighted_f1=r1["weighted_f1"],
                                         test_macro_f1=f1_score(y1, p1, average="macro"), n_test=r1["n"], identical_predictions=True,
                                         v1_acc=r1["acc"], v2_acc=r2["acc"], v1_weighted_f1=r1["weighted_f1"], v2_weighted_f1=r2["weighted_f1"]))
