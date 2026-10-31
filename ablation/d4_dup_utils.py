"""Shared helpers for the final-round duplicate checks and the error analysis (SELECTION_RULE_DAY4.md rules 21-22). No data loaders are imported here."""
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

TEXT_THR, IMG_THR = 0.9, 0.98          # fixed in rule 21 before any run


def norm_text(t):
    """lowercase; remove URLs; remove @mentions (with a trailing colon); remove leading 'RT'; drop the '#' of hashtags; collapse whitespace."""
    t = str(t).lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+:?", " ", t)
    t = re.sub(r"^\s*(rt\s+)+", "", t)
    t = t.replace("#", "")
    return re.sub(r"\s+", " ", t).strip()


def tfidf_sims(train_texts, test_texts):
    """Cosine similarity matrix (test x train, float32) of normalised texts; vectoriser fitted on the normalised train text (word 1-2-grams, sublinear tf)."""
    v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    Xtr = v.fit_transform(train_texts); Xte = v.transform(test_texts)
    return (Xte @ Xtr.T).toarray().astype("float32")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5s(paths, workers=8):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(md5, paths))


def n_words(t):
    return len(norm_text(t).split())
