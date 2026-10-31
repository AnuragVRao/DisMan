"""Day 4 step 5a, Task 2 classical text models on the 8 humanitarian classes (`label_image`): TF-IDF + LR (class-weighted), LinearSVC (class-weighted, probabilities from
CalibratedClassifierCV fitted with CV on TRAIN, never on dev), ComplementNB. Same vectoriser as Task 1 (50k features, 1-2 grams, sublinear tf, min_df 2) on the cleaned text. Deterministic, seed 42.
Saves dev and test probabilities. CPU only."""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

import d4_prov
import d4_t2_data as d4
import metrics_all as M
from common import cc

SEED = cc.SEED
ARMS = {"t2_tfidf_lr": lambda: LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED),
        "t2_tfidf_svm": lambda: CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=SEED), cv=5),
        "t2_tfidf_cnb": lambda: ComplementNB(),                      # unweighted (reference)
        "t2_tfidf_cnb_bal": lambda: ComplementNB()}                  # balanced sample weights (post-hoc, added after the first Day 4 Task 2 results)

if __name__ == "__main__":
    D = d4.data(); S, y, labels = D["S"], D["y"], D["labels"]
    print("train class counts:", dict(zip(labels, np.bincount(y["train"]))))
    for name, mk in ARMS.items():
        pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)), ("clf", mk())])
        kw = {"clf__sample_weight": compute_sample_weight("balanced", y["train"])} if name.endswith("_bal") else {}
        pipe.fit(S["train"].text, y["train"], **kw)
        pd_, pt_ = pipe.predict_proba(S["dev"].text), pipe.predict_proba(S["test"].text)
        row = M.save_run("T2", name, SEED, labels, y["dev"], pd_, y["test"], pt_, extra=dict(posthoc=name.endswith("_bal"), **d4_prov.prov("humanitarian", f"none (TF-IDF + {name[8:]} fit on T2 train)")))
        t, dv = row["test"], row["dev"]
        print(f"{name:14s} dev macro-F1 {dv['macro_f1']:.4f} | test acc {t['acc']:.4f} macro-F1 {t['macro_f1']:.4f} wF1 {t['weighted_f1']:.4f} bal-acc {t['balanced_acc']:.4f} top-2 {t['top2_acc']:.4f} MCC {t['mcc']:.4f}", flush=True)
    # majority-class baseline ("always" the train majority class)
    maj = int(np.bincount(y["train"]).argmax()); p = lambda n: np.tile(np.eye(len(labels))[maj], (n, 1))
    row = M.save_run("T2", "t2_majority", SEED, labels, y["dev"], p(len(y["dev"])), y["test"], p(len(y["test"])),
                     extra=dict(posthoc=True, class_predicted=labels[maj], **d4_prov.prov("humanitarian", "none (majority class of T2 train)")))
    print("majority:", labels[maj], {sp: {k: round(row[sp][k], 4) for k in ("acc", "weighted_f1", "macro_f1", "balanced_acc", "mcc")} for sp in ("dev", "test")})
