"""Day 4, Task 1 classical text models on the same TF-IDF as the deployed pipeline: LR (reference, reproduces 0.7513 macro-F1), NB, RF, Linear SVM.
Deterministic (RF and LR carry seed 42). Saves dev and test probabilities per run (SVM: CV-calibrated on train, dev untouched)."""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import d4_t1_data as d4
import d4_prov
import metrics_all as M
from common import cc

SEED = cc.SEED
ARMS = {
    "tfidf_lr": lambda: LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=SEED),
    "tfidf_nb": lambda: MultinomialNB(),
    "tfidf_rf": lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=SEED),
    "tfidf_svm": lambda: CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=SEED), cv=5),
}

if __name__ == "__main__":
    D = d4.data(); S = D["S"]
    for name, mk in ARMS.items():
        pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)), ("clf", mk())])
        pipe.fit(S["train"].text, D["y"]["train"])
        pd_, pt_ = pipe.predict_proba(S["dev"].text), pipe.predict_proba(S["test"].text)
        row = M.save_run("T1", name, SEED, d4.LABELS, D["y"]["dev"], pd_, D["y"]["test"], pt_, extra=d4_prov.prov("informative", "none (TF-IDF + %s fit on T1 train)" % name))
        t, dv = row["test"], row["dev"]
        print(f"{name:10s} dev wF1 {dv['weighted_f1']:.4f} | test acc {t['acc']:.4f} wF1 {t['weighted_f1']:.4f} macroF1 {t['macro_f1']:.4f} "
              f"MCC {t['mcc']:.4f} AUC {t['roc_auc']:.4f} ECE {t['ece']:.4f}", flush=True)
