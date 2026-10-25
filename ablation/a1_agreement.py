"""A1: Task 2 on text/image-label AGREED vs DISAGREED test rows (inference only, existing checkpoints, 3 seeds).

agreed    = normalised label_text == label_image (955 rows; equals the official `label_text_image == Positive`)
disagreed = they differ (1,282 rows; `Negative`)
Scores L4a, image_only and text_only (artifacts/ablation/ckpt) on each subset. Macro-F1 averages the classes present in the subset's
true labels or predictions (v2's per-subset convention); the fixed 8-class macro-F1 is stored as well. Nothing is trained or tuned.
Writes results/ablation/X/a1_<model>_seed<k>.json (new files; no existing result is modified).
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import DistilBertConfig, DistilBertTokenizerFast

from common import ABL_ART, ART, DATA, SPLIT_DIR, cc, write_row
from t2_models import Variant

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = {"L4a": ("t2_L4a", None), "image_only": ("t2_image_only", "image_only"), "text_only": ("t2_text_only", "text_only")}


def no_miss():
    raise RuntimeError("cache miss")


tr, te = (cc.load_split("humanitarian", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "test"))
raw = pd.read_csv(SPLIT_DIR / "task_humanitarian_text_img_test.tsv", sep="\t", dtype=str)
te = te.merge(raw[["image_id", "label_text", "label_text_image"]], on="image_id", how="left")
te["txt_label"] = te["label_text"].map(cc.t2_normalise)
assert (te["txt_label"] != "Unknown").all() and len(te) == 2237
agreed = (te["txt_label"] == te["label"]).to_numpy()
assert (agreed == (te["label_text_image"] == "Positive").to_numpy()).all(), "agreement differs from the official label_text_image flag"
SUBSETS = {"agreed": agreed, "disagreed": ~agreed, "all": np.ones(len(te), bool)}
print({k: int(v.sum()) for k, v in SUBSETS.items()}, flush=True)

labels = sorted(tr["label"].unique()); l2i = {l: i for i, l in enumerate(labels)}; ALL = list(range(len(labels)))
y = np.array([l2i[l] for l in te["label"]])
vec = cc.cached(ART / "emb" / "clip_humanitarian_test.npy", te.image_id.tolist(), no_miss)
tok = DistilBertTokenizerFast.from_pretrained(ART / "tokenizer"); bcfg = DistilBertConfig.from_pretrained(ART / "distilbert_config")
ids, mask = cc.tokenize(tok, te.text.tolist(), 128)
maj = int(tr["label"].map(l2i).value_counts().idxmax())


def score(yt, yp):
    present = sorted(set(yt) | set(yp))
    return dict(n=int(len(yt)), acc=float((yt == yp).mean()),
                macro_f1=float(f1_score(yt, yp, labels=present, average="macro", zero_division=0)),
                macro_f1_all8=float(f1_score(yt, yp, labels=ALL, average="macro", zero_division=0)),
                weighted_f1=float(f1_score(yt, yp, labels=ALL, average="weighted", zero_division=0)))


for name, (cfg_name, variant) in MODELS.items():
    for seed in (42, 43, 44):
        model = (cc.DualStreamClassifier(len(labels), bert_config=bcfg) if variant is None else Variant(variant, len(labels), bert_config=bcfg)).to(DEVICE)
        model.load_state_dict(torch.load(ABL_ART / "ckpt" / f"{cfg_name}_seed{seed}.pt", map_location=DEVICE))
        pred = cc.t2_predict_proba(model, vec, ids, mask, DEVICE).argmax(1)
        res = {s: score(y[m], pred[m]) for s, m in SUBSETS.items()}
        base = {s: score(y[m], np.full(int(m.sum()), maj)) for s, m in SUBSETS.items()}
        stored = __import__("json").load(open(ABL_ART.parent.parent / "results" / "ablation" / ("A" if variant is None else "B") / f"{cfg_name}_seed{seed}.json"))
        assert abs(res["all"]["acc"] - stored["test_acc"]) < 1e-4 and abs(res["all"]["macro_f1_all8"] - stored["test_macro_f1"]) < 1e-4, "full-set score != stored result"
        write_row("X", f"a1_{name}", seed, dict(task="T2", tier="gpu-small", checkpoint=f"{cfg_name}_seed{seed}.pt",
                                                 n_agreed=int(agreed.sum()), n_disagreed=int((~agreed).sum()), subsets=res, majority_baseline=base))
        print(f"{name:10s} seed {seed}: " + " | ".join(f"{s} acc {r['acc']:.3f} F1 {r['macro_f1']:.3f}" for s, r in res.items()), flush=True)
