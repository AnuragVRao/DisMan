"""C1 missing-modality robustness, no retraining (ABLATION_PLAN.md, Part C).

At test time only: (a) the image is replaced by a blank grey image (RGB 128,128,128, 224x224: the same grey the pipeline already uses
for unreadable images) -> its CLIP embedding; (b) the text is replaced by an empty string (SBERT embeds '' as 'no text', v2's rule for
empty text; DistilBERT sees just [CLS][SEP]). Reports the drop from the full-input score.

  python c1_missing_modality.py t3      # CPU: v2 fused 5-NN, index = the 2,361-row tweet-disjoint v2 index
  python c1_missing_modality.py t2      # GPU (small): every available trained full-model checkpoint (L4a, L4; all seeds)

Writes results/ablation/C/c1_t3_seed42.json and results/ablation/C/c1_t2_<config>_seed<k>.json.
"""
import json
import os
import sys

which = sys.argv[1] if len(sys.argv) > 1 else "t3"
if which == "t3":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import faiss
import numpy as np
import torch
from PIL import Image

from common import ABL_ART, ART, DATA, SPLIT_DIR, cc, write_row

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GREY = Image.new("RGB", (224, 224), (128, 128, 128))


def no_miss():
    raise RuntimeError("cache miss")


def grey_clip_vector():
    clip, proc = cc.load_clip(DEVICE, ART / "clip")
    return cc.clip_image_features(clip, proc, [GREY], DEVICE)[0]            # raw 512-d, like the cached vectors


def t3():
    S = {s: cc.load_split("damage", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "test")}
    img = {s: cc.cached(ART / "emb" / f"clip_damage_{s}.npy", S[s].image_id.tolist(), no_miss) for s in S}
    txt = {s: cc.cached(ART / "emb" / f"sbert_damage_{s}.npy", S[s].image_id.tolist(), no_miss) for s in S}
    tr, te = S["train"], S["test"]
    keep = ~tr.tweet_id.isin(set(te.tweet_id)).to_numpy()
    assert int(keep.sum()) == 2361
    index = faiss.IndexFlatIP(896); index.add(cc.t3_vectors(img["train"], txt["train"])[keep])
    labs = tr.label.to_numpy()[keep]
    grey = np.tile(grey_clip_vector(), (len(te), 1))
    empty = np.tile(cc.embed_texts([""], cc.load_sbert(DEVICE, ART / "sbert")), (len(te), 1))
    out = {}
    for name, (i, t) in {"full": (img["test"], txt["test"]), "image_grey": (grey, txt["test"]),
                         "text_empty": (img["test"], empty)}.items():
        sims, nn_ = index.search(cc.t3_vectors(i, t), 5)
        pred = [cc.knn_vote(list(labs[r]), list(s))[0] for r, s in zip(nn_, sims)]
        out[name] = cc.metrics(te.label.to_numpy(), pred, cc.T3_LABELS)
        print(f"T3 {name:10s} acc {out[name]['acc']:.4f} macroF1 {out[name]['macro_f1']:.4f} wF1 {out[name]['weighted_f1']:.4f}")
    assert abs(out["full"]["macro_f1"] - 0.4914) < 5e-5
    row = dict(task="T3", tier="cpu", n_test=len(te), **{f"{k}_{m}": float(v) for k, d in out.items() for m, v in d.items() if m != "n"},
               **{f"drop_{k}_macro_f1": out["full"]["macro_f1"] - out[k]["macro_f1"] for k in ("image_grey", "text_empty")},
               **{f"drop_{k}_acc": out["full"]["acc"] - out[k]["acc"] for k in ("image_grey", "text_empty")})
    write_row("C", "c1_t3", 42, row)


def t2():
    from transformers import DistilBertConfig, DistilBertTokenizerFast
    sys.path.insert(0, str(ABL_ART.parent.parent / "ablation"))
    from t2_train import PRESETS
    S = {s: cc.load_split("humanitarian", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "test")}
    labels = sorted(S["train"].label.unique()); l2i = {l: i for i, l in enumerate(labels)}
    tok = DistilBertTokenizerFast.from_pretrained(ART / "tokenizer"); bcfg = DistilBertConfig.from_pretrained(ART / "distilbert_config")
    te = S["test"]; y = np.array([l2i[l] for l in te.label])
    vec = cc.cached(ART / "emb" / "clip_humanitarian_test.npy", te.image_id.tolist(), no_miss)
    grey = np.tile(grey_clip_vector(), (len(te), 1))
    ids, mask = cc.tokenize(tok, te.text.tolist(), 128)
    ids0, mask0 = cc.tokenize(tok, [""] * len(te), 128)
    ALL = list(range(len(labels)))
    for ck in sorted((ABL_ART / "ckpt").glob("t2_L4*_seed*.pt")):
        name, seed = ck.stem.rsplit("_seed", 1)
        cfgname = name.removeprefix("t2_")
        if cfgname not in ("L4", "L4a"):
            continue
        model = cc.DualStreamClassifier(len(labels), bert_config=bcfg).to(DEVICE)       # structure only; weights come from the checkpoint
        model.load_state_dict(torch.load(ck, map_location=DEVICE))
        out = {}
        for nm, (v, i_, m_) in {"full": (vec, ids, mask), "image_grey": (grey, ids, mask), "text_empty": (vec, ids0, mask0)}.items():
            probs = cc.t2_predict_proba(model, v, i_, m_, DEVICE)
            out[nm] = cc.metrics(y, probs.argmax(1), ALL)
        print(f"T2 {cfgname} seed {seed}: " + " | ".join(f"{k} F1 {v['macro_f1']:.4f}" for k, v in out.items()), flush=True)
        row = dict(task="T2", tier="gpu-small", checkpoint=ck.name, n_test=len(te),
                   **{f"{k}_{m}": float(v) for k, d in out.items() for m, v in d.items() if m != "n"},
                   **{f"drop_{k}_macro_f1": out["full"]["macro_f1"] - out[k]["macro_f1"] for k in ("image_grey", "text_empty")},
                   **{f"drop_{k}_acc": out["full"]["acc"] - out[k]["acc"] for k in ("image_grey", "text_empty")})
        write_row("C", f"c1_t2_{cfgname}", int(seed), row)


{"t3": t3, "t2": t2}[which]()
