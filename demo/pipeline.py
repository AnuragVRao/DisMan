"""Cascade inference over artifacts/ (offline). `Pipeline().predict(image, text) -> dict`."""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import sys
import time
from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import DistilBertConfig, DistilBertTokenizerFast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import crisis_common as cc  # noqa: E402

ART = ROOT / "artifacts"
DATA = ROOT / "data" / "CrisisMMD_v2.0"
# Task 2 weights: v2's checkpoint by default; set CAPSTONE_T2_CKPT to point the demo at another state dict of the same architecture
T2_CKPT = Path(os.environ.get("CAPSTONE_T2_CKPT", ART / "task2_model.pt"))


class Pipeline:
    def __init__(self, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cfg = json.load(open(ART / "config.json"))
        self.t1 = joblib.load(ART / "task1_pipeline.joblib")
        self.clip, self.proc = cc.load_clip(self.device, ART / "clip")
        self.sbert = cc.load_sbert(self.device, ART / "sbert")
        self.tok = DistilBertTokenizerFast.from_pretrained(ART / "tokenizer")
        self.t2_labels = self.cfg["t2_labels"]
        self.t2 = cc.DualStreamClassifier(len(self.t2_labels),
                                          bert_config=DistilBertConfig.from_pretrained(ART / "distilbert_config"))
        self.t2.load_state_dict(torch.load(T2_CKPT, map_location="cpu"))
        self.t2.to(self.device).eval()
        self.index = faiss.read_index(str(ART / "task3_index.faiss"))
        self.meta = pd.read_csv(ART / "task3_meta.csv", dtype=str, keep_default_na=False)
        self.k = self.cfg["t3_k"]

    def predict(self, image, text):
        """image: PIL.Image or path; text: tweet string. Returns a dict with t1, t2, t3, neighbours, latency_s."""
        t0 = time.time()
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        image = image.convert("RGB")
        text = text or ""
        clean = cc.clean_tweet(text)

        # Task 1 (text only; a badge, never a gate)
        p1 = self.t1.predict_proba([clean])[0]
        cls = list(self.t1.classes_)
        t1 = dict(label=cls[int(p1.argmax())], prob=float(p1.max()), probs=dict(zip(cls, map(float, p1))))

        # shared embeddings
        img_raw = cc.clip_image_features(self.clip, self.proc, [image], self.device)
        txt_emb = cc.embed_texts([clean], self.sbert)

        # Task 2
        ids, mask = cc.tokenize(self.tok, [clean], self.cfg["t2_max_len"])
        p2 = cc.t2_predict_proba(self.t2, img_raw, ids, mask, self.device)[0]
        t2 = dict(label=self.t2_labels[int(p2.argmax())], probs=dict(zip(self.t2_labels, map(float, p2))))

        # Task 3: kNN over the train-only index
        q = cc.t3_vectors(img_raw, txt_emb)
        scores, idx = self.index.search(q, self.k)
        rows = self.meta.iloc[idx[0]]
        label, share = cc.knn_vote(list(rows["label"]), list(scores[0]))
        t3 = dict(label=label, vote_share=share, note="agreement among similar past cases (k=%d vote share); not a calibrated probability" % self.k)
        neighbours = [dict(image_path=str(DATA / r.rel_image), tweet=r.tweet_text, label=r.label,
                           event=r.event_name, score=float(s))
                      for r, s in zip(rows.itertuples(), scores[0])]

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return dict(t1=t1, t2=t2, t3=t3, neighbours=neighbours, latency_s=time.time() - t0)
