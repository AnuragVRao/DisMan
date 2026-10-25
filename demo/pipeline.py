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
# Task 2 weights. Default: the ablation study's L4a seed-42 model (pretrained DistilBERT, same architecture and recipe as v2;
# test acc 0.8033 / macro-F1 0.6080, see docs/ablation_findings.md). v2's own checkpoint (random-init DistilBERT, 0.7975 / 0.5865)
# stays selectable: CAPSTONE_T2_CKPT=<repo>/artifacts/task2_model.pt. Any state dict of the same architecture works.
T2_CKPT_L4A = ART / "ablation" / "ckpt" / "t2_L4a_seed42.pt"
T2_CKPT_V2 = ART / "task2_model.pt"
if os.environ.get("CAPSTONE_T2_CKPT"):
    T2_CKPT = Path(os.environ["CAPSTONE_T2_CKPT"])
elif T2_CKPT_L4A.exists():
    T2_CKPT = T2_CKPT_L4A
else:
    print(f"[demo] {T2_CKPT_L4A} not found; falling back to v2's checkpoint {T2_CKPT_V2.name}", file=sys.stderr)
    T2_CKPT = T2_CKPT_V2


class Pipeline:
    def __init__(self, device=None, hybrid_t3=None):
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
        self.t2_ckpt = T2_CKPT.name
        # Temperatures for the displayed confidences (fitted on dev, see demo/calibration.json). They never change a predicted label.
        self.calib = json.load(open(Path(__file__).resolve().parent / "calibration.json"))
        self.t2_T = float(self.calib["t2"]["temperature"]) if self.t2_ckpt == self.calib["t2"]["checkpoint"] else 1.0   # only valid for that checkpoint
        self.t2.to(self.device).eval()
        self.index = faiss.read_index(str(ART / "task3_index.faiss"))
        self.meta = pd.read_csv(ART / "task3_meta.csv", dtype=str, keep_default_na=False)
        self.k = self.cfg["t3_k"]
        # Hybrid Task 3 (ablation E2): a logistic probe predicts, the k-NN search supplies the evidence. Default ON;
        # CAPSTONE_T3_HYBRID=0 (or Pipeline(hybrid_t3=False)) restores v2's 5-NN vote.
        if hybrid_t3 is None:
            hybrid_t3 = os.environ.get("CAPSTONE_T3_HYBRID", "1") != "0"
        self.hybrid = None
        if hybrid_t3:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from hybrid import HybridT3
            self.hybrid = HybridT3(self.index, self.meta["label"], temperature=self.calib["t3_probe"]["temperature"])

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
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=(self.device.type == "cuda")):
            logits = self.t2(torch.as_tensor(img_raw).float().to(self.device), ids.to(self.device), mask.to(self.device)).float()
        p2_raw = torch.softmax(logits, -1)[0].cpu().numpy()                # what t2_predict_proba returned before calibration
        p2 = torch.softmax(logits / self.t2_T, -1)[0].cpu().numpy()        # displayed: temperature-scaled
        t2 = dict(label=self.t2_labels[int(p2.argmax())], probs=dict(zip(self.t2_labels, map(float, p2))),
                  temperature=self.t2_T, label_unscaled=self.t2_labels[int(p2_raw.argmax())])

        # Task 3: kNN over the train-only index
        q = cc.t3_vectors(img_raw, txt_emb)
        scores, idx = self.index.search(q, self.k)
        rows = self.meta.iloc[idx[0]]
        label, share = cc.knn_vote(list(rows["label"]), list(scores[0]))
        t3 = dict(label=label, vote_share=share, note="agreement among similar past cases (k=%d vote share); not a calibrated probability" % self.k)
        if self.hybrid is not None:
            h = self.hybrid.predict(q, rows["label"], scores[0])
            t3 = dict(label=h["label"], vote_share=h["vote_share"], probe_probs=h["probs"], probe_confidence=h["confidence"],
                      probe_probs_unscaled=h["probs_unscaled"], temperature=self.hybrid.temperature, knn_label=h["knn_label"],
                      neighbours_agree=h["agrees"],
                      note="hybrid: a logistic probe predicts the label; the k=%d retrieved cases are the evidence (their majority is "
                           "shown for comparison); probe confidence is temperature-scaled on the dev split" % self.k)
        neighbours = [dict(image_path=str(DATA / r.rel_image), tweet=r.tweet_text, label=r.label,
                           event=r.event_name, score=float(s))
                      for r, s in zip(rows.itertuples(), scores[0])]

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return dict(t1=t1, t2=t2, t3=t3, neighbours=neighbours, latency_s=time.time() - t0)
