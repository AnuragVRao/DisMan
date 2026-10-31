"""Task 1 backend: fine-tuned BERTweet (artifacts/task1_bertweet, seed 42, the best DEV seed; see ablation/d4_t1_bertweet.py --save). Takes the RAW tweet text:
BERTweet's own tokenizer normalisation (@USER, HTTPURL, demojised emoji) replaces crisis_common.clean_tweet. Offline: loads from the local directory only."""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class T1Bertweet:
    def __init__(self, path, device, temperature=1.0):
        path = Path(path)
        self.meta = json.load(open(path / "t1_meta.json"))
        self.labels, self.max_len, self.device, self.T = self.meta["labels"], self.meta["max_len"], device, float(temperature)
        self.tok = AutoTokenizer.from_pretrained(path, use_fast=False, normalization=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()

    @torch.no_grad()
    def logits(self, text):
        e = self.tok([text or ""], truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            return self.model(**e).logits.float()[0].cpu().numpy()

    def predict(self, text):
        """-> (calibrated probs, unscaled probs) in self.labels order."""
        z = self.logits(text)
        sm = lambda v: np.exp(v - v.max()) / np.exp(v - v.max()).sum()
        return sm(z / self.T), sm(z)
