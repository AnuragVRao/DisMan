"""Validation for the fine-tuned fusion addendum (SELECTION_RULE.md): t2_models.FusionModel built with DistilBERT must load the stored L4a-L14 `inv` seed-42 checkpoint
(trained with crisis_common.DualStreamClassifier) and reproduce its stored dev and test predictions exactly; the two classes must have identical state-dict keys and shapes."""
import json

import numpy as np
import torch
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizerFast

from common import ABL_ART, ART, DATA, RES, SPLIT_DIR, cc
from t2_models import FusionModel

dev = torch.device("cuda")
S = {s: cc.load_split("humanitarian", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("dev", "test")}
bcfg = DistilBertConfig.from_pretrained(ART / "distilbert_config")
ref = cc.DualStreamClassifier(8, bert_config=bcfg, img_dim=768)
new = FusionModel(8, DistilBertModel(bcfg), bcfg.dim, img_dim=768)
rs, ns = ref.state_dict(), new.state_dict()
assert list(rs) == list(ns) and all(rs[k].shape == ns[k].shape for k in rs), "state-dict keys or shapes differ"
print(f"identical state-dict keys and shapes ({len(rs)} tensors, {sum(p.numel() for p in new.parameters()):,} parameters)")
new.load_state_dict(torch.load(ABL_ART / "ckpt" / "t2_L4a-L14_inv_seed42.pt", map_location="cpu")); new.to(dev).eval()
tok = DistilBertTokenizerFast.from_pretrained(ART / "tokenizer")
stored = json.load(open(RES / "G" / "g_fused_L4a-L14_none_vs_inv_seed42.json"))["arms"]["inv"]
for split in ("dev", "test"):
    X = torch.tensor(cc.cached(ABL_ART / "clip_l14" / f"clip_humanitarian_{split}.npy", S[split].image_id.tolist(), lambda: (_ for _ in ()).throw(RuntimeError)), dtype=torch.float32)
    ids, mask = cc.tokenize(tok, S[split]["text"].tolist(), 128)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 128):
            with torch.autocast("cuda", dtype=torch.float16):
                out.append(new(X[i:i + 128].to(dev), ids[i:i + 128].to(dev), mask[i:i + 128].to(dev)).float().cpu())
    pred = torch.cat(out).argmax(1).numpy()
    ok = pred.tolist() == stored[f"pred_{split}"]["42"]
    print(f"{split}: FusionModel(DistilBERT) predictions equal the stored L4a-L14 inv seed 42 predictions on all {len(pred)} rows: {ok}")
    assert ok
print("VALIDATED")
