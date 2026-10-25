"""E1 addition: paired bootstrap of T2 macro-F1, L1 - L0 (seed 42), 10,000 resamples of the 2,237 test rows.

L0 and L1 are scored with v1's evaluation protocol: online open_clip CLIP and the RandomResizedCrop(0.9-1.0) transform at test time. Per-row
predictions were not saved by t2_train.py, so they are regenerated from the checkpoints. Because of the random crop, a regenerated pass
differs a little from the stored score; the script runs three crop seeds, reports the spread against the stored value, and bootstraps the
pass with crop seed 0. Writes results/ablation/X/e1_t2_L1_vs_L0_seed42.json.
"""
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertConfig, DistilBertTokenizerFast

from bootstrap_utils import boot, confusion, macro_f1
from common import ABL_ART, ART, DATA, RES, SPLIT_DIR, cc
from t2_train import Frozen, ImgDS, load_v1_t2, make_transform


def main():                      # the guard below is required: DataLoader workers re-import this module on Windows
    DEVICE = torch.device("cuda")
    tr_full, te = (cc.load_split("humanitarian", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "test"))
    labels = sorted(tr_full["label"].unique()); l2i = {l: i for i, l in enumerate(labels)}
    assert sorted(load_v1_t2("train")["label"].unique()) == labels          # L0's train set has the same 8 classes
    y = np.array([l2i[l] for l in te["label"]])
    tok = DistilBertTokenizerFast.from_pretrained(ART / "tokenizer"); bcfg = DistilBertConfig.from_pretrained(ART / "distilbert_config")
    ids, mask = cc.tokenize(tok, te.text.tolist(), 128)
    yt = torch.tensor(y)
    tf = make_transform("v1_random")
    clip = Frozen("open_clip", DEVICE)                                       # both rungs use open_clip's GELU CLIP

    def predict(rung, crop_seed):
        model = cc.DualStreamClassifier(len(labels), bert_config=bcfg).to(DEVICE)
        model.load_state_dict(torch.load(ABL_ART / "ckpt" / f"t2_{rung}_seed42.pt", map_location=DEVICE)); model.eval()
        torch.manual_seed(crop_seed)
        dl = DataLoader(ImgDS(te, tf, ids, mask, yt), batch_size=32, shuffle=False, num_workers=4)
        out = []
        with torch.no_grad():
            for px, i_, m_, _ in dl:
                out.append(model(clip(px).float(), i_.to(DEVICE), m_.to(DEVICE)).argmax(-1).cpu())
        return torch.cat(out).numpy()

    stored = {r: json.load(open(RES / "A" / f"t2_{r}_seed42.json"))["test_macro_f1"] for r in ("L0", "L1")}
    preds, regen = {}, {}
    for rung in ("L0", "L1"):
        regen[rung] = []
        for cs in (0, 1, 2):
            p = predict(rung, cs)
            regen[rung].append(macro_f1(confusion(y, p, 8), list(range(8))))
            if cs == 0:
                preds[rung] = p
        print(f"{rung}: stored test macro-F1 {stored[rung]:.4f} | regenerated under crop seeds 0,1,2: {np.round(regen[rung], 4).tolist()}", flush=True)
    boot("t2_L1_vs_L0", y, preds["L1"], preds["L0"], 8, list(range(8)), "macro_f1",
         extra=dict(stored_macro_f1_L0=stored["L0"], stored_macro_f1_L1=stored["L1"],
                    regen_macro_f1_L0_crop_seeds=regen["L0"], regen_macro_f1_L1_crop_seeds=regen["L1"], bootstrapped_crop_seed=0))


if __name__ == "__main__":
    main()
