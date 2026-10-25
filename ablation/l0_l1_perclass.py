"""Per-class F1 for T2 L0 vs L1 (seed 42, the 2,237 test rows, crop seed 0), CPU inference only; no training.

E1 regenerated the crop-seed-0 predictions of the L0 and L1 checkpoints but did not save the per-row predictions, so they are regenerated here
exactly as in e1b_l0_l1.py (open_clip CLIP online, RandomResizedCrop(0.9-1.0) test transform, torch.manual_seed(0) before the loader), now on the CPU,
and ASSERTED to reproduce E1's macro-F1 values before use. The per-row predictions are saved this time.
Reports each class's F1 for both, the difference (L1 - L0), the class support, the fraction of its rows from the two earthquake events and its share
of the summed gain, in the same format as the D per-class table. Writes results/ablation/X/l0_l1_perclass_seed42.json.
"""
import json

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import DistilBertConfig, DistilBertTokenizerFast

from common import ABL_ART, ART, DATA, RES, SPLIT_DIR, cc, write_row
from t2_train import Frozen, ImgDS, make_transform

EQ = ("iraq_iran_earthquake", "mexico_earthquake")


def main():
    torch.set_num_threads(12)
    DEVICE = torch.device("cpu")
    tr, te = (cc.load_split("humanitarian", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "test"))
    labels = sorted(tr["label"].unique()); l2i = {l: i for i, l in enumerate(labels)}; ALL = list(range(len(labels)))
    y = np.array([l2i[l] for l in te["label"]])
    tok = DistilBertTokenizerFast.from_pretrained(ART / "tokenizer"); bcfg = DistilBertConfig.from_pretrained(ART / "distilbert_config")
    ids, mask = cc.tokenize(tok, te.text.tolist(), 128)
    yt = torch.tensor(y)
    tf = make_transform("v1_random")
    clip = Frozen("open_clip", DEVICE)
    e1 = json.load(open(RES / "X" / "e1_t2_L1_vs_L0_seed42.json"))
    ref = {"L0": e1["regen_macro_f1_L0_crop_seeds"][0], "L1": e1["regen_macro_f1_L1_crop_seeds"][0]}

    preds = {}
    for rung in ("L0", "L1"):
        model = cc.DualStreamClassifier(len(labels), bert_config=bcfg).to(DEVICE)
        model.load_state_dict(torch.load(ABL_ART / "ckpt" / f"t2_{rung}_seed42.pt", map_location=DEVICE)); model.eval()
        torch.manual_seed(0)                                            # crop seed 0, as in E1
        dl = DataLoader(ImgDS(te, tf, ids, mask, yt), batch_size=32, shuffle=False, num_workers=4)
        out = []
        with torch.no_grad():
            for px, i_, m_, _ in dl:
                out.append(model(clip(px).float(), i_.to(DEVICE), m_.to(DEVICE)).argmax(-1).cpu())
        preds[rung] = torch.cat(out).numpy()
        mf = float(f1_score(y, preds[rung], labels=ALL, average="macro", zero_division=0))
        print(f"{rung}: CPU macro-F1 {mf:.6f} | E1 value {ref[rung]:.6f} | difference {mf - ref[rung]:+.6f}", flush=True)
        assert abs(mf - ref[rung]) < 1e-3, f"{rung}: CPU predictions do not reproduce E1's crop-seed-0 score"

    f0 = f1_score(y, preds["L0"], labels=ALL, average=None, zero_division=0)
    f1 = f1_score(y, preds["L1"], labels=ALL, average=None, zero_division=0)
    eq = te.event_name.isin(EQ).to_numpy()
    gain_total = float((f1 - f0).sum())
    rows = []
    for c, name in enumerate(labels):
        sup = int((y == c).sum()); n_eq = int(((y == c) & eq).sum())
        rows.append(dict(cls=name, support=sup, support_earthquake=n_eq, earthquake_fraction=n_eq / sup, f1_L0=float(f0[c]), f1_L1=float(f1[c]),
                         diff=float(f1[c] - f0[c]), share_of_total_gain=float((f1[c] - f0[c]) / gain_total)))
    print(f"{'class':42s} {'support':>7s} {'eq rows':>7s} {'eq frac':>7s} {'F1 L0':>7s} {'F1 L1':>7s} {'diff':>7s} {'share of gain':>13s}")
    for r in rows:
        print(f"{r['cls']:42s} {r['support']:7d} {r['support_earthquake']:7d} {r['earthquake_fraction']:7.3f} {r['f1_L0']:7.3f} {r['f1_L1']:7.3f} {r['diff']:+7.3f} {r['share_of_total_gain']:13.1%}")
    print(f"macro-F1 {f0.mean():.4f} -> {f1.mean():.4f}; gain {f1.mean() - f0.mean():+.4f}; sum of per-class gains {gain_total:.4f}")
    write_row("X", "l0_l1_perclass", 42, dict(task="T2", tier="cpu", n_test=int(len(y)), crop_seed=0, earthquake_events=list(EQ), earthquake_rows=int(eq.sum()),
                                              cpu_macro_f1_L0=float(f0.mean()), cpu_macro_f1_L1=float(f1.mean()), e1_macro_f1_L0=ref["L0"], e1_macro_f1_L1=ref["L1"],
                                              classes=rows, sum_of_class_gains=gain_total, pred_L0=preds["L0"].tolist(), pred_L1=preds["L1"].tolist()))


if __name__ == "__main__":
    main()
