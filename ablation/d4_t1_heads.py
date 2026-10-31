"""Day 4, Task 1 image-only heads on frozen features (scored against the text label, SELECTION_RULE_DAY4.md): ResNet50 / DINOv2 / CLIP L/14 -> LayerNorm + MLP.
Fusion models are in d4_t1_fusion.py. Recipe = d2_common.train_head's: AdamW wd 1e-4, lr 1e-4, 10 epochs, batch 32, 3 warm-up epochs then cosine, inverse-frequency CE weights, best dev weighted F1 (T1 headline).
Usage: python d4_t1_heads.py resnet50 dinov2 clip_l14"""
import json
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

import d4_t1_data as d4
import d4_prov
import metrics_all as M
from common import ABL_ART, cc

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS, EPOCHS, WARMUP, BATCH, WD, LR = (42, 43, 44), 10, 3, 32, 1e-4, 1e-4


def feats(name):
    D = d4.data(); out = {}
    for s in ("train", "dev", "test"):
        d = ABL_ART / "d4_feats" / name
        assert json.load(open(d / f"informative_{s}.ids.json")) == D["ids"][s]
        out[s] = torch.tensor(np.load(d / f"informative_{s}.npy"), dtype=torch.float32)
    assert json.load(open(d / "meta.json"))["model"] in d4_prov.HUB, "feature encoder is not a pretrained hub model"
    return out


class Head(nn.Module):
    def __init__(self, d, d_model=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, 2))

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def logits(model, x):
    model.eval()
    return torch.cat([model(x[i:i + 1024].to(DEV)).float().cpu() for i in range(0, len(x), 1024)]).numpy()


def train(x, y, seed):
    cc.seed_all(seed); ytr = torch.tensor(y["train"])
    model = Head(x["train"].shape[1]).to(DEV)
    cnt = np.bincount(y["train"], minlength=2).astype("float32"); w = 1.0 / cnt; w /= w.sum() / 2
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, device=DEV))
    opt = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = lr_scheduler.SequentialLR(opt, [lr_scheduler.LinearLR(opt, 1e-6, 1.0, WARMUP), lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS - WARMUP, eta_min=1e-7)], milestones=[WARMUP])
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(TensorDataset(x["train"], ytr), batch_size=BATCH, shuffle=True, generator=g)
    best, best_state, best_ep = -1.0, None, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(set_to_none=True)
            crit(model(xb.to(DEV)), yb.to(DEV)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        f = float(f1_score(y["dev"], logits(model, x["dev"]).argmax(1), average="weighted"))
        if f > best:
            best, best_ep = f, ep; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best, best_ep


if __name__ == "__main__":
    D = d4.data()
    for name in sys.argv[1:]:
        x = feats(name)
        for seed in SEEDS:
            model, dev_f, ep = train(x, D["y"], seed)
            row = M.save_run("T1", f"img_{name}", seed, d4.LABELS, D["y"]["dev"], logits(model, x["dev"]), D["y"]["test"], logits(model, x["test"]), extra=dict(best_epoch=ep, **d4_prov.prov("informative", f"none (head trained from scratch on frozen {name} features)", feature=name)))
            t = row["test"]
            print(f"img_{name} seed {seed}: ep {ep} dev wF1 {dev_f:.4f} | test acc {t['acc']:.4f} wF1 {t['weighted_f1']:.4f} macroF1 {t['macro_f1']:.4f} MCC {t['mcc']:.4f} AUC {t['roc_auc']:.4f}", flush=True)
