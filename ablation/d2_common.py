"""Day 2 (architecture comparison, ablation/SELECTION_RULE.md "Day 2 addendum"): shared data loading, heads and the trainer for frozen cached features.

The trainer is d_lodo_t2.train_fold generalised to several input tensors (same recipe: AdamW wd 1e-4, lr 1e-4 default, 3 warm-up epochs then cosine, 10 epochs, batch 32,
fp16 autocast, grad clip 1.0, inverse-frequency class weights in float32, v2's shuffle convention, checkpoint = best dev macro-F1 with strict '>'). For the `image_only` head
the module is built in the same order as t2_models.Variant("image_only"), so with the same seed it reproduces the stored runs exactly (checked in d2_run.py).
Results: one JSON per run in results/ablation/H/ (with the dev and test predictions, so every bootstrap can be recomputed).
"""
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

from common import ABL_ART, ART, DATA, RES, SPLIT_DIR, cc, write_row
from extra_metrics import full_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS, EPOCHS, WARMUP, BATCH, WD = (42, 43, 44), 10, 3, 32, 1e-4
HP0 = dict(lr=1e-4, dropout=0.2, d_model=256)
BACKBONE_M = {"clip_b32": 87.8, "clip_l14": 304.0, "dinov2": 86.6, "resnet50": 23.5, "minilm": 22.7, "mpnet": 109.5}   # millions of parameters (SELECTION_RULE.md)
KINDS = ("image_only", "text_only", "concat", "gated_late", "cross_attention")
MISS, INJ, VEH = "Missing or found people", "Injured or dead people", "Vehicle damage"
_D = {}


def _no_miss():
    raise RuntimeError("cache miss")


def data():
    """Humanitarian splits in v2's row order: labels, label ids, ids. Cached per process."""
    if not _D:
        S = {s: cc.load_split("humanitarian", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "dev", "test")}
        labels = sorted(S["train"].label.unique()); l2i = {l: i for i, l in enumerate(labels)}
        _D.update(S=S, labels=labels, l2i=l2i, nc=len(labels), ids={s: S[s].image_id.tolist() for s in S},
                  y={s: torch.tensor([l2i[l] for l in S[s].label]) for s in S})
    return _D


def img_feats(name):
    D = data(); out = {}
    for s in ("train", "dev", "test"):
        if name == "clip_b32":
            a = cc.cached(ART / "emb" / f"clip_humanitarian_{s}.npy", D["ids"][s], _no_miss)          # v2's cache: raw 512-d
        elif name == "clip_l14":
            a = cc.cached(ABL_ART / "clip_l14" / f"clip_humanitarian_{s}.npy", D["ids"][s], _no_miss)  # L2-normalised 768-d
        else:
            d = ABL_ART / "d2_feats" / name
            assert json.load(open(d / f"humanitarian_{s}.ids.json")) == D["ids"][s]
            a = np.load(d / f"humanitarian_{s}.npy")
        out[s] = torch.tensor(a, dtype=torch.float32)
    return out


def txt_feats(name):
    D = data(); out = {}
    for s in ("train", "dev", "test"):
        d = ABL_ART / "d2_feats" / name
        assert json.load(open(d / f"humanitarian_{s}.ids.json")) == D["ids"][s]
        out[s] = torch.tensor(np.load(d / f"humanitarian_{s}.npy"), dtype=torch.float32)
    return out


def tok_feats(name):
    """Frozen token features of the chosen text encoder: fp16 [N, 64, D] and a bool padding mask [N, 64] (True = pad)."""
    D = data(); out = {}
    for s in ("train", "dev", "test"):
        d = ABL_ART / "d2_feats" / name
        z = np.load(d / f"tokens_humanitarian_{s}.npz")
        out[s] = (torch.tensor(z["tok"]), torch.tensor(z["pad"]))
    return out


class Head(nn.Module):
    """Small trainable head on frozen features. d_model 256, dropout 0.2, 4 heads by default (SELECTION_RULE.md)."""

    def __init__(self, kind, nc, d_img=None, d_txt=None, d_model=256, dropout=0.2, heads=4):
        super().__init__()
        assert kind in KINDS
        self.kind = kind
        if kind != "text_only":
            self.img_proj = nn.Sequential(nn.Linear(d_img, d_model), nn.LayerNorm(d_model))
        if kind != "image_only":
            self.txt_proj = nn.Sequential(nn.Linear(d_txt, d_model), nn.LayerNorm(d_model))
        mlp = lambda w: nn.Sequential(nn.Dropout(dropout), nn.Linear(w, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, nc))
        if kind == "cross_attention":
            self.img_to_txt = cc.CrossModalAttention(d_model, heads, dropout)
            self.txt_to_img = cc.CrossModalAttention(d_model, heads, dropout)   # single image key: effectively a projection (as in v2)
        if kind == "gated_late":
            self.cls_img, self.cls_txt = mlp(d_model), mlp(d_model)
            self.gate = nn.Linear(2 * d_model, 1)
        else:
            self.classifier = mlp(d_model * (2 if kind in ("concat", "cross_attention") else 1))

    def forward(self, *x):
        k = self.kind
        if k == "image_only":
            return self.classifier(self.img_proj(x[0]))
        if k == "text_only":
            return self.classifier(self.txt_proj(x[0]))
        if k == "concat":
            return self.classifier(torch.cat([self.img_proj(x[0]), self.txt_proj(x[1])], dim=-1))
        if k == "gated_late":
            i, t = self.img_proj(x[0]), self.txt_proj(x[1])
            g = torch.sigmoid(self.gate(torch.cat([i, t], dim=-1)))
            return g * self.cls_img(i) + (1 - g) * self.cls_txt(t)
        img = self.img_proj(x[0]).unsqueeze(1)                 # cross_attention: x = (image vector, token features, padding mask)
        txt = self.txt_proj(x[1])
        img_att = self.img_to_txt(img, txt, x[2]).squeeze(1)
        txt_att = self.txt_to_img(txt[:, 0:1, :], img).squeeze(1)
        return self.classifier(torch.cat([img_att, txt_att], dim=-1))


def n_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _dev(t):
    return t.to(DEVICE).float() if t.is_floating_point() else t.to(DEVICE)


@torch.no_grad()
def predict(model, xs):
    model.eval(); out = []
    n = len(xs[0])
    for i in range(0, n, 512):
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            out.append(model(*[_dev(x[i:i + 512]) for x in xs]).float().cpu())
    return torch.cat(out).numpy()


def train_head(kind, xs, seed, hp, d_img=None, d_txt=None):
    """xs = {'train': (tensors...), 'dev': (...), 'test': (...)}: the inputs the head consumes, in forward order. Returns model (best dev checkpoint loaded), best dev F1, best epoch."""
    D = data(); nc = D["nc"]; ALL = list(range(nc)); ytr, ydv = D["y"]["train"], D["y"]["dev"]
    cc.seed_all(seed)
    model = Head(kind, nc, d_img, d_txt, hp["d_model"], hp["dropout"]).to(DEVICE)
    cnt = np.bincount(ytr.numpy(), minlength=nc).astype("float32")
    w = 1.0 / cnt; w /= w.sum() / nc
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, device=DEVICE))
    opt = AdamW(model.parameters(), lr=hp["lr"], weight_decay=WD)
    sched = lr_scheduler.SequentialLR(opt, [lr_scheduler.LinearLR(opt, 1e-6, 1.0, WARMUP),
                                            lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS - WARMUP, eta_min=1e-7)], milestones=[WARMUP])
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(TensorDataset(*xs["train"], ytr), batch_size=BATCH, shuffle=True, num_workers=0, generator=g)
    best, best_state, best_ep = -1.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        torch.empty((), dtype=torch.int64).random_(generator=g)          # v2's tqdm(dl) double iter(): the shuffle convention of every other run
        for b in dl:
            *x, yy = b
            x, yy = [_dev(t) for t in x], yy.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                loss = crit(model(*x).float(), yy)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        sched.step()
        f1 = float(f1_score(ydv.numpy(), predict(model, xs["dev"]).argmax(1), labels=ALL, average="macro", zero_division=0))
        if f1 > best:
            best, best_ep = f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best, best_ep


def run_arm(decision, name, kind, xs, seed, hp, d_img=None, d_txt=None, backbones=(), extra=None, write=True):
    """Train one arm at one seed, score dev and test once with the dev-selected checkpoint, write one JSON. Returns the row."""
    D = data(); labels = D["labels"]; t0 = time.time()
    model, dev_f1, best_ep = train_head(kind, xs, seed, hp, d_img, d_txt)
    pd_, pt_ = (predict(model, xs[s]).argmax(1) for s in ("dev", "test"))
    m = full_metrics(D["y"]["test"].numpy(), pt_, labels)
    row = dict(task="T2", tier="gpu-small", decision=decision, arm=name, kind=kind, hp=hp, backbones=list(backbones), backbone_params_m=float(sum(BACKBONE_M[b] for b in backbones)),
               n_trainable=n_trainable(model), dev_macro_f1=dev_f1, best_epoch=best_ep, test=m, test_macro_f1=m["macro_f1"], test_balanced_acc=m["balanced_acc"],
               recall_injured=m["per_class"][INJ]["recall"], recall_vehicle=m["per_class"][VEH]["recall"], recall_missing=m["per_class"][MISS]["recall"],
               n_test_injured=m["per_class"][INJ]["support"], n_test_vehicle=m["per_class"][VEH]["support"],
               pred_dev=pd_.tolist(), pred_test=pt_.tolist(), wall_clock_s=time.time() - t0, **(extra or {}))
    if write:
        write_row("H", name, seed, row)
    print(f"{name:34s} seed {seed}: dev F1 {dev_f1:.4f} @ep{best_ep} | test acc {m['acc']:.4f} macro-F1 {m['macro_f1']:.4f} bal-acc {m['balanced_acc']:.4f} | "
          f"params {row['n_trainable']:,} | {time.time() - t0:.0f}s", flush=True)
    return row
