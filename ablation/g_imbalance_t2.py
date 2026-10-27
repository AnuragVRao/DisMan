"""G: Task 2 class-imbalance treatments (image-only head on cached CLIP features; guide feedback after the presentation).

Same recipe as d_lodo_t2.train_fold / b_t2_l14.py (10 epochs, lr 1e-4, 3 warm-up + cosine, AdamW wd 1e-4, batch 32, fp16 autocast, v2's shuffle
convention, checkpoint = best dev macro-F1 over all 8 classes with strict '>'), seeds 42/43/44. Only the imbalance treatment changes:
  none         unweighted cross-entropy
  inv          inverse-frequency weights, rescaled to mean 1 (the current choice; must reproduce the stored image_only runs exactly for B32)
  sqrt_inv     weights ~ 1/sqrt(count), rescaled to mean 1
  eff_num      class-balanced weights (1-beta)/(1-beta^n), beta = 0.999 (Cui et al. 2019), rescaled to mean 1
  focal1/2     focal loss, gamma 1 / 2, unweighted
  focal2_inv   focal loss, gamma 2, inverse-frequency alpha
  sampler      WeightedRandomSampler (inverse-frequency, with replacement), unweighted cross-entropy
  logit_adj    logit-adjusted loss: CE(z + tau*log prior), tau = 1; plain argmax of z at inference (Menon et al. 2021)
  none_posthoc post-hoc logit adjustment of the 'none' model: z - tau*log prior, tau in {0, .25, ..., 2} picked on dev macro-F1 (never on test)
The test set is scored once per run, with the dev-selected checkpoint. Dev/test logits go to artifacts/ablation/imbalance/ (for extra metrics),
one JSON row per run to results/ablation/G/g_t2_<feat>_<arm>_seed<k>.json.
Usage: python g_imbalance_t2.py [--feat b32|l14] [--arms none,inv,...]
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from common import ABL_ART, ART, DATA, RES, SPLIT_DIR, cc, write_row
from d_lodo_t2 import BATCH, DEVICE, EPOCHS, LR, SEEDS, WARMUP, no_miss
from extra_metrics import full_metrics, macro_f1_ci
from t2_models import Variant, param_groups

ARMS = ("none", "inv", "sqrt_inv", "eff_num", "focal1", "focal2", "focal2_inv", "sampler", "logit_adj")
TAUS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
OUT = ABL_ART / "imbalance"


def class_weights(arm, cnt):
    nc = len(cnt)
    if arm in ("inv", "focal2_inv"):
        w = 1.0 / cnt
    elif arm == "sqrt_inv":
        w = 1.0 / np.sqrt(cnt)
    elif arm == "eff_num":
        w = (1 - 0.999) / (1 - 0.999 ** cnt)
    else:
        return None
    w = w / (w.sum() / nc)
    return torch.tensor(w, device=DEVICE)


def make_loss(arm, cnt):
    w = class_weights(arm, cnt)
    gamma = {"focal1": 1.0, "focal2": 2.0, "focal2_inv": 2.0}.get(arm)
    log_prior = torch.tensor(np.log(cnt / cnt.sum()), dtype=torch.float32, device=DEVICE)
    if arm == "logit_adj":
        return lambda z, y: F.cross_entropy(z + log_prior, y)
    if gamma is None:
        return lambda z, y: F.cross_entropy(z, y, weight=w)
    def focal(z, y):
        ce = F.cross_entropy(z, y, reduction="none")
        pt = torch.exp(-ce)
        a = w[y] if w is not None else 1.0
        return (a * (1 - pt) ** gamma * ce).mean()
    return focal


@torch.no_grad()
def logits_of(model, X):
    model.eval(); out = []
    for i in range(0, len(X), 512):
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            out.append(model(X[i:i + 512].to(DEVICE), None, None).float().cpu())
    return torch.cat(out).numpy()


def train_arm(arm, Xtr, ytr, Xdv, ydv, seed, nc, img_dim):
    cc.seed_all(seed)
    model = Variant("image_only", nc, img_dim=img_dim).to(DEVICE)
    cnt = np.bincount(ytr.numpy(), minlength=nc).astype("float32")     # float32 like d_lodo_t2.train_fold, so 'inv' reproduces it bit for bit
    crit = make_loss(arm, cnt)
    opt = AdamW(param_groups(model, LR), weight_decay=1e-4)
    sched = lr_scheduler.SequentialLR(opt, [lr_scheduler.LinearLR(opt, 1e-6, 1.0, WARMUP),
                                            lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS - WARMUP, eta_min=1e-7)], milestones=[WARMUP])
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    g = torch.Generator().manual_seed(seed)
    if arm == "sampler":
        sw = torch.tensor((1.0 / cnt)[ytr.numpy()], dtype=torch.double)
        dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, sampler=WeightedRandomSampler(sw, len(ytr), replacement=True, generator=g),
                        num_workers=0)
    else:
        dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True, num_workers=0, generator=g)
    ids = list(range(nc)); best, best_state, best_ep = -1.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        torch.empty((), dtype=torch.int64).random_(generator=g)          # v2's shuffle convention (see d_lodo_t2.train_fold)
        for x, yy in dl:
            x, yy = x.to(DEVICE), yy.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                loss = crit(model(x, None, None).float(), yy)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        sched.step()
        f1 = float(f1_score(ydv.numpy(), logits_of(model, Xdv).argmax(1), labels=ids, average="macro", zero_division=0))
        if f1 > best:
            best, best_ep = f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best, best_ep


def record(feat, arm, seed, labels, ydv, ytest, zdv, zte, dev_f1, best_ep, wall, extra=None):
    pred = zte.argmax(1); k = len(labels)
    m = full_metrics(ytest, pred, labels, zte)
    row = dict(task="T2", tier="gpu-small", feat=feat, arm=arm, dev_macro_f1=dev_f1, best_epoch=best_ep, test=m,
               macro_f1_ci=macro_f1_ci(ytest, pred, k), wall_clock_s=wall, pred=pred.tolist(), **(extra or {}))
    write_row("G", f"g_t2_{feat}_{arm}", seed, row)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{feat}_{arm}_seed{seed}.npz", dev=zdv, test=zte)
    print(f"{feat} {arm:12s} seed {seed}: dev F1 {dev_f1:.4f} @ep{best_ep} | test acc {m['acc']:.4f} macro-F1 {m['macro_f1']:.4f} "
          f"bal-acc {m['balanced_acc']:.4f} MCC {m['mcc']:.4f} PR-AUC {m['macro_pr_auc']:.4f} | "
          f"min-class F1 {min(v['f1'] for v in m['per_class'].values()):.3f} | {wall:.0f}s", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", choices=("b32", "l14"), default="b32")
    ap.add_argument("--arms", default=",".join(ARMS))
    a = ap.parse_args()
    S = {s: cc.load_split("humanitarian", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "dev", "test")}
    labels = sorted(S["train"].label.unique()); l2i = {l: i for i, l in enumerate(labels)}; NC = len(labels)
    Y = {s: torch.tensor([l2i[l] for l in S[s].label]) for s in S}
    if a.feat == "b32":
        X = {s: torch.tensor(cc.cached(ART / "emb" / f"clip_humanitarian_{s}.npy", S[s].image_id.tolist(), no_miss), dtype=torch.float32) for s in S}
    else:
        L14 = ABL_ART / "clip_l14"
        X = {s: torch.tensor(cc.cached(L14 / f"clip_humanitarian_{s}.npy", S[s].image_id.tolist(), no_miss), dtype=torch.float32) for s in S}
    dim = X["train"].shape[1]; ytest, ydv = Y["test"].numpy(), Y["dev"].numpy()
    log_prior = np.log(np.bincount(Y["train"].numpy(), minlength=NC) / len(Y["train"]))
    for arm in a.arms.split(","):
        for seed in SEEDS:
            t0 = time.time()
            model, dev_f1, best_ep = train_arm(arm, X["train"], Y["train"], X["dev"], Y["dev"], seed, NC, dim)
            zdv, zte = logits_of(model, X["dev"]), logits_of(model, X["test"])
            m = record(a.feat, arm, seed, labels, ydv, ytest, zdv, zte, dev_f1, best_ep, time.time() - t0)
            if arm == "inv":      # validation: must reproduce the stored image_only runs (B/32: Part B; L/14 normalised: b_t2_l14.py)
                st = json.load(open(RES / "B" / f"t2_image_only_seed{seed}.json" if a.feat == "b32"
                                    else RES / "X" / f"b_t2_imageonly_L14_norm_seed{seed}.json"))
                d = max(abs(m["acc"] - st["test_acc"]), abs(m["macro_f1"] - st["test_macro_f1"]), abs(dev_f1 - st["dev_macro_f1"]))
                ref = "Part B image_only" if a.feat == "b32" else "L/14 normalised image-only (b_t2_l14.py)"
                print(f"          validation vs stored {ref} seed {seed}: max |difference| {d:.2e}", flush=True)
                assert d < 1e-4, "the 'inv' arm does not reproduce Part B image_only"
            if arm == "none":                          # post-hoc logit adjustment of the unweighted model; tau chosen on dev only
                ph = []
                for tau in TAUS:
                    f1 = float(f1_score(ydv, (zdv - tau * log_prior).argmax(1), labels=list(range(NC)), average="macro", zero_division=0))
                    ph.append(f1)
                tau = TAUS[int(np.argmax(ph))]           # argmax returns the first (smallest tau) on ties
                record(a.feat, "none_posthoc", seed, labels, ydv, ytest, zdv - tau * log_prior, zte - tau * log_prior, max(ph), best_ep, 0.0,
                       extra=dict(tau=tau, dev_f1_by_tau=dict(zip(map(str, TAUS), ph))))


if __name__ == "__main__":
    main()
