"""D: Task 2 leave-one-disaster-out with the image-only model on cached CLIP features (a proxy for L4a; they tie on the full test set).

7 folds x seeds 42, 43, 44. Per fold: train on the OTHER six events' train rows, select the checkpoint (best dev macro-F1 over all 8 classes, strict >)
on the OTHER six events' dev rows only, and test on the held-out event's test rows. Recipe = the existing image_only runs (t2_train.py, L4a config):
10 epochs, lr 1e-4, 3 warm-up epochs + cosine, AdamW wd 1e-4, batch 32, inverse-frequency class weights from the fold's train rows, fp16 autocast.
Asserted per fold: the held-out event contributes zero train rows AND zero dev rows (by event_name, tweet_id and image_id), and no class has zero training rows.
In-distribution reference: the existing 3-seed image_only checkpoints (trained on all events) on the same per-event test rows.
Stops (non-zero exit) if any fold-seed run falls below its event's majority-class macro-F1 (always predicting the event's most frequent test label).
Per-event macro-F1 averages the classes present in that event's labels or predictions (as in the C2 tables).
Writes results/ablation/X/d_lodo_t2_<event>_seed<k>.json, d_lodo_t2_overall_seed42.json and e1_t2_lodo_vs_indist_imageonly_seed42.json.
"""
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

from bootstrap_utils import boot
from common import ABL_ART, ART, DATA, SPLIT_DIR, cc, write_row
from t2_models import Variant, param_groups

SEEDS, EPOCHS, WARMUP, LR, BATCH = (42, 43, 44), 10, 3, 1e-4, 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def no_miss():
    raise RuntimeError("cache miss")


def score(yt, yp, all_classes):
    present = sorted(set(yt.tolist()) | set(yp.tolist()))
    return dict(n=int(len(yt)), acc=float((yt == yp).mean()),
                macro_f1=float(f1_score(yt, yp, labels=present, average="macro", zero_division=0)),
                macro_f1_all8=float(f1_score(yt, yp, labels=all_classes, average="macro", zero_division=0)),
                weighted_f1=float(f1_score(yt, yp, labels=all_classes, average="weighted", zero_division=0)))


@torch.no_grad()
def predict(model, X):
    model.eval(); out = []
    for i in range(0, len(X), 512):
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            out.append(model(X[i:i + 512].to(DEVICE), None, None).argmax(-1).cpu())
    return torch.cat(out).numpy()


def train_fold(Xtr, ytr, Xdv, ydv, seed, nc, all_classes):
    cc.seed_all(seed)
    model = Variant("image_only", nc).to(DEVICE)
    cnt = np.bincount(ytr.numpy(), minlength=nc).astype("float32")
    w = 1.0 / cnt; w /= w.sum() / nc
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, device=DEVICE))
    opt = AdamW(param_groups(model, LR), weight_decay=1e-4)
    sched = lr_scheduler.SequentialLR(opt, [lr_scheduler.LinearLR(opt, 1e-6, 1.0, WARMUP),
                                            lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS - WARMUP, eta_min=1e-7)], milestones=[WARMUP])
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True, num_workers=0, generator=g)
    best, best_state, best_ep = -1.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        torch.empty((), dtype=torch.int64).random_(generator=g)          # v2's tqdm(dl) double iter(): same shuffle convention as every other run
        for x, yy in dl:
            x, yy = x.to(DEVICE), yy.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                loss = crit(model(x, None, None).float(), yy)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        sched.step()
        f1 = float(f1_score(ydv.numpy(), predict(model, Xdv), labels=all_classes, average="macro", zero_division=0))
        if f1 > best:
            best, best_ep = f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best, best_ep


def main():
    t_start = time.time()
    S = {s: cc.load_split("humanitarian", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "dev", "test")}
    labels = sorted(S["train"].label.unique()); l2i = {l: i for i, l in enumerate(labels)}; NC = len(labels); ALL = list(range(NC))
    X = {s: torch.tensor(cc.cached(ART / "emb" / f"clip_humanitarian_{s}.npy", S[s].image_id.tolist(), no_miss), dtype=torch.float32) for s in S}
    Y = {s: torch.tensor([l2i[l] for l in S[s].label]) for s in S}
    events = sorted(S["test"].event_name.unique()); assert len(events) == 7

    # in-distribution reference: the existing image_only checkpoints (all events seen) on the full test set
    idp = {}
    for seed in SEEDS:
        m = Variant("image_only", NC).to(DEVICE)
        m.load_state_dict(torch.load(ABL_ART / "ckpt" / f"t2_image_only_seed{seed}.pt", map_location=DEVICE))
        idp[seed] = predict(m, X["test"])
        stored = json.load(open(ABL_ART.parent.parent / "results" / "ablation" / "B" / f"t2_image_only_seed{seed}.json"))
        assert abs(float((idp[seed] == Y["test"].numpy()).mean()) - stored["test_acc"]) < 1e-4, "in-distribution reference != stored image_only result"

    # pre-flight for ALL folds before any training: leakage assertions and class counts
    fold = {}
    for e in events:
        tr = (S["train"].event_name != e).to_numpy(); dv = (S["dev"].event_name != e).to_numpy(); te = (S["test"].event_name == e).to_numpy()
        for name, sel in (("train", tr), ("dev", dv)):
            d = S[name][sel]
            assert (d.event_name == e).sum() == 0, f"LEAK: held-out event {e} has rows in {name}"
            held = S["test"][te]
            assert not set(d.tweet_id) & set(held.tweet_id) and not set(d.image_id) & set(held.image_id), f"LEAK: {e} tweet/image ids in {name}"
        counts = np.bincount(Y["train"].numpy()[tr], minlength=NC)
        zero = [labels[i] for i in range(NC) if counts[i] == 0]
        print(f"pre-flight {e:22s} train {int(tr.sum()):5d} dev {int(dv.sum()):4d} test {int(te.sum()):4d} | smallest class count {counts.min()} ({labels[int(counts.argmin())]}) | "
              f"held-out rows in train/dev: 0/0", flush=True)
        if zero:
            print(f"STOP: fold {e} has no training rows for {zero}"); sys.exit(2)
        fold[e] = (tr, dv, te, counts)

    lodo = {seed: np.full(len(Y["test"]), -1) for seed in SEEDS}
    for e in events:
        tr, dv, te, counts = fold[e]
        yte = Y["test"].numpy()[te]
        maj = int(np.bincount(yte, minlength=NC).argmax())
        base_event = score(yte, np.full(len(yte), maj), ALL)
        base_train = score(yte, np.full(len(yte), int(counts.argmax())), ALL)
        for seed in SEEDS:
            t0 = time.time()
            model, dev_f1, best_ep = train_fold(X["train"][tr], Y["train"][tr], X["dev"][dv], Y["dev"][dv], seed, NC, ALL)
            p = predict(model, X["test"][te]); lodo[seed][te] = p
            m_l, m_i = score(yte, p, ALL), score(yte, idp[seed][te], ALL)
            row = dict(task="T2", tier="gpu-small", model="image_only on cached CLIP features (proxy for L4a)", held_out_event=e,
                       n_train=int(tr.sum()), n_dev=int(dv.sum()), n_test=int(te.sum()), heldout_rows_in_train=0, heldout_rows_in_dev=0,
                       train_class_counts=dict(zip(labels, map(int, counts))), best_epoch=best_ep, dev_macro_f1=dev_f1,
                       lodo=m_l, indist=m_i, majority_event=base_event, majority_train=base_train, pred=p.tolist(), indist_pred=idp[seed][te].tolist(),
                       wall_clock_s=time.time() - t0)
            write_row("X", f"d_lodo_t2_{e}", seed, row)
            print(f"{e:22s} seed {seed}: LODO acc {m_l['acc']:.3f} macro-F1 {m_l['macro_f1']:.3f} | in-dist acc {m_i['acc']:.3f} macro-F1 {m_i['macro_f1']:.3f} | "
                  f"majority-class macro-F1 {base_event['macro_f1']:.3f} | dev F1 {dev_f1:.3f} @ep{best_ep} | {time.time() - t0:.0f}s", flush=True)
            if m_l["macro_f1"] < base_event["macro_f1"]:
                print(f"STOP: fold {e}, seed {seed}: LODO macro-F1 {m_l['macro_f1']:.4f} is below the event's majority-class macro-F1 {base_event['macro_f1']:.4f}")
                sys.exit(3)

    # pooled comparison (seed 42): concatenate the 7 folds' test predictions into one 2,237-row set
    y = Y["test"].numpy()
    assert (lodo[42] >= 0).all() and len(y) == 2237
    pooled = {seed: dict(lodo=score(y, lodo[seed], ALL), indist=score(y, idp[seed], ALL)) for seed in SEEDS}
    b = boot("t2_lodo_vs_indist_imageonly", y, lodo[42], idp[42], NC, ALL, "macro_f1")
    write_row("X", "d_lodo_t2_overall", 42, dict(task="T2", tier="cpu", n_test=int(len(y)), pooled_by_seed={str(k): v for k, v in pooled.items()},
                                                 bootstrap_lodo_minus_indist_seed42=dict(diff=b["diff"], ci95_low=b["ci95_low"], ci95_high=b["ci95_high"],
                                                                                          n_boot=b["n_boot"], excludes_zero=b["ci_excludes_zero"],
                                                                                          lodo_macro_f1=b["score_a"], indist_macro_f1=b["score_b"]),
                                                 total_wall_clock_s=time.time() - t_start))
    print(f"pooled 2,237-row macro-F1, seed 42: LODO {b['score_a']:.4f} vs in-distribution {b['score_b']:.4f}; diff {b['diff']:+.4f} CI [{b['ci95_low']:+.4f}, {b['ci95_high']:+.4f}]")
    print(f"total {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
