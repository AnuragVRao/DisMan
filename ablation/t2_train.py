"""Task 2 training with the v1 -> v2 ladder switched by flags (ABLATION_PLAN.md, Part A).

Rungs (each flag is independent; presets just name the combinations):
  L0  v1 buggy train/dev (lstrip path logic), open_clip GELU CLIP, v1 random-crop transform, 30 epochs, fp32, online CLIP
  L1  full corrected train/dev, otherwise L0
  L2  L1 with the fixed activation (HF CLIPModel, QuickGELU); still the v1 random-crop transform, online
  L3  L2 with the epoch budget 30 -> 10 (warm-up 3 + cosine over the remaining epochs, as v2)
  L3b L3 with the deterministic eval transform (v2 preprocessing) so features can be cached; fp32, no AMP
  L4a L3b + AMP (fp16 autocast + GradScaler), pretrained DistilBERT: v2 as intended
  L4  v2 exactly: L4a but DistilBERT randomly initialised (v2's DistilBertModel(config) behaviour; see PRESETS)
Part B variants (--variant text_only|image_only|concat|no_t2i) run on L4a's config and write to results/ablation/B/.
The test set is always the full corrected humanitarian test split (2237 rows), evaluated once with the dev-selected checkpoint.
Does not touch artifacts/ files that v2 uses; checkpoints go to artifacts/ablation/ckpt/.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import DistilBertConfig, DistilBertTokenizerFast
from PIL import Image
import torchvision.transforms as T

from common import ABL_ART, ART, DATA, SPLIT_DIR, cc, write_row
from t2_models import VARIANTS, Variant, param_groups

PRESETS = {
    # bert = DistilBERT initialisation: v1 loaded the pretrained weights; v2's training cell builds DistilBertModel(config), i.e. RANDOM init
    #       data     clip     transform  epochs cache  amp    bert
    "L0":  dict(data="v1", clip="open_clip", tf="v1_random", epochs=30, cache=False, amp=False, bert="pretrained"),
    "L1":  dict(data="full", clip="open_clip", tf="v1_random", epochs=30, cache=False, amp=False, bert="pretrained"),
    "L2":  dict(data="full", clip="hf", tf="v1_random", epochs=30, cache=False, amp=False, bert="pretrained"),
    "L3":  dict(data="full", clip="hf", tf="v1_random", epochs=10, cache=False, amp=False, bert="pretrained"),
    "L3b": dict(data="full", clip="hf", tf="deterministic", epochs=10, cache=True, amp=False, bert="pretrained"),
    "L4a": dict(data="full", clip="hf", tf="deterministic", epochs=10, cache=True, amp=True, bert="pretrained"),
    "L4":  dict(data="full", clip="hf", tf="deterministic", epochs=10, cache=True, amp=True, bert="random"),   # = v2
}
T2_BATCH, T2_MAX_LEN, T2_LR, T2_WARMUP = 32, 128, 1e-4, 3
MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


# ── data ──────────────────────────────────────────────────────────────────────
def load_v1_t2(split):
    """capstone-final.ipynb cell 17 (t2_load), reproduced line for line, including the lstrip path bug."""
    df = pd.read_csv(SPLIT_DIR / f"task_humanitarian_text_img_{split}.tsv", sep="\t", dtype=str)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.rename(columns={k: v for k, v in {"tweet_text": "text", "text_info": "text"}.items() if k in df.columns})
    df["label_text"] = df["label_text"].apply(cc.t2_normalise)
    df["label_image"] = df["label_image"].apply(cc.t2_normalise)

    def resolve_img(row):
        if pd.isna(row.get("image")):
            return None
        first = str(row["image"]).split(",")[0].strip().lstrip("data_image/")
        full = DATA / "data_image" / first
        return str(full) if full.exists() else None

    df["img_path"] = df.apply(resolve_img, axis=1)
    df = df[df["img_path"].notna()].copy()
    df = df[df["text"].notna() & (df["text"].str.strip() != "")].copy()
    df = df[df["label_text"] != "Unknown"].copy()
    df["text"] = df["text"].apply(cc.clean_tweet)
    df["label"] = df["label_image"]
    return df.reset_index(drop=True)


def load_data(kind):
    full = {s: cc.load_split("humanitarian", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "dev", "test")}
    if kind == "v1":  # buggy train/dev only; test stays the full corrected split
        full["train"], full["dev"] = load_v1_t2("train"), load_v1_t2("dev")
    return full


# ── image side ────────────────────────────────────────────────────────────────
def make_transform(kind):
    if kind == "v1_random":  # open_clip's *train* transform, which v1 used for train, dev and test
        return T.Compose([T.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.75, 1.3333),
                                              interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                          T.ToTensor(), T.Normalize(MEAN, STD)])
    return None  # deterministic path is the HF CLIPProcessor (cached embeddings)


class Frozen:
    """Frozen CLIP image tower returning the raw 512-d projected feature, for either implementation."""

    def __init__(self, kind, device):
        self.kind, self.device = kind, device
        if kind == "open_clip":
            import open_clip
            m, _, _ = open_clip.create_model_and_transforms("ViT-B/32", pretrained="openai", device=device)
            self.net = m.visual.eval()
            self.fn = lambda px: self.net(px)
        else:
            m, _ = cc.load_clip(device, ART / "clip")
            assert m.config.vision_config.hidden_act == "quick_gelu"
            self.net = m
            self.fn = lambda px: m.get_image_features(pixel_values=px)
        for p in self.net.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def __call__(self, px):
        out = self.fn(px.to(self.device))
        return out if isinstance(out, torch.Tensor) else out.image_embeds


class ImgDS(Dataset):
    def __init__(self, df, tf, ids, mask, y):
        self.paths, self.tf, self.ids, self.mask, self.y = df["img_path"].tolist(), tf, ids, mask, y

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        return self.tf(img), self.ids[i], self.mask[i], self.y[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=PRESETS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int)               # override the preset (e.g. 1 for a smoke test)
    ap.add_argument("--max-minutes", type=float)        # abort and report if the run exceeds this
    ap.add_argument("--workers", type=int, default=6)   # train loader workers for the online-CLIP rungs (dev/test use 4)
    ap.add_argument("--debug-steps", type=int, default=0)  # print the first N step losses and exit (determinism debugging)
    ap.add_argument("--variant", choices=("full",) + VARIANTS, default="full")   # Part B component ablation
    ap.add_argument("--tier", default="")               # batch tier, recorded in the result JSON
    ap.add_argument("--tag", default="")                # suffix for smoke/timing runs so real rows are never overwritten
    # opt-in switches for the L4a-L14 imbalance check (SELECTION_RULE.md addendum); the defaults reproduce every earlier run exactly
    ap.add_argument("--feat", choices=("b32", "l14"), default="b32")   # cached CLIP features: v2's B/32 (512-d, raw) or L/14 (768-d, L2-normalised)
    ap.add_argument("--loss", choices=("inv", "none"), default="inv")  # inverse-frequency class weights (all earlier runs) or unweighted cross-entropy
    ap.add_argument("--part", default=None)                            # result folder (default: A for the full model, B for the Part B variants)
    a = ap.parse_args()
    cfg = dict(PRESETS[a.config]); E = a.epochs or cfg["epochs"]
    device = torch.device("cuda")
    cc.seed_all(a.seed)
    t_start = time.time()

    data = load_data(cfg["data"])
    print({s: len(d) for s, d in data.items()}, "| config", a.config, cfg, "epochs", E, flush=True)
    tr, dv, te = data["train"], data["dev"], data["test"]
    labels = sorted(tr["label"].unique()); l2i = {l: i for i, l in enumerate(labels)}; NC = len(labels)
    ALL = list(range(NC))
    assert set(dv.label) <= set(labels) and set(te.label) <= set(labels)

    tok = DistilBertTokenizerFast.from_pretrained(ART / "tokenizer")
    bert_cfg = DistilBertConfig.from_pretrained(ART / "distilbert_config")
    tokd, y = {}, {}
    for s, d in data.items():
        tokd[s] = cc.tokenize(tok, d["text"].tolist(), T2_MAX_LEN)
        y[s] = torch.tensor([l2i[l] for l in d["label"]])

    # image features: cached (deterministic) or online (frozen CLIP + v1 random-crop transform)
    assert a.feat == "b32" or (cfg["cache"] and cfg["data"] == "full" and a.variant == "full"), "--feat l14 needs a cached-feature preset and the full model"
    fdir = ART / "emb" if a.feat == "b32" else ABL_ART / "clip_l14"
    if cfg["cache"]:
        feats = {s: torch.tensor(cc.cached(fdir / f"clip_humanitarian_{s}.npy", d["image_id"].tolist(),
                                           lambda: (_ for _ in ()).throw(RuntimeError("cache miss"))), dtype=torch.float32)
                 for s, d in data.items()}
        loaders = {s: DataLoader(TensorDataset(feats[s], *tokd[s], y[s]), batch_size=T2_BATCH if s == "train" else 128,
                                 shuffle=(s == "train"), num_workers=0,
                                 generator=torch.Generator().manual_seed(a.seed) if s == "train" else None)
                   for s in data}
        frozen = None
    else:
        frozen = Frozen(cfg["clip"], device)
        tf = make_transform(cfg["tf"])
        # dev is evaluated every epoch, so its workers persist; test is read once, so its workers are not kept alive
        nw = {"train": a.workers, "dev": 4, "test": 4}
        loaders = {s: DataLoader(ImgDS(d, tf, *tokd[s], y[s]), batch_size=T2_BATCH, shuffle=(s == "train"),
                                 num_workers=nw[s], pin_memory=True, persistent_workers=(s != "test"),
                                 generator=torch.Generator().manual_seed(a.seed) if s == "train" else None)
                   for s, d in data.items()}

    cc.seed_all(a.seed)  # same head init regardless of which CLIP implementation was loaded above
    bcfg = bert_cfg if cfg["bert"] == "random" else None
    img_dim = feats["train"].shape[1] if cfg["cache"] else 512
    model = (cc.DualStreamClassifier(NC, bert_config=bcfg, img_dim=img_dim) if a.variant == "full"
             else Variant(a.variant, NC, bert_config=bcfg)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    cnt = tr["label"].map(l2i).value_counts().reindex(ALL).values.astype("float32")
    w = 1.0 / cnt; w /= w.sum() / NC
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, device=device)) if a.loss == "inv" else nn.CrossEntropyLoss()
    opt = AdamW(param_groups(model, T2_LR), weight_decay=1e-4)
    sched = lr_scheduler.SequentialLR(opt, [lr_scheduler.LinearLR(opt, 1e-6, 1.0, T2_WARMUP),
                                            lr_scheduler.CosineAnnealingLR(opt, T_max=E - T2_WARMUP, eta_min=1e-7)],
                                      milestones=[T2_WARMUP])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])

    def batch_to_dev(b):
        if frozen is not None:
            px, ids, mask, yy = b
            img = frozen(px).float()
        else:
            img, ids, mask, yy = b
            img = img.to(device)
        return img, ids.to(device), mask.to(device), yy.to(device)

    @torch.no_grad()
    def evaluate(split):
        model.eval(); preds = []
        for b in loaders[split]:
            img, ids, mask, _ = batch_to_dev(b)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg["amp"]):
                preds.append(model(img, ids, mask).argmax(-1).cpu())
        p = torch.cat(preds).numpy(); t = y[split].numpy()
        return p, dict(acc=float((p == t).mean()),
                       macro_f1=float(f1_score(t, p, labels=ALL, average="macro", zero_division=0)),
                       weighted_f1=float(f1_score(t, p, labels=ALL, average="weighted", zero_division=0)))

    best_f1, best_state, best_epoch, hist, ep_times, aborted = -1.0, None, 0, [], [], False
    dbg = []
    for epoch in range(1, E + 1):
        t0 = time.time(); model.train(); tot = 0.0
        # v2 iterates `tqdm(dl)`; tqdm.auto calls iter(dl) twice, and the discarded iterator draws one base seed from the
        # loader's generator. Draw it here too, so every rung shuffles in v2's order (verified: reproduces v2's step losses).
        torch.empty((), dtype=torch.int64).random_(generator=loaders["train"].generator)
        for b in loaders["train"]:
            img, ids, mask, yy = batch_to_dev(b)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg["amp"]):
                loss = crit(model(img, ids, mask).float(), yy)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); tot += loss.item()
            if a.debug_steps:
                dbg.append(loss.item())
                if len(dbg) >= a.debug_steps:
                    print("DEBUG step losses:", [round(x, 6) for x in dbg]); return
        sched.step()
        _, dm = evaluate("dev")
        ep_times.append(time.time() - t0)
        hist.append(dict(epoch=epoch, train_loss=tot / len(loaders["train"]), **{f"dev_{k}": v for k, v in dm.items()}))
        flag = ""
        if dm["macro_f1"] > best_f1:  # v1 rule: strict '>' on dev macro-F1 (identical in v2)
            best_f1, best_epoch, flag = dm["macro_f1"], epoch, "  <- best"
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:2d}/{E} loss {tot/len(loaders['train']):.4f} dev acc {dm['acc']:.4f} macroF1 {dm['macro_f1']:.4f} "
              f"{ep_times[-1]:.0f}s (elapsed {(time.time()-t_start)/60:.1f} min){flag}", flush=True)
        if a.max_minutes and (time.time() - t_start) / 60 > a.max_minutes:
            aborted = True
            print(f"ABORT: exceeded --max-minutes {a.max_minutes} after epoch {epoch}/{E}", flush=True)
            break

    part = a.part or ("A" if a.variant == "full" else "B")
    name = f"t2_{a.config if a.variant == 'full' else a.variant}{a.tag}"
    if aborted:
        row = dict(task="T2", config=name, aborted=True, tier=a.tier, epochs_trained=len(hist), epochs_planned=E,
                   mean_epoch_s=float(np.mean(ep_times)), projected_total_min=float(np.mean(ep_times) * E / 60),
                   wall_clock_s=time.time() - t_start, history=hist)
        print(write_row(part, name, a.seed, row)); return

    model.load_state_dict(best_state)
    (ABL_ART / "ckpt").mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ABL_ART / "ckpt" / f"{name}_seed{a.seed}.pt")
    p, tm = evaluate("test")                      # test is evaluated once, with the dev-selected checkpoint
    t = y["test"].numpy()
    pcf = f1_score(t, p, labels=ALL, average=None, zero_division=0)
    cm = confusion_matrix(t, p, labels=ALL)
    maj = np.full_like(t, int(np.argmax(cnt)))        # majority-class baseline (stop-and-ask condition in the plan)
    maj_f1 = float(f1_score(t, maj, labels=ALL, average="macro", zero_division=0))
    row = dict(task="T2", config=name, dev_macro_f1=best_f1, test_acc=tm["acc"], test_macro_f1=tm["macro_f1"],
               test_weighted_f1=tm["weighted_f1"], per_class_f1=dict(zip(labels, map(float, pcf))),
               confusion_matrix=cm.tolist(), labels=labels, epochs_trained=len(hist), best_epoch=best_epoch,
               wall_clock_s=time.time() - t_start, n_train=len(tr), n_dev=len(dv), n_test=len(te), n_params=n_params,
               mean_epoch_s=float(np.mean(ep_times)), flags=dict(cfg, epochs=E, feat=a.feat, loss=a.loss), history=hist,
               variant=a.variant, tier=a.tier, majority_baseline_macro_f1=maj_f1,
               below_majority_baseline=bool(tm["macro_f1"] < maj_f1))
    print(write_row(part, name, a.seed, row))
    print(f"TEST acc {tm['acc']:.4f} macroF1 {tm['macro_f1']:.4f} wF1 {tm['weighted_f1']:.4f} | best dev macroF1 {best_f1:.4f} "
          f"@ epoch {best_epoch} | wall {row['wall_clock_s']/60:.1f} min")


if __name__ == "__main__":
    main()
