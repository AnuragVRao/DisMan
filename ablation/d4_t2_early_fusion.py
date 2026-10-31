"""Day 4 step 5d, Task 2 core model: BERTweet (fine-tuned) + ResNet50 (frozen), concat -> MLP, 8 humanitarian classes (`label_image`). Recipe and architecture exactly as pre-registered
(SELECTION_RULE_DAY4.md rule 16): the DistilBERT/L4a-matched recipe (10 epochs, batch 32, AdamW wd 1e-4, encoder lr 1e-5 / every other module lr 1e-4, 3 linear warm-up epochs then cosine,
grad clip 1.0, fp16 autocast, inverse-frequency weighted CE, dev macro-F1 selection with strict >), BERTweet from the hub id, raw tweet_text with BERTweet normalisation,
frozen ResNet50 cache artifacts/ablation/d2_feats/resnet50. Run alone on the GPU. Usage: python d4_t2_early_fusion.py [seed ...]"""
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW, lr_scheduler
from transformers import AutoModel, AutoTokenizer

import d4_prov
import d4_t2_data as d4
import metrics_all as M
from common import ABL_ART, cc

HUB_ID, MAXLEN, BATCH, WD, E = "vinai/bertweet-base", 128, 32, 1e-4, 10
NAME = "t2_early_bertweet_resnet50"
DEV = torch.device("cuda")


class Fusion(nn.Module):
    """Variant('concat') of t2_models with a BERTweet text tower: classifier(cat[img_proj(x), txt_proj(tokens)[:, 0]])."""

    def __init__(self, nc, d_img=2048, d_model=256, dropout=0.2):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(HUB_ID)
        self.img_proj = nn.Sequential(nn.Linear(d_img, d_model), nn.LayerNorm(d_model))
        self.txt_proj = nn.Sequential(nn.Linear(768, d_model), nn.LayerNorm(d_model))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, nc))

    def forward(self, ids, mask, x):
        cls = self.txt_proj(self.text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state)[:, 0]
        return self.classifier(torch.cat([self.img_proj(x), cls], dim=-1))


def trim(ids, mask):
    L = int(mask.sum(1).max()); return ids[:, :L], mask[:, :L]


@torch.no_grad()
def logits(model, ids, mask, x):
    model.eval(); out = []
    for i in range(0, len(ids), 128):
        a, m = trim(ids[i:i + 128], mask[i:i + 128])
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(a.to(DEV), m.to(DEV), x[i:i + 128].to(DEV)).float().cpu())
    return torch.cat(out).numpy()


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [42, 43, 44]
    d4_prov.assert_init(HUB_ID, "humanitarian")
    D = d4.data(); S, labels, y = D["S"], D["labels"], D["y"]; NC = len(labels); ALL = list(range(NC))
    fc = d4_prov.check_t2_cache("resnet50", D["ids"])                       # row order = T2 split image_id order; pretrained, frozen hub encoder
    IMG = {s: torch.tensor(np.load(ABL_ART / "d2_feats" / "resnet50" / f"humanitarian_{s}.npy"), dtype=torch.float32) for s in S}
    assert all(len(IMG[s]) == len(S[s]) and IMG[s].shape[1] == 2048 and torch.isfinite(IMG[s]).all() for s in S)
    tok = AutoTokenizer.from_pretrained(HUB_ID, use_fast=False, normalization=True)
    probe = tok.tokenize("Flood in #Houston! @NWS see https://t.co/abc")
    assert "HTTPURL" in probe and "@USER" in probe, "BERTweet normalisation did not run"
    X = {s: tuple(tok(list(S[s].tweet_text.fillna("")), padding="max_length", truncation=True, max_length=MAXLEN, return_tensors="pt")[k] for k in ("input_ids", "attention_mask")) for s in S}
    ytr = torch.tensor(y["train"]); n = len(ytr)
    cnt = np.bincount(y["train"], minlength=NC).astype("float32"); w = 1.0 / cnt; w /= w.sum() / NC
    for seed in seeds:
        t0 = time.time(); cc.seed_all(seed)
        model = Fusion(NC, IMG["train"].shape[1]).to(DEV)
        crit = nn.CrossEntropyLoss(weight=torch.tensor(w, device=DEV))
        rest = [p for nme, p in model.named_parameters() if not nme.startswith("text_encoder.")]
        opt = AdamW([{"params": model.text_encoder.parameters(), "lr": 1e-5}, {"params": rest, "lr": 1e-4}], weight_decay=WD)
        sched = lr_scheduler.SequentialLR(opt, [lr_scheduler.LinearLR(opt, 1e-6, 1.0, 3), lr_scheduler.CosineAnnealingLR(opt, T_max=E - 3, eta_min=1e-7)], milestones=[3])
        scaler = torch.amp.GradScaler("cuda"); g = torch.Generator().manual_seed(seed)
        best, best_ep, best_state, hist = -1.0, 0, None, []
        for ep in range(1, E + 1):
            model.train(); perm = torch.randperm(n, generator=g)
            for i in range(0, n, BATCH):
                b = perm[i:i + BATCH]; a, m = trim(X["train"][0][b], X["train"][1][b]); opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(a.to(DEV), m.to(DEV), IMG["train"][b].to(DEV))
                scaler.scale(crit(out.float(), ytr[b].to(DEV))).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
            sched.step()
            zd = logits(model, *X["dev"], IMG["dev"]); f = float(f1_score(y["dev"], zd.argmax(1), labels=ALL, average="macro", zero_division=0)); hist.append(f)
            print(f"{NAME} seed {seed} ep {ep}/{E}: dev macro-F1 {f:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if f > best:
                best, best_ep = f, ep; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        zd, zt = logits(model, *X["dev"], IMG["dev"]), logits(model, *X["test"], IMG["test"])
        row = M.save_run("T2", NAME, seed, labels, y["dev"], zd, y["test"], zt, extra=dict(best_epoch=best_ep, dev_history=hist, wall_clock_s=time.time() - t0, **dict(d4_prov.prov("humanitarian", HUB_ID), feature_cache=fc)))
        t = row["test"]
        print(f"DONE {NAME} seed {seed}: best ep {best_ep} dev macro-F1 {best:.4f} | test acc {t['acc']:.4f} macro-F1 {t['macro_f1']:.4f} wF1 {t['weighted_f1']:.4f} bal-acc {t['balanced_acc']:.4f} "
              f"top-2 {t['top2_acc']:.4f} MCC {t['mcc']:.4f}", flush=True)
        del model, opt; torch.cuda.empty_cache()
