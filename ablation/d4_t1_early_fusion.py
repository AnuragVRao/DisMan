"""Day 4, Task 1 early fusion (guide's core model): BERTweet (fine-tuned) [<s>/CLS] state (768) concatenated with a frozen cached image feature -> MLP -> 2 classes.
Image feature: resnet50 (2048-d, guide's model, "BERTweet (fine-tuned) + ResNet50 (frozen)") or clip_l14 (768-d, optional CLIP concat). Everything else follows d4_t1_bertweet.py:
raw tweet_text with BERTweet normalisation, max_len 128, dynamic padding, one AdamW lr 2e-5 for ALL parameters (encoder and fusion head; the pre-registered 'same LR recipe'),
4 epochs, batch 32, fp16 autocast, inverse-frequency weighted CE, best epoch on dev weighted F1, seeds 42/43/44. Run alone on the GPU.
Fusion head: image LayerNorm(d) -> Linear(d,256) -> ReLU -> Dropout(0.2); concat [dropout(CLS) | image] (1024) -> Linear(1024,256) -> ReLU -> Dropout(0.2) -> Linear(256,2).
Usage: python d4_t1_early_fusion.py resnet50|clip_l14 [seed ...]"""
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

import d4_prov
import d4_t1_data as d4
import metrics_all as M
from common import ABL_ART, cc

MAXLEN, LR, EPOCHS, BATCH = 128, 2e-5, 4, 32
HUB_ID = "vinai/bertweet-base"
DEV = torch.device("cuda")


class Fusion(nn.Module):
    def __init__(self, d_img):
        super().__init__()
        self.enc = AutoModel.from_pretrained(HUB_ID)
        self.img = nn.Sequential(nn.LayerNorm(d_img), nn.Linear(d_img, 256), nn.ReLU(), nn.Dropout(0.2))
        self.drop = nn.Dropout(0.1)
        self.head = nn.Sequential(nn.Linear(768 + 256, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 2))

    def forward(self, ids, mask, x):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.head(torch.cat([self.drop(h), self.img(x)], dim=-1))


def trim(ids, mask):
    L = int(mask.sum(1).max()); return ids[:, :L], mask[:, :L]


@torch.no_grad()
def predict(model, ids, mask, x):
    model.eval(); out = []
    for i in range(0, len(ids), 128):
        a, m = trim(ids[i:i + 128], mask[i:i + 128])
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(a.to(DEV), m.to(DEV), x[i:i + 128].to(DEV)).float().cpu())
    return torch.cat(out).numpy()


if __name__ == "__main__":
    feat = sys.argv[1]; seeds = [int(s) for s in sys.argv[2:]] or [42, 43, 44]
    assert feat in ("resnet50", "clip_l14")
    d4_prov.assert_init(HUB_ID, "informative")
    D = d4.data(); S = D["S"]; name = f"early_bertweet_{feat}"
    fdir = ABL_ART / "d4_feats" / feat
    assert json.load(open(fdir / "meta.json"))["model"] in d4_prov.HUB, "feature encoder is not a pretrained hub model"
    IMG = {}
    for s in S:
        assert json.load(open(fdir / f"informative_{s}.ids.json")) == D["ids"][s], "feature rows are not in T1 split image_id order"
        IMG[s] = torch.tensor(np.load(fdir / f"informative_{s}.npy"), dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(HUB_ID, use_fast=False, normalization=True)
    probe = tok.tokenize("Flood in #Houston! @NWS see https://t.co/abc")
    assert "HTTPURL" in probe and "@USER" in probe, "BERTweet normalisation did not run"
    enc = lambda texts: tuple(tok(list(texts), padding="max_length", truncation=True, max_length=MAXLEN, return_tensors="pt")[k] for k in ("input_ids", "attention_mask"))
    X = {s: enc(S[s].tweet_text.fillna("")) for s in S}
    ytr = torch.tensor(D["y"]["train"]); n = len(ytr)
    for seed in seeds:
        t0 = time.time(); cc.seed_all(seed)
        model = Fusion(IMG["train"].shape[1]).to(DEV)
        cnt = np.bincount(D["y"]["train"], minlength=2).astype("float32"); w = 1.0 / cnt; w /= w.sum() / 2
        crit = nn.CrossEntropyLoss(weight=torch.tensor(w, device=DEV))
        opt = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        steps = EPOCHS * ((n + BATCH - 1) // BATCH); sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        scaler = torch.amp.GradScaler("cuda"); g = torch.Generator().manual_seed(seed)
        best, best_ep, best_state = -1.0, 0, None
        for ep in range(1, EPOCHS + 1):
            model.train(); perm = torch.randperm(n, generator=g)
            for i in range(0, n, BATCH):
                b = perm[i:i + BATCH]; a, m = trim(X["train"][0][b], X["train"][1][b]); opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(a.to(DEV), m.to(DEV), IMG["train"][b].to(DEV))
                scaler.scale(crit(logits.float(), ytr[b].to(DEV))).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
            zd = predict(model, *X["dev"], IMG["dev"]); f = float(f1_score(D["y"]["dev"], zd.argmax(1), average="weighted"))
            print(f"{name} seed {seed} ep {ep}: dev weighted F1 {f:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if f > best:
                best, best_ep = f, ep; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        zd, zt = predict(model, *X["dev"], IMG["dev"]), predict(model, *X["test"], IMG["test"])
        pv = d4_prov.prov("informative", HUB_ID, feature=feat)
        row = M.save_run("T1", name, seed, d4.LABELS, D["y"]["dev"], zd, D["y"]["test"], zt, extra=dict(best_epoch=best_ep, wall_clock_s=time.time() - t0, **pv))
        t = row["test"]
        print(f"DONE {name} seed {seed}: best ep {best_ep} dev wF1 {best:.4f} | test acc {t['acc']:.4f} wF1 {t['weighted_f1']:.4f} macroF1 {t['macro_f1']:.4f} MCC {t['mcc']:.4f} AUC {t['roc_auc']:.4f}", flush=True)
        del model, opt; torch.cuda.empty_cache()
