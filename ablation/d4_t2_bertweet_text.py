"""Day 4 step 5b, Task 2: BERTweet text-only on the 8 humanitarian classes (`label_image`), inverse-frequency weighted CE. Two arms (SELECTION_RULE_DAY4.md rule 13):
  matched  = the recipe of the DistilBERT text-only run (t2_train.py --variant text_only, L4a config): 10 epochs, batch 32, AdamW wd 1e-4, encoder lr 1e-5 / head lr 1e-4,
             3 linear warm-up epochs then cosine (epoch-level), grad clip 1.0, fp16 autocast, max length 128, select on dev macro-F1 (strict >).
  rule8    = rule 8's BERTweet recipe: 4 epochs, one lr 2e-5, 6% linear warm-up + linear decay (step-level), otherwise the same.
Head = Linear(768,256)+LayerNorm on the token states, first token, MLP(256,256,8), dropout 0.2 (the Variant text_only head). Raw tweet_text with BERTweet normalisation, dynamic padding.
Initialised from the hub id only (rule 11). Usage: python d4_t2_bertweet_text.py matched|rule8 [seed ...]"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW, lr_scheduler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

import d4_prov
import d4_t2_data as d4
import metrics_all as M
from common import cc

HUB_ID, MAXLEN, BATCH, WD = "vinai/bertweet-base", 128, 32, 1e-4
DEV = torch.device("cuda")
ARMS = {"matched": dict(epochs=10, name="bertweet_text_matched"), "rule8": dict(epochs=4, name="bertweet_text_rule8")}


class TextOnly(nn.Module):
    def __init__(self, nc, dropout=0.2):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(HUB_ID)
        self.txt_proj = nn.Sequential(nn.Linear(768, 256), nn.LayerNorm(256))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, nc))

    def forward(self, ids, mask):
        return self.classifier(self.txt_proj(self.text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state)[:, 0])


def trim(ids, mask):
    L = int(mask.sum(1).max()); return ids[:, :L], mask[:, :L]


@torch.no_grad()
def logits(model, ids, mask):
    model.eval(); out = []
    for i in range(0, len(ids), 128):
        a, m = trim(ids[i:i + 128], mask[i:i + 128])
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(a.to(DEV), m.to(DEV)).float().cpu())
    return torch.cat(out).numpy()


if __name__ == "__main__":
    arm = sys.argv[1]; seeds = [int(s) for s in sys.argv[2:]] or [42, 43, 44]; E = ARMS[arm]["epochs"]; name = ARMS[arm]["name"]
    d4_prov.assert_init(HUB_ID, "humanitarian")
    D = d4.data(); S, labels, y = D["S"], D["labels"], D["y"]; NC = len(labels); ALL = list(range(NC))
    tok = AutoTokenizer.from_pretrained(HUB_ID, use_fast=False, normalization=True)
    probe = tok.tokenize("Flood in #Houston! @NWS see https://t.co/abc")
    assert "HTTPURL" in probe and "@USER" in probe, "BERTweet normalisation did not run"
    X = {s: tuple(tok(list(S[s].tweet_text.fillna("")), padding="max_length", truncation=True, max_length=MAXLEN, return_tensors="pt")[k] for k in ("input_ids", "attention_mask")) for s in S}
    ytr = torch.tensor(y["train"]); n = len(ytr)
    cnt = np.bincount(y["train"], minlength=NC).astype("float32"); w = 1.0 / cnt; w /= w.sum() / NC
    for seed in seeds:
        t0 = time.time(); cc.seed_all(seed)
        model = TextOnly(NC).to(DEV)
        crit = nn.CrossEntropyLoss(weight=torch.tensor(w, device=DEV))
        if arm == "matched":
            opt = AdamW([{"params": model.text_encoder.parameters(), "lr": 1e-5}, {"params": list(model.txt_proj.parameters()) + list(model.classifier.parameters()), "lr": 1e-4}], weight_decay=WD)
            sched = lr_scheduler.SequentialLR(opt, [lr_scheduler.LinearLR(opt, 1e-6, 1.0, 3), lr_scheduler.CosineAnnealingLR(opt, T_max=E - 3, eta_min=1e-7)], milestones=[3]); per_step = False
        else:
            opt = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
            steps = E * ((n + BATCH - 1) // BATCH); sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps); per_step = True
        scaler = torch.amp.GradScaler("cuda"); g = torch.Generator().manual_seed(seed)
        best, best_ep, best_state, hist = -1.0, 0, None, []
        for ep in range(1, E + 1):
            model.train(); perm = torch.randperm(n, generator=g)
            for i in range(0, n, BATCH):
                b = perm[i:i + BATCH]; a, m = trim(X["train"][0][b], X["train"][1][b]); opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(a.to(DEV), m.to(DEV))
                scaler.scale(crit(out.float(), ytr[b].to(DEV))).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
                if per_step:
                    sched.step()
            if not per_step:
                sched.step()
            zd = logits(model, *X["dev"]); f = float(f1_score(y["dev"], zd.argmax(1), labels=ALL, average="macro", zero_division=0)); hist.append(f)
            print(f"{name} seed {seed} ep {ep}/{E}: dev macro-F1 {f:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if f > best:
                best, best_ep = f, ep; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        zd, zt = logits(model, *X["dev"]), logits(model, *X["test"])
        row = M.save_run("T2", name, seed, labels, y["dev"], zd, y["test"], zt, extra=dict(best_epoch=best_ep, dev_history=hist, wall_clock_s=time.time() - t0, arm=arm,
                                                                                         **d4_prov.prov("humanitarian", HUB_ID)))
        t = row["test"]
        print(f"DONE {name} seed {seed}: best ep {best_ep} dev macro-F1 {best:.4f} | test acc {t['acc']:.4f} macro-F1 {t['macro_f1']:.4f} wF1 {t['weighted_f1']:.4f} bal-acc {t['balanced_acc']:.4f} "
              f"top-2 {t['top2_acc']:.4f} MCC {t['mcc']:.4f}", flush=True)
        del model, opt; torch.cuda.empty_cache()
