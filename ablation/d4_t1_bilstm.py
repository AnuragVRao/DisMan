"""Day 4, Task 1: BiLSTM + GloVe Twitter 200d (frozen embeddings read from artifacts/glove/glove.twitter.27B.zip, member glove.twitter.27B.200d.txt).
Input = crisis_common.clean_tweet text (the TF-IDF input), '@user' mapped to GloVe's <user>. 1 BiLSTM layer (hidden 128 per direction), max-pool over time, dropout 0.3,
inverse-frequency CE, AdamW lr 1e-3, 10 epochs, batch 64, best epoch on dev weighted F1, seeds 42/43/44."""
import re
import sys
import zipfile
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

import d4_t1_data as d4
import metrics_all as M
from common import ART, cc

DEV = torch.device("cuda")
ZIP, MEMBER, DIM = ART / "glove" / "glove.twitter.27B.zip", "glove.twitter.27B.200d.txt", 200
EPOCHS, BATCH, LR, MAXLEN = 10, 64, 1e-3, 60
TOK = re.compile(r"<user>|[#\w']+|[^\w\s]")


def tokens(t):
    return TOK.findall(t.lower().replace("@user", "<user>"))


class Net(nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.tensor(emb), freeze=True, padding_idx=0)
        self.lstm = nn.LSTM(DIM, 128, batch_first=True, bidirectional=True)
        self.drop, self.out = nn.Dropout(0.3), nn.Linear(256, 2)

    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        h = h.masked_fill((x == 0).unsqueeze(-1), -1e4).max(1).values       # max-pool over time, ignoring padding
        return self.out(self.drop(h))


def build(S):
    toks = {s: [tokens(t) for t in S[s].text] for s in S}
    vocab = [w for w, c in Counter(w for ts in toks["train"] for w in ts).items() if c >= 2]
    idx = {w: i + 2 for i, w in enumerate(vocab)}                              # 0 pad, 1 unk
    emb = np.zeros((len(idx) + 2, DIM), dtype="float32"); emb[1] = np.random.default_rng(0).normal(0, 0.1, DIM); found = 0
    with zipfile.ZipFile(ZIP) as z, z.open(MEMBER) as f:
        for line in f:
            w, _, rest = line.decode("utf8").partition(" ")
            if w in idx:
                emb[idx[w]] = np.asarray(rest.split(), dtype="float32"); found += 1
    print(f"vocab {len(idx)}, found in GloVe {found} ({found / len(idx):.1%})", flush=True)
    def enc(ts):
        x = torch.zeros(len(ts), MAXLEN, dtype=torch.long)
        for i, t in enumerate(ts):
            ids = [idx.get(w, 1) for w in t][:MAXLEN]; x[i, :len(ids)] = torch.tensor(ids)
        return x
    return {s: enc(toks[s]) for s in toks}, emb


@torch.no_grad()
def logits(model, x):
    model.eval()
    return torch.cat([model(x[i:i + 512].to(DEV)).float().cpu() for i in range(0, len(x), 512)]).numpy()


if __name__ == "__main__":
    D = d4.data(); X, emb = build(D["S"]); ytr = torch.tensor(D["y"]["train"]); n = len(ytr)
    for seed in (42, 43, 44):
        cc.seed_all(seed); model = Net(emb).to(DEV)
        cnt = np.bincount(D["y"]["train"], minlength=2).astype("float32"); w = 1.0 / cnt; w /= w.sum() / 2
        crit = nn.CrossEntropyLoss(weight=torch.tensor(w, device=DEV)); opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4)
        g = torch.Generator().manual_seed(seed); best, best_ep, best_state = -1.0, 0, None
        for ep in range(1, EPOCHS + 1):
            model.train(); perm = torch.randperm(n, generator=g)
            for i in range(0, n, BATCH):
                b = perm[i:i + BATCH]; opt.zero_grad(set_to_none=True)
                crit(model(X["train"][b].to(DEV)), ytr[b].to(DEV)).backward(); opt.step()
            f = float(f1_score(D["y"]["dev"], logits(model, X["dev"]).argmax(1), average="weighted"))
            if f > best:
                best, best_ep = f, ep; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        row = M.save_run("T1", "bilstm_glove", seed, d4.LABELS, D["y"]["dev"], logits(model, X["dev"]), D["y"]["test"], logits(model, X["test"]), extra=dict(best_epoch=best_ep))
        t = row["test"]
        print(f"bilstm seed {seed}: ep {best_ep} dev wF1 {best:.4f} | test acc {t['acc']:.4f} wF1 {t['weighted_f1']:.4f} macroF1 {t['macro_f1']:.4f} MCC {t['mcc']:.4f} AUC {t['roc_auc']:.4f}", flush=True)
