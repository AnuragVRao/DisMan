"""Day 5 preprocessing A/B on Task 1 BERTweet. Same recipe as d4_t1_bertweet.py (bertweet-base, max_len 128, lr 2e-5, 4 epochs, batch 32, weighted CE, epoch picked on dev); only the input text changes.
One flag per run:  --norm    preprocessing/ clean_text_bertweet, BERTweet normalization on   (arm A)
                   --nonorm  preprocessing/ clean_text_bertweet, normalization off          (arm B)
                   --c       minimal cleaning from d5_prep_c.py, normalization on            (arm C)
                   --raw     raw tweet_text, normalization on (the D4 baseline recipe, extra seeds)
Usage: python d5_t1_bertweet_prep.py --c 42 43 44. Runs go to results/ablation/D5_prep/; D4 files are never touched."""
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import d4_t1_data as d4
import d4_prov
import metrics_all as M
from common import cc

import os, pandas as pd
from pathlib import Path
SHEK = Path(os.environ.get("SHEK_DIR", Path(__file__).resolve().parent.parent / "preprocessing" / "preprocessed_text_task1"))
VARIANT = "nonorm" if "--nonorm" in sys.argv else ("C" if "--c" in sys.argv else ("raw" if "--raw" in sys.argv else "norm"))
M.RUN_DIR = Path(M.RUN_DIR).parent / "D5_prep"
NOWEIGHT = False
SAVE_UNUSED = "--noweight" in sys.argv          # post-hoc arm (SELECTION_RULE_DAY4.md rule 10): plain unweighted cross-entropy
SAVE = False                  # step 4: retrain the dev-selected seed, keep the weights (artifacts/task1_bertweet) and a temperature fitted on dev
NAME, MAXLEN, LR, EPOCHS, BATCH = "bertweet_noweight" if NOWEIGHT else ("bertweet_prepC" if VARIANT == "C" else "bertweet_raw" if VARIANT == "raw" else "bertweet_shek_" + VARIANT), 128, 2e-5, 4, 32
DEV = torch.device("cuda")


def encode(tok, texts):
    e = tok(list(texts), padding="max_length", truncation=True, max_length=MAXLEN, return_tensors="pt")
    return e["input_ids"], e["attention_mask"]


def trim(ids, mask):
    """Dynamic padding: cut the batch to its longest sequence (same outputs as full padding, attention-masked; much faster on short tweets)."""
    L = int(mask.sum(1).max())
    return ids[:, :L], mask[:, :L]


@torch.no_grad()
def predict(model, ids, mask):
    model.eval(); out = []
    for i in range(0, len(ids), 128):
        a, m = trim(ids[i:i + 128], mask[i:i + 128])
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(input_ids=a.to(DEV), attention_mask=m.to(DEV)).logits.float().cpu())
    return torch.cat(out).numpy()


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:] if not s.startswith("--")] or [42, 43, 44]
    D = d4.data(); S = D["S"]
    tok = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False, normalization=(VARIANT != "nonorm"))
    probe = tok.tokenize("Flood in #Houston! @NWS see https://t.co/abc 😢")
    print("sanity:", probe, flush=True)
    if VARIANT != "nonorm":
        assert "HTTPURL" in probe and "@USER" in probe, "BERTweet normalisation did not run"
    T = {}
    for sp in S:
        if VARIANT == "raw":
            T[sp] = S[sp].tweet_text.fillna("").values; continue
        b = pd.read_csv((SHEK / f"task1_text_{sp}.tsv") if VARIANT != "C" else (Path(__file__).resolve().parent.parent / "results" / "ablation" / "D5_prep" / f"task1_text_C_{sp}.tsv"), sep="	", dtype=str).drop_duplicates(subset=["image_id"]).set_index("image_id")
        assert set(S[sp].image_id) <= set(b.index), (sp, "image_ids missing in branch file")
        assert VARIANT == "C" or (b.loc[S[sp].image_id, "label_text"].str.strip().values == S[sp].label.values).all(), (sp, "labels differ")
        T[sp] = b.loc[S[sp].image_id, "text" if VARIANT == "C" else "clean_text_bertweet"].fillna("").values
        print(sp, "rows", len(T[sp]), "text changed vs raw:", float((T[sp] != S[sp].tweet_text.fillna("").values).mean()), flush=True)
    X = {s: encode(tok, T[s]) for s in S}
    ytr = torch.tensor(D["y"]["train"])
    for seed in seeds:
        t0 = time.time(); cc.seed_all(seed)
        model = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=2).to(DEV)
        cnt = np.bincount(D["y"]["train"], minlength=2).astype("float32"); w = 1.0 / cnt; w /= w.sum() / 2
        if NOWEIGHT:
            w = np.ones(2, dtype="float32")
        crit = nn.CrossEntropyLoss(weight=torch.tensor(w, device=DEV))
        opt = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        n = len(ytr); steps = EPOCHS * ((n + BATCH - 1) // BATCH)
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        scaler = torch.amp.GradScaler("cuda"); g = torch.Generator().manual_seed(seed)
        best, best_ep, best_state = -1.0, 0, None
        for ep in range(1, EPOCHS + 1):
            model.train(); perm = torch.randperm(n, generator=g)
            for i in range(0, n, BATCH):
                b = perm[i:i + BATCH]
                opt.zero_grad(set_to_none=True)
                a, m = trim(X["train"][0][b], X["train"][1][b])
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(input_ids=a.to(DEV), attention_mask=m.to(DEV)).logits
                loss = crit(logits.float(), ytr[b].to(DEV))
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
            zd = predict(model, *X["dev"])
            f = float(f1_score(D["y"]["dev"], zd.argmax(1), average="weighted"))
            print(f"seed {seed} ep {ep}: dev weighted F1 {f:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if f > best:
                best, best_ep = f, ep; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        zd, zt = predict(model, *X["dev"]), predict(model, *X["test"])
        row = M.save_run("T1", NAME, seed, d4.LABELS, D["y"]["dev"], zd, D["y"]["test"], zt, extra=dict(best_epoch=best_ep, wall_clock_s=time.time() - t0, **d4_prov.prov("informative", "vinai/bertweet-base")))
        t = row["test"]
        if SAVE:
            import json as _json
            from common import ART
            out = ART / "task1_bertweet"; out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out); tok.save_pretrained(out)
            T = M.fit_temperature(zd, D["y"]["dev"])
            (out / "t1_meta.json").write_text(_json.dumps(dict(labels=d4.LABELS, seed=seed, best_epoch=best_ep, dev_weighted_f1=best, temperature_dev=T, max_len=MAXLEN,
                                                               init_from="vinai/bertweet-base", normalization=True, input="raw tweet_text (not clean_tweet)", train_split_sha256=row["train_split_sha256"]), indent=1))
            print(f"saved {out}; dev temperature {T:.3f}", flush=True)
        print(f"DONE seed {seed}: best ep {best_ep} dev wF1 {best:.4f} | test acc {t['acc']:.4f} wF1 {t['weighted_f1']:.4f} macroF1 {t['macro_f1']:.4f} "
              f"MCC {t['mcc']:.4f} AUC {t['roc_auc']:.4f}", flush=True)
        del model, opt; torch.cuda.empty_cache()
