"""Day 4 final round B5: leave-one-disaster-out for Task 1 with BERTweet, ONE seed (42) per fold, labelled single-seed (SELECTION_RULE_DAY4.md rule 23).
Fold definitions as ablation/c2_lodo.py: train on the OTHER 6 events' TRAIN rows, test on the held-out event's TEST rows; the held-out event contributes no train / dev rows (asserted by event_name, tweet_id and image_id).
Recipe = the Task 1 BERTweet recipe (d4_t1_bertweet.py: raw tweet_text with BERTweet normalisation, max length 128, lr 2e-5, 4 epochs, batch 32, weighted CE), the epoch is selected on the DEV rows of the other events.
Per-event macro-F1 averages the classes present in that event's labels or predictions (v2's per-event convention); the fixed 2-class macro-F1 is stored too. In-distribution = the standard BERTweet run (seed 42, trained on all events) on the same held-out rows.
TF-IDF + LR is recomputed with c2_lodo's t1_model for the same table. Writes results/ablation/D4_T1_lodo.{json,md}."""
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import d4_prov
import d4_t1_data as d4
import metrics_all as M
from common import RES, cc

HUB_ID, MAXLEN, LR, EPOCHS, BATCH, SEED = "vinai/bertweet-base", 128, 2e-5, 4, 32, 42
DEV = torch.device("cuda")


def trim(ids, mask):
    L = int(mask.sum(1).max()); return ids[:, :L], mask[:, :L]


@torch.no_grad()
def logits(model, ids, mask):
    model.eval(); out = []
    for i in range(0, len(ids), 128):
        a, m = trim(ids[i:i + 128], mask[i:i + 128])
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(input_ids=a.to(DEV), attention_mask=m.to(DEV)).logits.float().cpu())
    return torch.cat(out).numpy()


def score(y, pred):
    present = sorted(set(y) | set(pred))
    return dict(n=int(len(y)), acc=float((y == pred).mean()), macro_f1=float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
                macro_f1_fixed=float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)), weighted_f1=float(f1_score(y, pred, labels=[0, 1], average="weighted", zero_division=0)))


if __name__ == "__main__":
    d4_prov.assert_init(HUB_ID, "informative")
    D = d4.data(); S = D["S"]; y = D["y"]
    tok = AutoTokenizer.from_pretrained(HUB_ID, use_fast=False, normalization=True)
    assert "HTTPURL" in tok.tokenize("see https://t.co/abc") and "@USER" in tok.tokenize("hi @NWS"), "BERTweet normalisation did not run"
    X = {s: tuple(tok(list(S[s].tweet_text.fillna("")), padding="max_length", truncation=True, max_length=MAXLEN, return_tensors="pt")[k] for k in ("input_ids", "attention_mask")) for s in S}
    events = sorted(S["test"].event_name.unique()); assert len(events) == 7, events
    ind = M.load_run("T1", "bertweet", SEED); ind_pred = np.array(ind["probs_test"]).argmax(1)
    rows, t_start = [], time.time()
    for e in events:
        t0 = time.time(); cc.seed_all(SEED)
        fit = (S["train"].event_name != e).to_numpy(); dv = (S["dev"].event_name != e).to_numpy(); held = (S["test"].event_name == e).to_numpy()
        assert (S["train"][fit].event_name == e).sum() == 0 and (S["dev"][dv].event_name == e).sum() == 0
        assert not set(S["train"][fit].tweet_id) & set(S["test"][held].tweet_id) and not set(S["train"][fit].image_id) & set(S["test"][held].image_id), "LEAK"
        ytr = torch.tensor(y["train"][fit]); idx = np.where(fit)[0]; n = len(ytr)
        model = AutoModelForSequenceClassification.from_pretrained(HUB_ID, num_labels=2).to(DEV)
        cnt = np.bincount(y["train"][fit], minlength=2).astype("float32"); w = 1.0 / cnt; w /= w.sum() / 2
        crit = nn.CrossEntropyLoss(weight=torch.tensor(w, device=DEV)); opt = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        steps = EPOCHS * ((n + BATCH - 1) // BATCH); sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        scaler = torch.amp.GradScaler("cuda"); g = torch.Generator().manual_seed(SEED); best, best_ep, best_state = -1.0, 0, None
        for ep in range(1, EPOCHS + 1):
            model.train(); perm = torch.randperm(n, generator=g)
            for i in range(0, n, BATCH):
                b = perm[i:i + BATCH]; ii = idx[b.numpy()]; a, m = trim(X["train"][0][ii], X["train"][1][ii]); opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(input_ids=a.to(DEV), attention_mask=m.to(DEV)).logits
                scaler.scale(crit(out.float(), ytr[b].to(DEV))).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
            zd = logits(model, X["dev"][0][dv], X["dev"][1][dv]); f = float(f1_score(y["dev"][dv], zd.argmax(1), average="weighted"))
            if f > best:
                best, best_ep = f, ep; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        zt = logits(model, X["test"][0][held], X["test"][1][held]); pred = zt.argmax(1)
        r = dict(event=e, n_train=int(n), n_dev=int(dv.sum()), best_epoch=best_ep, dev_weighted_f1=best, held_out=score(y["test"][held], pred), in_dist=score(y["test"][held], ind_pred[held]), minutes=(time.time() - t0) / 60)
        rows.append(r); print(f"{e}: held-out macro-F1 {r['held_out']['macro_f1']:.3f} / in-dist {r['in_dist']['macro_f1']:.3f} (n={r['held_out']['n']}, {r['minutes']:.1f} min; elapsed {(time.time() - t_start) / 60:.1f} min)", flush=True)
        del model, opt; torch.cuda.empty_cache()
    # TF-IDF + LR (c2_lodo's model) for the same table
    import c2_lodo
    tf = {}
    for e, m, indist, _ in c2_lodo.run_t1():
        tf[e] = dict(held_out=m, in_dist=indist)
    avg = lambda k, which: float(np.mean([r[which][k] for r in rows]))
    out = dict(seed=SEED, single_seed=True, recipe="BERTweet, 4 epochs, lr 2e-5, weighted CE, epoch by dev weighted F1 on the other events' dev rows", rows=rows, tfidf=tf,
               average=dict(bertweet_held_out=avg("macro_f1", "held_out"), bertweet_in_dist=avg("macro_f1", "in_dist"),
                            tfidf_held_out=float(np.mean([tf[e]["held_out"]["macro_f1"] for e in events])), tfidf_in_dist=float(np.mean([tf[e]["in_dist"]["macro_f1"] for e in events]))),
               provenance=d4_prov.prov("informative", HUB_ID), total_minutes=(time.time() - t_start) / 60)
    (RES / "D4_T1_lodo.json").write_text(json.dumps(out, indent=1))
    md = "# Task 1 leave-one-disaster-out (macro-F1 held out / in-distribution; single seed 42 for BERTweet)\n\n| Held-out event | T1 test rows | BERTweet | TF-IDF + LR |\n|---|---|---|---|\n"
    for r in rows:
        e = r["event"]; md += f"| {e} | {r['held_out']['n']} | {r['held_out']['macro_f1']:.3f} / {r['in_dist']['macro_f1']:.3f} | {tf[e]['held_out']['macro_f1']:.3f} / {tf[e]['in_dist']['macro_f1']:.3f} |\n"
    a = out["average"]; md += f"| Average over events | {sum(r['held_out']['n'] for r in rows)} | {a['bertweet_held_out']:.3f} / {a['bertweet_in_dist']:.3f} | {a['tfidf_held_out']:.3f} / {a['tfidf_in_dist']:.3f} |\n"
    (RES / "D4_T1_lodo.md").write_text(md, encoding="utf-8"); print(md)
