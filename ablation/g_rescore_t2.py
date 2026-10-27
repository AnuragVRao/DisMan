"""G: re-score the existing Task 2 checkpoints (inference only, no retraining) with the extended metrics in extra_metrics.py.

For each config in CONFIGS and seeds 42/43/44: load artifacts/ablation/ckpt/t2_<config>_seed<k>.pt, predict the corrected humanitarian test split
(2237 rows, same cached B/32 features and tokenisation as t2_train.py), and write accuracy, macro/weighted F1, balanced accuracy, MCC, per-class
precision/recall/F1/ROC-AUC/PR-AUC and a bootstrap CI on macro-F1 to results/ablation/G/g_rescore_<config>_seed<k>.json (logits to
artifacts/ablation/rescore/). The re-computed accuracy and macro-F1 are checked against the stored result JSON (tolerance 2e-3: the stored runs
used fp16 autocast with a different batch size, so a prediction can flip on a near-tie).
"""
import json

import numpy as np
import torch
from transformers import DistilBertConfig, DistilBertTokenizerFast

from common import ABL_ART, ART, DATA, RES, SPLIT_DIR, cc, write_row
from extra_metrics import full_metrics, macro_f1_ci
from t2_models import Variant

CONFIGS = ("L4a", "L4", "image_only", "concat", "no_t2i", "text_only")   # L4a/L4 are the full DualStreamClassifier
PART = {"L4a": "A", "L4": "A", "image_only": "B", "concat": "B", "no_t2i": "B", "text_only": "B"}
SEEDS = (42, 43, 44)
OUT = ABL_ART / "rescore"


def no_miss():
    raise RuntimeError("cache miss")


@torch.no_grad()
def logits(model, X, ids, mask, device):
    model.eval(); out = []
    for i in range(0, len(X), 128):
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(model(X[i:i + 128].to(device), ids[i:i + 128].to(device), mask[i:i + 128].to(device)).float().cpu())
    return torch.cat(out).numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S = {s: cc.load_split("humanitarian", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "test")}
    labels = sorted(S["train"].label.unique()); l2i = {l: i for i, l in enumerate(labels)}; NC = len(labels)
    y = np.array([l2i[l] for l in S["test"].label])
    X = torch.tensor(cc.cached(ART / "emb" / "clip_humanitarian_test.npy", S["test"].image_id.tolist(), no_miss), dtype=torch.float32)
    tok = DistilBertTokenizerFast.from_pretrained(ART / "tokenizer")
    ids, mask = cc.tokenize(tok, S["test"]["text"].tolist(), 128)
    bcfg = DistilBertConfig.from_pretrained(ART / "distilbert_config")   # weights are overwritten by the checkpoint, so no pretrained download is needed
    OUT.mkdir(parents=True, exist_ok=True)
    for cfg in CONFIGS:
        for seed in SEEDS:
            model = (cc.DualStreamClassifier(NC, bert_config=bcfg) if cfg in ("L4a", "L4") else Variant(cfg, NC, bert_config=bcfg)).to(device)
            model.load_state_dict(torch.load(ABL_ART / "ckpt" / f"t2_{cfg}_seed{seed}.pt", map_location=device))
            z = logits(model, X, ids, mask, device); pred = z.argmax(1)
            m = full_metrics(y, pred, labels, z)
            st = json.load(open(RES / PART[cfg] / f"t2_{cfg}_seed{seed}.json"))
            d = max(abs(m["acc"] - st["test_acc"]), abs(m["macro_f1"] - st["test_macro_f1"]))
            row = dict(task="T2", tier="gpu-small", model=cfg, test=m, macro_f1_ci=macro_f1_ci(y, pred, NC),
                       stored_test_acc=st["test_acc"], stored_test_macro_f1=st["test_macro_f1"], max_abs_diff_vs_stored=d, pred=pred.tolist())
            write_row("G", f"g_rescore_{cfg}", seed, row)
            np.save(OUT / f"{cfg}_seed{seed}.npy", z)
            print(f"{cfg:10s} seed {seed}: acc {m['acc']:.4f} macro-F1 {m['macro_f1']:.4f} bal-acc {m['balanced_acc']:.4f} MCC {m['mcc']:.4f} "
                  f"ROC-AUC {m['macro_roc_auc']:.4f} PR-AUC {m['macro_pr_auc']:.4f} | CI [{row['macro_f1_ci']['ci95_low']:.3f}, "
                  f"{row['macro_f1_ci']['ci95_high']:.3f}] | |diff vs stored| {d:.1e}", flush=True)
            assert d < 2e-3, f"{cfg} seed {seed} does not reproduce the stored result"


if __name__ == "__main__":
    main()
