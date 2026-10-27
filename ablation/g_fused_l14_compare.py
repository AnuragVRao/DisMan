"""G: the pre-registered fused check (ablation/SELECTION_RULE.md addendum): `none` against `inv` on L4a-L14, seeds 42, 43, 44.

L4a-L14 = the L4a recipe (pretrained DistilBERT, AMP, cached features, 10 epochs, batch 32, lr 1e-4, checkpoint = best dev macro-F1, test scored once)
with the CLIP L/14 768-d L2-normalised image vector. It is NOT a re-run of L4a. Both arms were trained by ablation/t2_train.py (--feat l14 --loss inv|none --part G),
the same code path as the L4a runs. This script re-scores the saved checkpoints on dev and test (inference only), checks the re-scored dev/test macro-F1 against the
values the training run stored, and reports macro-F1, balanced accuracy and minority recall per arm (seed mean +- seed std, plus the test-sampling 95% CI) and the
paired bootstrap of none - inv:
  DEV  (2000 resamples, same indices as the imbalance study)  -> the only version eligible for selection
  TEST (10,000 resamples)                                     -> REPORTING ONLY, test-sampling CIs, separate from the seed std
Verdict words: 'better'/'worse' only if the paired CI excludes 0 by more than 0.002, 'borderline' within 0.002 of zero, otherwise 'not distinguishable';
one rule for every comparison (also the image-only imbalance study): a CI that excludes 0 gets better/worse/borderline, and only a CI that INCLUDES 0 on the
injured-or-dead class (8 dev / 10 test rows) is 'not resolvable'. Each recall comparison records its row count. No multiplicity correction.
Writes results/ablation/G/g_fused_L4a-L14_none_vs_inv_seed42.json.
"""
import json

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from transformers import DistilBertConfig, DistilBertTokenizerFast

import bootstrap_utils as bu
from common import ABL_ART, ART, DATA, RES, SPLIT_DIR, cc, write_row
from extra_metrics import full_metrics, macro_f1_ci

SEEDS, ARMS = (42, 43, 44), ("inv", "none")
MISS, INJ, VEH = "Missing or found people", "Injured or dead people", "Vehicle damage"


def no_miss():
    raise RuntimeError("cache miss")


def verdict(r):
    if r["ci95_low"] > 0:
        return "borderline" if r["ci95_low"] < 0.002 else "better"
    if r["ci95_high"] < 0:
        return "borderline" if r["ci95_high"] > -0.002 else "worse"
    return "not distinguishable"


@torch.no_grad()
def logits(model, X, ids, mask, device):
    model.eval(); out = []
    for i in range(0, len(X), 128):
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(model(X[i:i + 128].to(device), ids[i:i + 128].to(device), mask[i:i + 128].to(device)).float().cpu())
    return torch.cat(out).numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S = {s: cc.load_split("humanitarian", s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "dev", "test")}
    labels = sorted(S["train"].label.unique()); l2i = {l: i for i, l in enumerate(labels)}; K = len(labels); ALL = list(range(K))
    y = {s: np.array([l2i[l] for l in S[s].label]) for s in ("dev", "test")}
    X = {s: torch.tensor(cc.cached(ABL_ART / "clip_l14" / f"clip_humanitarian_{s}.npy", S[s].image_id.tolist(), no_miss), dtype=torch.float32) for s in ("dev", "test")}
    tok = DistilBertTokenizerFast.from_pretrained(ART / "tokenizer"); bcfg = DistilBertConfig.from_pretrained(ART / "distilbert_config")
    T = {s: cc.tokenize(tok, S[s]["text"].tolist(), 128) for s in ("dev", "test")}
    Z = {}
    for arm in ARMS:
        for seed in SEEDS:
            name = f"t2_L4a-L14_{arm}"
            st = json.load(open(RES / "G" / f"{name}_seed{seed}.json"))
            assert st["flags"]["feat"] == "l14" and st["flags"]["loss"] == arm and st["flags"]["bert"] == "pretrained" and st["n_params"] > 0
            m = cc.DualStreamClassifier(K, bert_config=bcfg, img_dim=768).to(device)
            m.load_state_dict(torch.load(ABL_ART / "ckpt" / f"{name}_seed{seed}.pt", map_location=device))
            zd, zt = (logits(m, X[s], *T[s], device) for s in ("dev", "test"))
            pd_, pt_ = zd.argmax(1), zt.argmax(1)
            d_f1 = float(f1_score(y["dev"], pd_, labels=ALL, average="macro", zero_division=0)); t_f1 = float(f1_score(y["test"], pt_, labels=ALL, average="macro", zero_division=0))
            diff = max(abs(d_f1 - st["dev_macro_f1"]), abs(t_f1 - st["test_macro_f1"]))
            print(f"{arm:5s} seed {seed}: dev F1 {d_f1:.4f} (stored {st['dev_macro_f1']:.4f}) test F1 {t_f1:.4f} (stored {st['test_macro_f1']:.4f}) best epoch {st['best_epoch']} | max |diff| {diff:.1e}", flush=True)
            assert diff < 2e-3, f"{arm} seed {seed}: re-scored != stored"
            Z[(arm, seed)] = dict(zd=zd, zt=zt, pd=pd_, pt=pt_, st=st, dev_f1=d_f1, test_f1=t_f1)
    out = dict(task="T2", model="L4a-L14", n_boot=dict(dev=2000, test=10000), arms={}, pairs={})
    didx, tidx = bu.resample_idx(len(y["dev"]), 2000), bu.resample_idx(len(y["test"]), 10000)
    rec = lambda yy, p, c: float(recall_score(yy, p, labels=[c], average="macro", zero_division=0))
    point = {}
    for arm in ARMS:
        pm = {}
        for split, key in (("dev", "pd"), ("test", "pt")):
            pm[split] = dict(f1=[float(f1_score(y[split], Z[(arm, s)][key], labels=ALL, average="macro", zero_division=0)) for s in SEEDS],
                             bal=[float(balanced_accuracy_score(y[split], Z[(arm, s)][key])) for s in SEEDS],
                             inj=[rec(y[split], Z[(arm, s)][key], l2i[INJ]) for s in SEEDS], veh=[rec(y[split], Z[(arm, s)][key], l2i[VEH]) for s in SEEDS],
                             mis=[rec(y[split], Z[(arm, s)][key], l2i[MISS]) for s in SEEDS])
        point[arm] = pm
        ext = [full_metrics(y["test"], Z[(arm, s)]["pt"], labels, Z[(arm, s)]["zt"]) for s in SEEDS]
        ci = macro_f1_ci(y["test"], Z[(arm, 42)]["pt"], K)
        out["arms"][arm] = dict(point=pm, test_extended=dict(acc=[e["acc"] for e in ext], mcc=[e["mcc"] for e in ext], macro_pr_auc=[e["macro_pr_auc"] for e in ext],
                                                              macro_roc_auc=[e["macro_roc_auc"] for e in ext],
                                                              per_class_f1={c: [e["per_class"][c]["f1"] for e in ext] for c in labels}),
                                macro_f1_ci_seed42=ci, best_epoch=[Z[(arm, s)]["st"]["best_epoch"] for s in SEEDS], wall_clock_s=[Z[(arm, s)]["st"]["wall_clock_s"] for s in SEEDS],
                                pred_dev={str(s): Z[(arm, s)]["pd"].tolist() for s in SEEDS}, pred_test={str(s): Z[(arm, s)]["pt"].tolist() for s in SEEDS})
        for split in ("dev", "test"):
            p = pm[split]
            print(f"{arm:5s} {split:4s}: macro-F1 {np.mean(p['f1']):.3f} +- {np.std(p['f1'], ddof=1):.3f} | bal-acc {np.mean(p['bal']):.3f} +- {np.std(p['bal'], ddof=1):.3f} | "
                  f"recall injured {np.mean(p['inj']):.3f} (rows {int((y[split] == l2i[INJ]).sum())}) vehicle {np.mean(p['veh']):.3f} (rows {int((y[split] == l2i[VEH]).sum())}) | missing recall {max(p['mis']):.3f}")
    assert all(max(point[a][s]["mis"]) == 0.0 for a in ARMS for s in ("dev", "test"))
    for split, idx in (("dev", didx), ("test", tidx)):
        yy = y[split]; key = "pd" if split == "dev" else "pt"
        vec = {}
        for arm in ARMS:
            vec[arm] = dict(
                f1={s: bu.macro_f1_vec(yy, Z[(arm, s)][key], K, idx) for s in SEEDS}, bal={s: bu.balanced_acc_vec(yy, Z[(arm, s)][key], K, idx) for s in SEEDS},
                inj={s: bu.recall_vec(yy, Z[(arm, s)][key], K, idx)[0][:, l2i[INJ]] for s in SEEDS}, veh={s: bu.recall_vec(yy, Z[(arm, s)][key], K, idx)[0][:, l2i[VEH]] for s in SEEDS})
        for m, label in (("f1", "macro-F1"), ("bal", "balanced accuracy"), ("inj", "recall, injured or dead"), ("veh", "recall, vehicle damage")):
            pdiff = float(np.mean(point["none"][split][m]) - np.mean(point["inv"][split][m]))
            r = bu.paired_diff_ci(vec["none"][m], vec["inv"][m], point_diff=pdiff)
            r["verdict"] = verdict(r)
            if m == "inj" and r["verdict"] == "not distinguishable":
                r["verdict"] = "not resolvable"      # one rule for every comparison: better/worse where the CI excludes 0; 'not resolvable' only for a CI that includes 0 on the 8/10-row class
            if m in ("inj", "veh"):
                r["n_rows"] = int((yy == l2i[INJ if m == "inj" else VEH]).sum())
            out["pairs"][f"{split}:{m}"] = r
            print(f"{split:4s} none - inv, {label:24s}: {r['diff']:+.4f} [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] {r['verdict']}"
                  f"{'  (selection-eligible)' if split == 'dev' else '  (REPORTING ONLY, test-sampling CI)'}", flush=True)
        if split == "test":
            for arm in ARMS:
                c = bu.mean_ci(vec[arm]["f1"], point=float(np.mean(point[arm]["test"]["f1"])))
                out["arms"][arm]["test_macro_f1_sampling_ci"] = c
                print(f"test {arm:5s} macro-F1 test-sampling 95% CI [{c['ci95_low']:.3f}, {c['ci95_high']:.3f}]")
    write_row("G", "g_fused_L4a-L14_none_vs_inv", 42, out)


if __name__ == "__main__":
    main()
