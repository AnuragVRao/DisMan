"""Fine-tuned fusion addendum (ablation/SELECTION_RULE.md): fine-tuned fusion against the frozen image-only head, Task 2, seeds 42, 43, 44.

Arms: FROZEN = the Day 2 pick (frozen CLIP L/14 image-only head, lr 1e-4, dropout 0.2, d_model 128; results/ablation/H/d4_lr0.0001_do0.2_dm128_seed*.json);
      F1     = L4a-L14 `inv` (pretrained DistilBERT fine-tuned in the v2 head; results/ablation/G/t2_L4a-L14_inv_seed*.json, run earlier and reused);
      F2     = L4a-L14-MiniLM (pretrained all-MiniLM-L6-v2 fine-tuned in the same head; 3 new runs, results/ablation/G/t2_L4a-L14-MiniLM_inv_seed*.json).
F2's checkpoints are re-scored here on dev and test (inference only) and checked against the dev and test macro-F1 the training runs stored.
Rule: mean dev macro-F1 over the seeds; paired dev bootstrap (2000 resamples, seed 20241025, seed-averaged difference) of best minus arm; tied if the CI includes 0; among the tied arms
the simplest (fewest trainable parameters, then the smaller frozen backbone). Then the paired bootstrap of (fine-tuned arm minus FROZEN) for macro-F1, balanced accuracy and the recall of
injured-or-dead and vehicle damage, on DEV (the only version eligible for selection) and on TEST (10,000 resamples; REPORTING ONLY, test-sampling CIs, separate from the seed std).
Labels: better/worse where the CI excludes 0 by more than 0.002, borderline within 0.002, not distinguishable otherwise; injured-or-dead recall 'not resolvable' only if its CI includes 0.
Writes results/ablation/G/g_finetune_compare_seed42.json.
"""
import json

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from transformers import AutoConfig, AutoModel, AutoTokenizer

import bootstrap_utils as bu
from common import ABL_ART, ART, DATA, RES, SPLIT_DIR, cc, write_row
from d2_common import BACKBONE_M
from extra_metrics import macro_f1_ci
from t2_models import FusionModel

SEEDS = (42, 43, 44)
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
INJ, VEH, MISS = "Injured or dead people", "Vehicle damage", "Missing or found people"
ARMS = ("frozen", "f1_distilbert", "f2_minilm")


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
    # ---- predictions of the three arms ----
    P, ST = {a: {} for a in ARMS}, {a: {} for a in ARMS}
    for s in SEEDS:
        r = json.load(open(RES / "H" / f"d4_lr0.0001_do0.2_dm128_seed{s}.json")); P["frozen"][s] = (np.array(r["pred_dev"]), np.array(r["pred_test"])); ST["frozen"][s] = r
    old = json.load(open(RES / "G" / "g_fused_L4a-L14_none_vs_inv_seed42.json"))["arms"]["inv"]
    for s in SEEDS:
        r = json.load(open(RES / "G" / f"t2_L4a-L14_inv_seed{s}.json")); P["f1_distilbert"][s] = (np.array(old["pred_dev"][str(s)]), np.array(old["pred_test"][str(s)])); ST["f1_distilbert"][s] = r
    tok = AutoTokenizer.from_pretrained(MINILM)
    X = {s: torch.tensor(cc.cached(ABL_ART / "clip_l14" / f"clip_humanitarian_{s}.npy", S[s].image_id.tolist(), lambda: (_ for _ in ()).throw(RuntimeError)), dtype=torch.float32) for s in ("dev", "test")}
    T = {s: cc.tokenize(tok, S[s]["text"].tolist(), 128) for s in ("dev", "test")}
    for s in SEEDS:
        st = json.load(open(RES / "G" / f"t2_L4a-L14-MiniLM_inv_seed{s}.json"))
        assert st["flags"]["feat"] == "l14" and st["flags"]["loss"] == "inv" and st["flags"]["text"] == "minilm" and st["flags"]["bert"] == "pretrained"
        enc = AutoModel.from_config(AutoConfig.from_pretrained(MINILM))
        m = FusionModel(K, enc, enc.config.hidden_size, img_dim=768).to(device)
        m.load_state_dict(torch.load(ABL_ART / "ckpt" / f"t2_L4a-L14-MiniLM_inv_seed{s}.pt", map_location=device))
        pd_, pt_ = (logits(m, X[sp], *T[sp], device).argmax(1) for sp in ("dev", "test"))
        d, t = (float(f1_score(y[sp], p, labels=ALL, average="macro", zero_division=0)) for sp, p in (("dev", pd_), ("test", pt_)))
        assert abs(d - st["dev_macro_f1"]) < 1e-12 and abs(t - st["test_macro_f1"]) < 1e-12, f"MiniLM seed {s}: re-scored != stored"
        print(f"F2 MiniLM seed {s}: dev F1 {d:.4f} test F1 {t:.4f} best epoch {st['best_epoch']} (re-scored equals the stored run exactly) | trainable params {sum(p.numel() for p in m.parameters()):,}", flush=True)
        P["f2_minilm"][s] = (pd_, pt_); ST["f2_minilm"][s] = st
    params = {"frozen": ST["frozen"][42]["n_trainable"], "f1_distilbert": ST["f1_distilbert"][42]["n_params"], "f2_minilm": ST["f2_minilm"][42]["n_params"]}
    bb = {"frozen": BACKBONE_M["clip_l14"], "f1_distilbert": BACKBONE_M["clip_l14"], "f2_minilm": BACKBONE_M["clip_l14"]}   # frozen CLIP L/14 image tower (the text encoders are trained, so counted as trainable)
    didx, tidx = bu.resample_idx(len(y["dev"]), 2000), bu.resample_idx(len(y["test"]), 10000)
    rec = lambda yy, p, c: float(recall_score(yy, p, labels=[c], average="macro", zero_division=0))
    pt = {}      # point estimates per arm / split / metric / seed
    for a in ARMS:
        pt[a] = {}
        for sp, k in (("dev", 0), ("test", 1)):
            pt[a][sp] = dict(f1=[float(f1_score(y[sp], P[a][s][k], labels=ALL, average="macro", zero_division=0)) for s in SEEDS], bal=[float(balanced_accuracy_score(y[sp], P[a][s][k])) for s in SEEDS],
                             inj=[rec(y[sp], P[a][s][k], l2i[INJ]) for s in SEEDS], veh=[rec(y[sp], P[a][s][k], l2i[VEH]) for s in SEEDS], mis=[rec(y[sp], P[a][s][k], l2i[MISS]) for s in SEEDS])
            assert max(pt[a][sp]["mis"]) == 0.0
    out = dict(task="T2", arms=list(ARMS), n_params=params, backbone_m=bb, n_boot=dict(dev=2000, test=10000), point=pt, rule={}, pairs={})
    # ---- the pre-registered rule on dev ----
    vecd = {a: {s: bu.macro_f1_vec(y["dev"], P[a][s][0], K, didx) for s in SEEDS} for a in ARMS}
    mean = {a: float(np.mean(pt[a]["dev"]["f1"])) for a in ARMS}; best = max(ARMS, key=lambda a: mean[a])
    diff = {a: bu.paired_diff_ci(vecd[best], vecd[a], point_diff=mean[best] - mean[a]) for a in ARMS if a != best}
    tied = [best] + [a for a in ARMS if a != best and not (diff[a]["ci95_low"] > 0 or diff[a]["ci95_high"] < 0)]
    pick = min(tied, key=lambda a: (params[a], bb[a]))
    out["rule"] = dict(best=best, tied=tied, pick=pick, mean_dev_macro_f1=mean, best_minus_arm=diff)
    print(f"\nRULE (dev only): best {best} ({mean[best]:.4f}); tied: {tied}; PICK: {pick}")
    for a in ARMS:
        print(f"  {a:14s} dev F1 {mean[a]:.4f} ± {np.std(pt[a]['dev']['f1'], ddof=1):.4f} | trainable params {params[a]:,} | " + ("best" if a == best else f"best - arm {diff[a]['diff']:+.4f} [{diff[a]['ci95_low']:+.4f}, {diff[a]['ci95_high']:+.4f}]"))
    # ---- paired bootstraps against the frozen head: dev (selection-eligible) and test (reporting only) ----
    for sp, idx, k in (("dev", didx, 0), ("test", tidx, 1)):
        yy = y[sp]; vec = {}
        for a in ARMS:
            rv = {s: bu.recall_vec(yy, P[a][s][k], K, idx)[0] for s in SEEDS}
            vec[a] = dict(f1={s: bu.macro_f1_vec(yy, P[a][s][k], K, idx) for s in SEEDS}, bal={s: bu.balanced_acc_vec(yy, P[a][s][k], K, idx) for s in SEEDS},
                          inj={s: rv[s][:, l2i[INJ]] for s in SEEDS}, veh={s: rv[s][:, l2i[VEH]] for s in SEEDS})
            if sp == "test":
                out.setdefault("test_ci", {})[a] = bu.mean_ci(vec[a]["f1"], point=float(np.mean(pt[a]["test"]["f1"])))
        for a in ("f1_distilbert", "f2_minilm"):
            for m_, lab in (("f1", "macro-F1"), ("bal", "balanced accuracy"), ("inj", "recall, injured or dead"), ("veh", "recall, vehicle damage")):
                r = bu.paired_diff_ci(vec[a][m_], vec["frozen"][m_], point_diff=float(np.mean(pt[a][sp][m_]) - np.mean(pt["frozen"][sp][m_])))
                r["verdict"] = verdict(r)
                if m_ == "inj" and r["verdict"] == "not distinguishable":
                    r["verdict"] = "not resolvable"
                if m_ in ("inj", "veh"):
                    r["n_rows"] = int((yy == l2i[INJ if m_ == "inj" else VEH]).sum())
                out["pairs"][f"{sp}:{a}:{m_}"] = r
                print(f"{sp:4s} {a:14s} - frozen, {lab:24s}: {r['diff']:+.4f} [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] {r['verdict']}{' (n=%d)' % r['n_rows'] if 'n_rows' in r else ''}"
                      f"{'  (selection-eligible)' if sp == 'dev' else '  (REPORTING ONLY, test-sampling CI)'}")
    for a in ARMS:
        c = out["test_ci"][a]
        print(f"test macro-F1 {a:14s}: {np.mean(pt[a]['test']['f1']):.4f} ± {np.std(pt[a]['test']['f1'], ddof=1):.4f}  test-sampling 95% CI [{c['ci95_low']:.3f}, {c['ci95_high']:.3f}]")
    out["pred"] = {a: {sp: {str(s): P[a][s][k].tolist() for s in SEEDS} for sp, k in (("dev", 0), ("test", 1))} for a in ARMS if a == "f2_minilm"}
    write_row("G", "g_finetune_compare", 42, out)


if __name__ == "__main__":
    main()
