"""Regenerate every number in docs/ablation_findings.md from the result JSONs and check it against the document.

For each table row / prose claim the document is located by a key string, the expected number strings are recomputed from
results/ablation/{A,B,C}/*.json (and the v2 artifact for the embedding cosines), and any expected string missing from that line is
reported as a mismatch (expected vs the numbers actually on the line). Exit code 1 if anything differs.
  python ablation/verify_findings.py [--write-log]     # writes results/ablation/findings_check.txt with every check
"""
import glob
import io
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "ablation"
DOC = (ROOT / "docs" / "ablation_findings.md").read_text(encoding="utf8")
LINES = [l.replace("**", "") for l in DOC.splitlines()]


def rows(cfg, part=None):
    out = {}
    for f in glob.glob(str(RES / (part or "[AB]") / f"{cfg}_seed*.json")):
        d = json.load(open(f))
        out[d["seed"]] = d
    return [out[k] for k in sorted(out)]


def col(cfg, key, part=None):
    return np.array([r[key] for r in rows(cfg, part)], dtype=float)


def ms(x, d=3):
    return f"{x.mean():.{d}f} ± {x.std(ddof=1):.{d}f}" if len(x) > 1 else f"{x.mean():.{d}f}"


def sg(x, d=3):
    return f"{x:+.{d}f}"


checks, bad = [], []


def chk(key, expected, what=""):
    """`key` identifies one line of the document; every string in `expected` must occur on that line."""
    hits = [l for l in LINES if key in l]
    if len(hits) != 1:
        bad.append((key, f"line key matched {len(hits)} lines", ""))
        return
    for e in ([expected] if isinstance(expected, str) else expected):
        ok = e in hits[0]
        checks.append((key, e, ok))
        if not ok:
            bad.append((key, e, hits[0].strip()[:220]))


# ── §2 ladder ─────────────────────────────────────────────────────────────────
for rung, key in [("L0", "| L0 | v1 as it was"), ("L1", "| L1 | + full"), ("L2", "| L2 | + Quick"), ("L3", "| L3 | + epoch"),
                  ("L3b", "| L3b | +"), ("L4a", "| L4a | +"), ("L4", "| L4 | v2")]:
    c = f"t2_{rung}"
    assert len(rows(c)) == 3, c
    chk(key, [ms(col(c, k)) for k in ("dev_macro_f1", "test_acc", "test_macro_f1", "test_weighted_f1")])
STEPS = [("| L0 → L1", "t2_L0", "t2_L1"), ("| L1 → L2", "t2_L1", "t2_L2"), ("| L2 → L3", "t2_L2", "t2_L3"), ("| L3 → L3b", "t2_L3", "t2_L3b"),
         ("| L3b → L4a", "t2_L3b", "t2_L4a"), ("| L4a → L4", "t2_L4a", "t2_L4"), ("| L3 → L4 ", "t2_L3", "t2_L4")]
for key, a, b in STEPS:
    d = col(b, "test_macro_f1") - col(a, "test_macro_f1")
    chk(key, [sg(x) for x in d] + [f"{sg(d.mean())} ± {d.std(ddof=1):.3f}"])
L4a_L4 = col("t2_L4", "test_macro_f1") - col("t2_L4a", "test_macro_f1")
L3b_L4a = col("t2_L4a", "test_macro_f1") - col("t2_L3b", "test_macro_f1")
L3_L4a = col("t2_L4a", "test_macro_f1") - col("t2_L3", "test_macro_f1")
# dev inflation prose
chk("Dev inflation from 30-epoch selection", [f"{col('t2_L1','dev_macro_f1').mean():.2f}", f"{col('t2_L2','dev_macro_f1').mean():.2f}",
                                              f"{col('t2_L1','test_macro_f1').mean():.2f}", f"({col('t2_L3','dev_macro_f1').mean():.3f})",
                                              f"{col('t2_L3','test_macro_f1').mean():.3f}"])
chk("Plan expectation", [f"{col('t2_L3','test_macro_f1').mean() - col('t2_L4','test_macro_f1').mean():.3f}", f"{col('t2_L3','test_macro_f1').std(ddof=1):.3f}"])

# ── §3 T3 ladder ──────────────────────────────────────────────────────────────
T3 = {r: json.load(open(RES / "A" / f"t3_{r}_seed42.json")) for r in ("L0", "L1", "L2", "L3", "L4")}
for r, key in [("L0", "| L0 v1 index"), ("L1", "| L1 test rows removed"), ("L2", "| L2 \"Don't know\" rows removed"),
               ("L3", "| L3 dev removed"), ("L4", "| L4 tweet siblings removed")]:
    d = T3[r]
    chk(key, [f"{d['index_rows']:,}", f"{d['test_acc']:.4f}", f"{d['test_macro_f1']:.4f}", f"{d['test_weighted_f1']:.4f}",
              f"{d['test_queries_with_self_in_top5']}/529" if r == "L0" else f"| {d['test_queries_with_self_in_top5']} |"])
chk("| L0 v1 index", f"top-1: {T3['L0']['test_queries_with_self_top1']}")
chk("L0 → L1 (-0.31 macro-F1)", f"{T3['L1']['test_macro_f1'] - T3['L0']['test_macro_f1']:.2f}")
chk("L1 → L2 (+0.126)", f"+{T3['L2']['test_macro_f1'] - T3['L1']['test_macro_f1']:.3f}")
chk("786 of 2,645", [f"{T3['L1']['neighbour_votes_for_dont_know']} of {T3['L1']['neighbour_votes_total']:,}",
                     f"{round(100 * T3['L1']['neighbour_votes_for_dont_know'] / T3['L1']['neighbour_votes_total'])}%", f"{T3['L1']['predicted_dont_know']} test queries"])
chk("L2 → L3 (dev rows removed)", [f"{T3['L2']['test_macro_f1']:.4f} → {T3['L3']['test_macro_f1']:.4f}", f"{T3['L2']['test_acc']:.4f} → {T3['L3']['test_acc']:.4f}"])
chk("L3 → L4 (tweet siblings removed)", [f"{T3['L3']['test_macro_f1']:.4f} → {T3['L4']['test_macro_f1']:.4f}",
                                         f"{T3['L3']['test_queries_with_same_tweet_sibling_in_top5']} test queries", f"{T3['L4']['test_queries_with_same_tweet_sibling_in_top5']}"])
v2m = json.load(open(ROOT / "results" / "metrics.json"))["T3_test"]
chk("reproduces v2's headline exactly", [f"{v2m['acc']:.4f}", f"{v2m['macro_f1']:.4f}", f"{v2m['weighted_f1']:.4f}"])

# ── §4 Part B T2 ──────────────────────────────────────────────────────────────
PB = [("| full (L4a)", "t2_L4a", "A"), ("| image only (CLIP", "t2_image_only", "B"), ("| concat, no attention", "t2_concat", "B"),
      ("| no text→image attention", "t2_no_t2i", "B"), ("| text only | 66,628,104", "t2_text_only", "B")]
for key, c, part in PB:
    chk(key, [f"{int(rows(c, part)[0]['n_params']):,}"] + [ms(col(c, k, part)) for k in ("dev_macro_f1", "test_acc", "test_macro_f1", "test_weighted_f1")])
full = col("t2_L4a", "test_macro_f1", "A")
pair = lambda c: col(c, "test_macro_f1", "B") - full
chk("Paired differences from the full model",
    [f"image_only {sg(pair('t2_image_only').mean())} ± {pair('t2_image_only').std(ddof=1):.3f}",
     f"concat {sg(pair('t2_concat').mean())} ± {pair('t2_concat').std(ddof=1):.3f}",
     f"no_t2i {sg(pair('t2_no_t2i').mean())} ± {pair('t2_no_t2i').std(ddof=1):.3f}"] + [sg(x) for x in pair("t2_no_t2i")] +
    [f"text_only {sg(pair('t2_text_only').mean())} ± {pair('t2_text_only').std(ddof=1):.3f}"])
maj = rows("t2_L4")[0]["majority_baseline_macro_f1"]
chk("No variant fell below", f"{maj:.4f}")
n_runs = len(glob.glob(str(RES / "A" / "t2_*_seed*.json"))) + len(glob.glob(str(RES / "B" / "t2_*_seed*.json")))
checks.append(("T2 run count", f"{n_runs} == 33", n_runs == 33))
if n_runs != 33:
    bad.append(("T2 run count", "33", str(n_runs)))
chk("33 T2 runs", f"{maj:.4f}")

# ── §5 Part B T3 ──────────────────────────────────────────────────────────────
def b3(emb, k, vote, key):
    return json.load(open(RES / "B" / f"t3_{emb}_k{k}_{vote}_seed42.json"))[key]
KS = (1, 3, 5, 10, 20)
chk("| Dev macro-F1 | 0.4627", [f"{b3('fused', k, 'majority', 'dev_macro_f1'):.4f}" for k in KS])
chk("| Test macro-F1 | 0.4538", [f"{b3('fused', k, 'majority', 'test_macro_f1'):.4f}" for k in KS])
chk("| Test acc. | 0.6011", [f"{b3('fused', k, 'majority', 'test_acc'):.4f}" for k in KS])
for key, v in [("| majority (v2) |", "majority"), ("| similarity-weighted |", "sim_weighted"), ("| class-balanced |", "class_balanced")]:
    chk(key, [f"{b3('fused', 5, v, 'dev_macro_f1'):.4f}", f"{b3('fused', 5, v, 'test_acc'):.4f}", f"{b3('fused', 5, v, 'test_macro_f1'):.4f}"])
lr = lambda e, k: json.load(open(RES / "B" / f"t3_lr_{e}_seed42.json"))[k]
for key, e in [("| image only | 0.", "image"), ("| text only | 0.", "text"), ("| fused | 0.", "fused")]:
    chk(key, [f"{b3(e,5,'majority','dev_macro_f1'):.4f} / {b3(e,5,'majority','test_macro_f1'):.4f} ({b3(e,5,'majority','test_acc'):.4f})",
              f"C = {lr(e,'C'):g}: {lr(e,'dev_macro_f1'):.4f} / {lr(e,'test_macro_f1'):.4f} ({lr(e,'test_acc'):.4f})"])
same = all(abs(b3(e, k, "sim_weighted", m) - b3(e, k, "majority", m)) < 1e-12 for e in ("image", "text", "fused") for k in KS
           for m in ("dev_macro_f1", "test_macro_f1", "test_acc"))
checks.append(("sim_weighted == majority in 15/15 cells", str(same), same))
if not same:
    bad.append(("sim_weighted == majority", "True", "False"))
chk("Similarity-weighted voting gives identical", "15 of 15")
chk("class-balanced voting raises", [f"+{b3('fused',5,'class_balanced','dev_macro_f1') - b3('fused',5,'majority','dev_macro_f1'):.3f}",
                                     f"+{b3('fused',5,'class_balanced','test_macro_f1') - b3('fused',5,'majority','test_macro_f1'):.3f}",
                                     f"{b3('fused',5,'majority','test_acc'):.3f} → {b3('fused',5,'class_balanced','test_acc'):.3f}"])
chk("Best dev-selected k-NN", [f"dev {b3('image',5,'class_balanced','dev_macro_f1'):.4f} → test {b3('image',5,'class_balanced','test_macro_f1'):.4f}",
                               f"dev {lr('image','dev_macro_f1'):.4f} → test {lr('image','test_macro_f1'):.4f}"])
best_knn = max(((b3(e, k, v, "dev_macro_f1"), e, k, v) for e in ("image", "text", "fused") for k in KS for v in ("majority", "sim_weighted", "class_balanced")))
checks.append(("best dev kNN cell is image/k5/class_balanced", str(best_knn[1:]), best_knn[1:] == ("image", 5, "class_balanced")))
if best_knn[1:] != ("image", 5, "class_balanced"):
    bad.append(("best dev kNN cell", "image k5 class_balanced", str(best_knn)))

# ── §6 Part C ─────────────────────────────────────────────────────────────────
def c1(cfg, cond, key="macro_f1"):
    return np.array([r[f"{cond}_{key}"] for r in rows(cfg, "C")])
for key, cfg in [("| T2 L4a (n = 3)", "c1_t2_L4a"), ("| T2 L4, v2 (n = 3)", "c1_t2_L4")]:
    full_ = c1(cfg, "full")
    chk(key, [ms(full_), f"{ms(c1(cfg,'image_grey'))} (Δ {sg((c1(cfg,'image_grey') - full_).mean())})",
              f"{ms(c1(cfg,'text_empty'))} (Δ {sg((c1(cfg,'text_empty') - full_).mean())})"])
t3c = json.load(open(RES / "C" / "c1_t3_seed42.json"))
chk("| T3 5-NN fused (n = 1)", [f"{t3c['full_macro_f1']:.4f}", f"{t3c['image_grey_macro_f1']:.4f} (Δ {sg(t3c['image_grey_macro_f1'] - t3c['full_macro_f1'], 3)})",
                                f"{t3c['text_empty_macro_f1']:.4f} (Δ {sg(t3c['text_empty_macro_f1'] - t3c['full_macro_f1'], 3)})"])
chk("A grey image collapses both", f"{maj:.4f}")
chk("v2's text stream is ignored", [f"{sg((c1('c1_t2_L4','text_empty') - c1('c1_t2_L4','full')).mean())}", f"{-(c1('c1_t2_L4a','text_empty') - c1('c1_t2_L4a','full')).mean():.3f}"])
c2 = lambda t: json.load(open(RES / "C" / f"c2_{t}_macro_avg_seed42.json"))
a1, a3 = c2("t1"), c2("t3")
chk("| Macro average over 7 events", [f"{a1['test_acc']:.3f} / {a1['test_macro_f1']:.3f}", f"{a1['in_dist_macro_f1']:.3f}",
                                      f"{a3['test_acc']:.3f} / {a3['test_macro_f1']:.3f}", f"{a3['in_dist_macro_f1']:.3f}"])
ev = lambda t, e: json.load(open(RES / "C" / f"c2_{t}_{e}_seed42.json"))
chk("T1 drops", [f"{a1['in_dist_macro_f1'] - a1['test_macro_f1']:.3f}", f"srilanka_floods ({ev('t1','srilanka_floods')['test_macro_f1']:.3f} vs {ev('t1','srilanka_floods')['in_dist_macro_f1']:.3f})",
                 f"hurricane_maria ({ev('t1','hurricane_maria')['test_macro_f1']:.3f} vs {ev('t1','hurricane_maria')['in_dist_macro_f1']:.3f})"])
chk("T3 drops", [f"{a3['in_dist_macro_f1'] - a3['test_macro_f1']:.3f}", f"srilanka_floods {ev('t3','srilanka_floods')['n']}", f"mexico {ev('t3','mexico_earthquake')['n']}", f"iraq_iran {ev('t3','iraq_iran_earthquake')['n']}"])
folds = len(glob.glob(str(RES / "C" / "c2_t[13]_*_seed42.json"))) - 2
chk("asserted by event, tweet and image id for all 14 folds", "14") if folds == 14 else bad.append(("C2 fold count", "14", str(folds)))

# ── §1 facts ──────────────────────────────────────────────────────────────────
reg = json.load(open(RES / "A" / "t1_regression_seed42.json"))
chk("T1 regression check passes", [f"{reg['test_acc']:.6f}", f"{reg['test_weighted_f1']:.6f}"])
v2r = json.load(open(RES / "v2_rerun" / "v2_rerun_metrics.json"))["T2_test"]
chk("v2 reproduces itself exactly", [f"{v2r['acc']:.4f}", f"{v2r['macro_f1']:.4f}", f"{v2r['weighted_f1']:.4f}"])
sup = json.load(open(RES / "A" / "_superseded" / "t2_L4_rerun_seed42.json"))
chk("gave test macro-F1 0.6022", f"{sup['test_macro_f1']:.4f}")
try:  # embedding statistics from the v2 artifact (float64 to avoid cosine rounding above 1)
    import torch
    from transformers import DistilBertConfig, DistilBertModel
    sys.path.insert(0, str(ROOT))
    import crisis_common as cc
    sd = torch.load(ROOT / "artifacts" / "task2_model.pt", map_location="cpu")
    w = sd["text_encoder.embeddings.word_embeddings.weight"].double()
    pre = DistilBertModel.from_pretrained("distilbert-base-uncased").embeddings.word_embeddings.weight.detach().double()
    cc.seed_all(42)
    init = cc.DualStreamClassifier(8, bert_config=DistilBertConfig.from_pretrained(ROOT / "artifacts" / "distilbert_config")).text_encoder.embeddings.word_embeddings.weight.detach().double()
    cos = lambda a, b: float((a.flatten() @ b.flatten()) / (a.norm() * b.norm()))
    chk("Evidence: saved word-embedding std", [f"{w.std():.4f}", f"{pre.std():.4f}", f"{cos(w, pre):.4f}"])
    chk("cosine with the model's own seed-42 random init", [f"{cos(w, init):.3f}", f"{float((w - init).norm() / init.norm()):.3f}"])
except Exception as e:  # keep the rest of the check usable without the artifact
    bad.append(("embedding statistics", "computed", f"skipped: {e}"))


# ── §4b / §6 additions: parameter gap, L4a empty-text values, attention probes ─────────────────────────────────────────
chk("txt_to_img` block, 263,680 parameters", f"{int(rows('t2_L4a', 'A')[0]['n_params'] - rows('t2_no_t2i', 'B')[0]['n_params']):,}")
chk("L4a with empty text:", [f"{x:.3f}" for x in c1("c1_t2_L4a", "text_empty")])
try:
    import torch
    sys.path.insert(0, str(ROOT))
    import crisis_common as cc2
    torch.manual_seed(0)
    blk = cc2.CrossModalAttention(256, 4, 0.2).eval()
    im = torch.randn(5, 1, 256); qa, qb = torch.randn(5, 1, 256), torch.randn(5, 1, 256)
    oa, wa = blk.attn(qa, im, im); ob, _ = blk.attn(qb, im, im)
    Wq = blk.attn.in_proj_weight; bq = blk.attn.in_proj_bias
    ref = (im @ Wq[512:].T + bq[512:]) @ blk.attn.out_proj.weight.T + blk.attn.out_proj.bias
    assert float(wa.min()) == 1.0 and float((oa - ob).abs().max()) == 0.0 and torch.allclose(oa, ref, atol=1e-5)
    torch.manual_seed(0)
    blk = cc2.CrossModalAttention(256, 4, 0.2); blk.train()
    img2 = torch.randn(2000, 1, 256); q2 = torch.randn(2000, 1, 256)
    _, w = blk.attn(q2, img2, img2, need_weights=True, average_attn_weights=False)
    chk("zeroed with probability", f"{float((w.flatten() == 0).float().mean()):.3f}")
except Exception as e:
    bad.append(("attention probes", "computed", f"skipped: {e}"))

# ── report ────────────────────────────────────────────────────────────────────
print(f"{len(checks)} numeric checks against docs/ablation_findings.md; {len(bad)} mismatches")
for k, e, found in bad:
    print(f"  MISMATCH [{k}] expected {e!r} | line: {found}")
if "--write-log" in sys.argv:
    (RES / "findings_check.txt").write_text("\n".join(f"{'OK ' if ok else 'BAD'} [{k}] {e}" for k, e, ok in checks) + "\n", encoding="utf8")
sys.exit(1 if bad else 0)
