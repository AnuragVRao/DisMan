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
    COS_PRE = cos(w, pre)
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


# ── §9.1 A1 (agreed vs disagreed), recomputed from results/ablation/X ─────────────────────────────────────────────────
X1 = {m: [json.load(open(f)) for f in sorted(glob.glob(str(RES / "X" / f"a1_{m}_seed*.json")))] for m in ("L4a", "image_only", "text_only")}
assert all(len(v) == 3 for v in X1.values())
x0 = X1["L4a"][0]
xg = lambda m, s, k: np.array([r["subsets"][s][k] for r in X1[m]])
xd = lambda a, b, s, k: xg(a, s, k) - xg(b, s, k)
xpm = lambda x: f"{sg(x.mean())} ± {x.std(ddof=1):.3f}"
mbx = x0["majority_baseline"]
chk("The test set splits into", [f"{x0['n_agreed']} agreed rows", f"{x0['n_disagreed']:,} disagreed rows"])
chk("Majority-class reference", [f"agreed acc {mbx['agreed']['acc']:.3f} / macro-F1 {mbx['agreed']['macro_f1']:.3f}",
                                 f"disagreed {mbx['disagreed']['acc']:.3f} / {mbx['disagreed']['macro_f1']:.3f}"])
for m, nm in (("L4a", "L4a (full)"), ("image_only", "image only"), ("text_only", "text only")):
    for s in ("agreed", "disagreed", "all"):
        chk(f"| {nm} | {s} |", [f"| {X1[m][0]['subsets'][s]['n']} | {len(X1[m])} |", ms(xg(m, s, "acc")), ms(xg(m, s, "macro_f1")), ms(xg(m, s, "weighted_f1"))])
chk("On agreed rows the full model beats image-only", [xpm(xd("L4a", "image_only", "agreed", "macro_f1")), xpm(xd("L4a", "image_only", "agreed", "acc")),
                                                       ", ".join(sg(x) for x in xd("L4a", "image_only", "agreed", "acc"))])
assert (xd("L4a", "image_only", "agreed", "acc") > 0).all()
chk("On disagreed rows it is", [xpm(xd("L4a", "image_only", "disagreed", "macro_f1")), xpm(xd("L4a", "image_only", "disagreed", "acc"))])
chk("Text-only on disagreed rows", [ms(xg("text_only", "disagreed", "acc")), f"{mbx['disagreed']['acc']:.3f}", ms(xg("text_only", "disagreed", "macro_f1")), f"{mbx['disagreed']['macro_f1']:.3f}"])
assert xg("text_only", "disagreed", "acc").mean() < mbx["disagreed"]["acc"]
gapx = lambda m, k: xg(m, "agreed", k) - xg(m, "disagreed", k)
chk("Agreed minus disagreed gap", [f"L4a {xpm(gapx('L4a', 'macro_f1'))} / {xpm(gapx('L4a', 'acc'))}", f"image only {xpm(gapx('image_only', 'macro_f1'))} / {xpm(gapx('image_only', 'acc'))}"])
chk("Caveats: the two subsets differ in class mix", [f"{mbx['agreed']['acc'] * 100:.1f}%", f"{mbx['disagreed']['acc'] * 100:.1f}%", f"{max(xg('text_only', s, 'macro_f1').std(ddof=1) for s in ('agreed', 'disagreed', 'all')):.2f} for text-only"])

# ── §9.2 E1 bootstrap intervals, recomputed from results/ablation/X/e1_* ──────────────────────────────────────────────
_E1N = ["t3_probe_vs_knn_fused", "t3_probe_vs_knn_image", "t3_L4_vs_L1", "t2_imageonly_vs_L4a", "t2_agreed_acc_L4a_vs_imageonly"]
_E1 = {n: json.load(open(RES / "X" / f"e1_{n}_seed42.json")) for n in _E1N}
for _n, _r in _E1.items():
    chk(f"| `{_n}` |", [f"{_r['n_rows']:,}", f"{_r['score_a']:.4f}", f"{_r['score_b']:.4f}", f"{_r['diff']:+.4f}",
                       f"[{_r['ci95_low']:+.4f}, {_r['ci95_high']:+.4f}]", "yes" if _r["ci_excludes_zero"] else "no"])
chk("Paired percentile bootstrap:", [f"{_E1['t3_L4_vs_L1']['n_boot']:,} resamples", f"seed {_E1['t3_L4_vs_L1']['rng_seed']}"])
chk("The probe beats 5-NN on both embeddings", [f"fused {_E1['t3_probe_vs_knn_fused']['diff']:+.3f}", f"image {_E1['t3_probe_vs_knn_image']['diff']:+.3f}"])
chk("The gap between v2's index (L4)", [f"{_E1['t3_L4_vs_L1']['diff']:+.3f}"])
chk("Image-only against L4a at seed 42", [f"{_E1['t2_imageonly_vs_L4a']['diff']:+.3f}", f"{_E1['t2_imageonly_vs_L4a']['ci95_low']:+.3f} to {_E1['t2_imageonly_vs_L4a']['ci95_high']:+.3f}"])
chk("The accuracy gain of L4a over image-only on the agreed rows", [f"{_E1['t2_agreed_acc_L4a_vs_imageonly']['diff']:+.3f}"])
assert not _E1["t2_imageonly_vs_L4a"]["ci_excludes_zero"] and all(_E1[n]["ci_excludes_zero"] for n in _E1N if n != "t2_imageonly_vs_L4a")

# ── §9.2 L1 - L0 row and §9.3 E2, recomputed from results/ablation/X ──────────────────────────────────────────────────
_e1b = json.load(open(RES / "X" / "e1_t2_L1_vs_L0_seed42.json"))
assert _e1b["ci_excludes_zero"] and _e1b["n_boot"] == 10000
chk("| `t2_L1_vs_L0` |", [f"{_e1b['n_rows']:,}", f"{_e1b['score_a']:.4f}", f"{_e1b['score_b']:.4f}", f"{_e1b['diff']:+.4f}",
                          f"[{_e1b['ci95_low']:+.4f}, {_e1b['ci95_high']:+.4f}]", "yes"])
chk("T2 L1 − L0 (the path fix)",
    [f"{_e1b['diff']:+.3f}", f"{_e1b['ci95_low']:+.3f} to {_e1b['ci95_high']:+.3f}",
     ", ".join(f"{v:.4f}" for v in _e1b["regen_macro_f1_L0_crop_seeds"]), f"stored {_e1b['stored_macro_f1_L0']:.4f}",
     ", ".join(f"{v:.4f}" for v in _e1b["regen_macro_f1_L1_crop_seeds"]), f"stored {_e1b['stored_macro_f1_L1']:.4f}",
     f"crop seed {_e1b['bootstrapped_crop_seed']}"])
_io = json.load(open(RES / "X" / "e1_t2_imageonly_vs_L4a_seed42.json"))
_sd = [col(c, "test_macro_f1").std(ddof=1) for c in ("t2_L0", "t2_L1", "t2_L2", "t2_L3", "t2_L3b", "t2_L4a", "t2_L4")]
chk("Test-sample uncertainty is wider than the seed noise floor",
    [f"half-width of {(_io['ci95_high'] - _io['ci95_low']) / 2:.3f}", f"{min(_sd):.3f} to {max(_sd):.3f}"])

_e2 = json.load(open(RES / "X" / "e2_hybrid_t3_seed42.json"))
_pr, _kn = json.load(open(RES / "B" / "t3_lr_fused_seed42.json")), json.load(open(RES / "B" / "t3_fused_k5_majority_seed42.json"))
assert _e2["equals_probe_row"] and abs(_e2["test_macro_f1"] - _pr["test_macro_f1"]) < 1e-12 and abs(_e2["knn_test_macro_f1"] - _kn["test_macro_f1"]) < 1e-12
_bb = _e2["bootstrap_hybrid_minus_knn"]
assert abs(_bb["diff"] - _E1["t3_probe_vs_knn_fused"]["diff"]) < 1e-12 and abs(_bb["ci95_low"] - _E1["t3_probe_vs_knn_fused"]["ci95_low"]) < 1e-12
chk("E2 was run because", [f"C = {_e2['probe_C']:g}", f"{_e2['index_rows']:,} vectors", f"{_e2['k']}-NN"])
chk("Because the hybrid's label is the probe's label", [f"{_e2['n_test']} queries"])
chk("| Hybrid (probe label) |", [f"{_e2['test_acc']:.4f}", f"{_e2['test_macro_f1']:.4f}", f"{_e2['test_weighted_f1']:.4f}"])
chk("| v2 5-NN (fused, k = 5) |", [f"{_e2['knn_test_acc']:.4f}", f"{_e2['knn_test_macro_f1']:.4f}", f"{_e2['knn_test_weighted_f1']:.4f}"])
chk("Bootstrap, hybrid minus 5-NN macro-F1", [f"{_bb['diff']:+.4f}", f"{_bb['ci95_low']:+.4f} to {_bb['ci95_high']:+.4f}", f"{_bb['n_boot']:,} resamples"])
chk("The neighbour majority disagrees with the probe on",
    [f"{_e2['n_disagree']} of {_e2['n_test']} queries ({100 * _e2['disagreement_rate']:.1f}%)", f"({_e2['n_test'] - _e2['n_disagree']} queries) accuracy is {_e2['probe_acc_where_agree']:.3f}",
     f"the probe is right on {_e2['probe_acc_where_disagree']:.3f}", f"majority on {_e2['knn_acc_where_disagree']:.3f}"])

# ── §9.4 C: calibration, recomputed from results/ablation/X/c_calibration_* ───────────────────────────────────────────
_c3 = json.load(open(RES / "X" / "c_calibration_t3_probe_seed42.json"))
_c2 = [json.load(open(RES / "X" / f"c_calibration_t2_L4a_seed{s_}.json")) for s_ in (42, 43, 44)]
assert "exceeds_threshold" not in _c3 and all("exceeds_threshold" not in r and "temperature" in r for r in _c2) and "temperature" in _c3   # no ECE threshold: every model is scaled
_demo_cal = json.load(open(ROOT / "demo" / "calibration.json"))
assert _demo_cal["t2"]["temperature"] == _c2[0]["temperature"] and _demo_cal["t3_probe"]["temperature"] == _c3["temperature"], "demo/calibration.json != results"
assert _demo_cal["t2"]["checkpoint"] == "t2_L4a_seed42.pt"
chk("Task 3, logistic probe (fused, C = 1;", f"{_c3['n']} test queries")
chk(f"| {_c3['accuracy']:.4f} | {_c3['mean_confidence']:.4f} |",
    [f"| {_c3['ece_before']:.4f} | {_c3['temperature']:.3f} | {_c3['ece_after']:.4f} | {_c3['nll_before']:.4f} | {_c3['nll_after']:.4f} | {_c3['brier_before']:.4f} | {_c3['brier_after']:.4f} |"])
for _sd, _r in zip((42, 43, 44), _c2):
    chk(f"| {_sd} | {_r['accuracy']:.4f} | {_r['mean_confidence']:.4f} |",
        [f"| {_r['ece_before']:.4f} | {_r['temperature']:.3f} | {_r['ece_after']:.4f} | {_r['nll_before']:.4f} | {_r['nll_after']:.4f} |"])
_m = lambda k, d: f"{np.mean([r[k] for r in _c2]):.{d}f} ± {np.std([r[k] for r in _c2], ddof=1):.{d}f}"
chk("| mean ± std (3 seeds) |", [_m("accuracy", 4), _m("mean_confidence", 4), _m("ece_before", 4), _m("temperature", 3), _m("ece_after", 4), _m("nll_before", 4), _m("nll_after", 4)])
assert _c3["mean_confidence"] < _c3["accuracy"] and _c3["temperature"] < 1                  # "under-confident ... T < 1"
assert all(r["mean_confidence"] > r["accuracy"] and r["temperature"] > 1 for r in _c2)      # "over-confident ... T > 1"
assert _c3["ece_after"] < 0.05 and all(r["ece_after"] < 0.05 for r in _c2)                  # "brings ECE below 0.05"
chk("The T3 probe is under-confident", [f"mean confidence {_c3['mean_confidence']:.3f} against accuracy {_c3['accuracy']:.3f}", f"T = {_c3['temperature']:.3f}",
                                         f"mean confidence {np.mean([r['mean_confidence'] for r in _c2]):.3f} against accuracy {np.mean([r['accuracy'] for r in _c2]):.3f}",
                                         f"T3 {_c3['ece_before']:.3f} to {_c3['ece_after']:.3f}",
                                         f"T2 {np.mean([r['ece_before'] for r in _c2]):.3f} to {np.mean([r['ece_after'] for r in _c2]):.3f}"])
_ag, _q, _x2 = _c3["by_agreement"], _c3["by_confidence_quintile"], _c3["agreement_x_confidence"]
chk("| agree |", [f"{_ag['agree']['n']}", f"{_ag['agree']['accuracy']:.3f}"])
chk("| disagree |", [f"{_ag['disagree']['n']}", f"{_ag['disagree']['accuracy']:.3f}"])
for _x in _q:
    chk(f"| Q{_x['quintile']} |", [f"{_x['n']}", f"{_x['conf_min']:.3f} to {_x['conf_max']:.3f}", f"{_x['mean_confidence']:.3f}", f"{_x['accuracy']:.3f}"])
assert all(_q[i]["accuracy"] < _q[i + 1]["accuracy"] for i in range(4))                     # "rises monotonically"
chk("| Agreement × confidence", [f"median {_c3['median_confidence']:.3f}"])
for _k, _lab in (("agree_high_conf", "agree, high confidence"), ("agree_low_conf", "agree, low confidence"),
                 ("disagree_high_conf", "disagree, high confidence"), ("disagree_low_conf", "disagree, low confidence")):
    chk(f"| {_lab} |", [f"{_x2[_k]['n']}", f"{_x2[_k]['accuracy']:.3f}"])
assert _c3["auroc_confidence_ci95"][0] > 0.5 and _c3["auroc_agreement_ci95"][0] > 0.5
chk("The demo shows temperature-scaled confidences", [f"T = {_c2[0]['temperature']:.3f} for the L4a seed-42", f"T = {_c3['temperature']:.3f} for the Task 3 probe"])
chk("Both signals track correctness", [f"{_q[0]['accuracy']:.3f} (Q1)", f"{_q[-1]['accuracy']:.3f} (Q5)", f"is {_ag['agree']['accuracy']:.3f}", f"against {_ag['disagree']['accuracy']:.3f}"])
chk("AUROC for predicting a correct answer", [f"probe confidence {_c3['auroc_confidence']:.3f}", f"{_c3['auroc_confidence_ci95'][0]:.3f} to {_c3['auroc_confidence_ci95'][1]:.3f}",
                                              f"agreement {_c3['auroc_agreement']:.3f}", f"{_c3['auroc_agreement_ci95'][0]:.3f} to {_c3['auroc_agreement_ci95'][1]:.3f}", f"{_c3['n_boot']:,} resamples"])
chk("The signals are partly independent", [f"{_x2['agree_low_conf']['accuracy']:.3f} (low confidence) to {_x2['agree_high_conf']['accuracy']:.3f} (high)",
                                           f"from {_x2['disagree_low_conf']['accuracy']:.3f} to {_x2['disagree_high_conf']['accuracy']:.3f}"])
# cross-check: the agreement split here equals E2's
_e2b = json.load(open(RES / "X" / "e2_hybrid_t3_seed42.json"))
assert _ag["disagree"]["n"] == _e2b["n_disagree"] and abs(_ag["agree"]["accuracy"] - _e2b["probe_acc_where_agree"]) < 1e-12

# ── neighbour-similarity spread (CPU faiss, v2 index): used by the findings doc and the .tex ──────────────────────────
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import faiss
sys.path.insert(0, str(ROOT / "ablation"))
from common import ART as _ART, DATA as _DATA, SPLIT_DIR as _SPLIT, cc as _cc


def _nm():
    raise RuntimeError("cache miss")


_S = {x: _cc.load_split("damage", x, _DATA, _SPLIT, verbose=False)[0] for x in ("train", "test")}
_V = {x: _cc.t3_vectors(_cc.cached(_ART / "emb" / f"clip_damage_{x}.npy", _S[x].image_id.tolist(), _nm),
                        _cc.cached(_ART / "emb" / f"sbert_damage_{x}.npy", _S[x].image_id.tolist(), _nm)) for x in _S}
_keep = ~_S["train"].tweet_id.isin(set(_S["test"].tweet_id)).to_numpy()
_ix = faiss.IndexFlatIP(896); _ix.add(_V["train"][_keep]); _sims, _ = _ix.search(_V["test"], 5)
SPREAD, MEANSIM = float((_sims[:, 0] - _sims[:, 4]).mean()), float(_sims.mean())
chk("Similarity-weighted voting gives identical", [f"{MEANSIM:.3f}", f"{SPREAD:.3f}"])

# ══ report/ablation_section.tex ═══════════════════════════════════════════════════════════════════════════════════════
TEX_PATH = ROOT / "report" / "ablation_section.tex"
TEX_RAW = TEX_PATH.read_text(encoding="utf8")


def norm(l):
    l = l.replace("\\,", ",").replace("\\pm", "±").replace("\\to", "→").replace("\\%", "%").replace("\\_", "_")
    l = l.replace("$", "").replace("~", " ")
    return re.sub(r"\s+", " ", l).strip()


TEX = [(i, norm(l)) for i, l in enumerate(TEX_RAW.splitlines()) if l.strip() and not l.lstrip().startswith("%")]
tex_spans = {i: [] for i, _ in TEX}          # character spans per line that a check has verified
tex_bad = []
tex_log = []
tex_nchecks = 0


def tchk(anchor, expected, cover_anchor=False):
    """The unique normalised .tex line containing `anchor` must contain every expected string; matched spans count as covered.
    cover_anchor=True also marks the anchor itself as verified (only for row anchors built from JSON-derived numbers)."""
    global tex_nchecks
    hits = [(i, l) for i, l in TEX if anchor in l]
    if len(hits) != 1:
        tex_bad.append((anchor, f"anchor matched {len(hits)} lines", ""))
        return
    i, l = hits[0]
    if cover_anchor:
        q = l.find(anchor)
        tex_spans[i].append((q, q + len(anchor)))
    for e in ([expected] if isinstance(expected, str) else expected):
        tex_nchecks += 1
        pos = [m.start() for m in re.finditer(re.escape(e), l)]
        tex_log.append((anchor, e, bool(pos)))
        if not pos:
            tex_bad.append((anchor, e, l[:230]))
        for q in pos:
            tex_spans[i].append((q, q + len(e)))


cc_T3_LABELS = LABELS_T3 = ["Severe damage", "Mild damage", "Little or no damage"]   # the 3 real classes (v2)
# 1. every embedded table must be identical to the generated tables.tex block; its numbers are re-derived from the JSONs below
GEN = (RES / "tables.tex").read_text(encoding="utf8")
for blk in re.findall(r"\\begin\{table\}.*?\\end\{table\}", TEX_RAW, re.S):
    tex_nchecks += 1
    if blk.strip() not in GEN:
        tex_bad.append(("embedded table", "identical to results/ablation/tables.tex", blk.splitlines()[3][:100]))

# 2. table rows, recomputed from the JSONs
nseed = lambda c, part=None: len(rows(c, part))
for rung, key, extra in [("L0", "L0 & v1 buggy", ["30 ep"]), ("L1", "L1 & + full", []), ("L2", "L2 & + QuickGELU", []),
                         ("L3", "L3 & + epoch", ["30 → 10"]), ("L3b", "L3b & + determ", []), ("L4a", "L4a & + AMP", []), ("L4", "L4 & v2:", [])]:
    c = f"t2_{rung}"
    tchk(key, [f"& {nseed(c)} &"] + [ms(col(c, k)) for k in ("dev_macro_f1", "test_acc", "test_macro_f1", "test_weighted_f1")] + extra)
for rung, key in [("L0", "L0 & v1 index"), ("L1", "L1 & test rows"), ("L2", "L2 & pollution"), ("L3", "L3 & dev removed"), ("L4", "L4 & tweet siblings")]:
    d = T3[rung]
    tchk(key, [f"& {d['index_rows']} &", f"{d['test_acc']:.4f}", f"{d['test_macro_f1']:.4f}", f"{d['test_weighted_f1']:.4f}"])
for label, c, part in [("full model (L4a)", "t2_L4a", "A"), ("image only", "t2_image_only", "B"), ("concat, no attention", "t2_concat", "B"),
                       ("no text→image attention", "t2_no_t2i", "B"), ("text only", "t2_text_only", "B")]:
    # anchored with the (JSON-derived) parameter count so the A1 table's rows with the same model names cannot match
    tchk(f"{label} & {nseed(c, part)} & {int(rows(c, part)[0]['n_params']):,}",
         [ms(col(c, k, part)) for k in ("dev_macro_f1", "test_acc", "test_macro_f1", "test_weighted_f1")], cover_anchor=True)
for k in KS:
    tchk(f"{k} & {b3('fused', k, 'majority', 'dev_macro_f1'):.4f}", [f"{b3('fused', k, 'majority', 'test_acc'):.4f}", f"{b3('fused', k, 'majority', 'test_macro_f1'):.4f}"], cover_anchor=True)
for key, v in [("majority (v2) &", "majority"), ("similarity-weighted &", "sim_weighted"), ("class-balanced &", "class_balanced")]:
    tchk(key, [f"{b3('fused', 5, v, 'dev_macro_f1'):.4f}", f"{b3('fused', 5, v, 'test_acc'):.4f}", f"{b3('fused', 5, v, 'test_macro_f1'):.4f}"])
for e in ("image", "text", "fused"):
    tchk(f"{e} & 5-NN majority &", ["5-NN", f"{b3(e,5,'majority','dev_macro_f1'):.4f}", f"{b3(e,5,'majority','test_acc'):.4f}", f"{b3(e,5,'majority','test_macro_f1'):.4f}"])
    tchk(f"{e} & logistic regression &", [f"{lr(e,'dev_macro_f1'):.4f}", f"{lr(e,'test_acc'):.4f}", f"{lr(e,'test_macro_f1'):.4f}"])
for key, cfg in [("T2 L4a & full input", "c1_t2_L4a"), ("T2 L4 (v2) & full input", "c1_t2_L4")]:
    f_ = c1(cfg, "full")
    tchk(key, [f"& {len(f_)} &", ms(c1(cfg, "full", "acc")), ms(f_)])
    for cond, word in (("image_grey", "grey image"), ("text_empty", "empty text")):
        x = c1(cfg, cond)
        tchk(f"& {word} & {len(x)} & {ms(c1(cfg, cond, 'acc'))}", [ms(x), sg((x - f_).mean())], cover_anchor=True)
tchk("T3 5-NN & full input", ["5-NN", "& 1 &", f"{t3c['full_acc']:.3f}", f"{t3c['full_macro_f1']:.3f}"])
tchk(f"& grey image & 1 & {t3c['image_grey_acc']:.3f}", [f"{t3c['image_grey_macro_f1']:.3f}", sg(t3c['image_grey_macro_f1'] - t3c['full_macro_f1'])], cover_anchor=True)
tchk(f"& empty text & 1 & {t3c['text_empty_acc']:.3f}", [f"{t3c['text_empty_macro_f1']:.3f}", sg(t3c['text_empty_macro_f1'] - t3c['full_macro_f1'])], cover_anchor=True)
EV = ["california_wildfires", "hurricane_harvey", "hurricane_irma", "hurricane_maria", "iraq_iran_earthquake", "mexico_earthquake", "srilanka_floods"]
for t in ("t1", "t3"):
    for e in EV:
        d = ev(t, e)
        tchk(f"{e} & {d['n']} & {d['test_acc']:.3f} & {d['test_macro_f1']:.3f}", [f"{d['in_dist_macro_f1']:.3f}"], cover_anchor=True)
a_ = {t: c2(t) for t in ("t1", "t3")}
for t in ("t1", "t3"):
    hits = [(i, l) for i, l in TEX if l.startswith(chr(92) + f"textbf{{macro average}} & {a_[t]['n']} & {a_[t]['test_acc']:.3f} & {a_[t]['test_macro_f1']:.3f}")]
    tex_nchecks += 1
    if len(hits) != 1:
        tex_bad.append((f"macro average row ({t})", "found once", str(len(hits))))
    else:
        tchk(hits[0][1], [f"{a_[t]['in_dist_macro_f1']:.3f}"], cover_anchor=True)

# 3. captions
tchk("Task 2 ladder from the original notebook", f"({int(rows('t2_L4')[0]['n_test'])} rows)")
tchk("Task 3 ladder (deterministic", ["n=1", f"{T3['L4']['n_test']} test queries", f"({len(cc_T3_LABELS)} real classes"])
tchk("Task 2 component ablation", f"{rows('t2_L4a', 'A')[0]['flags']['epochs']} epochs")
tchk("vary one factor of the nearest-neighbour", f"{T3['L4']['n_test']} test queries")
tchk("effect of k", ["k=5", "n=1"])
tchk("voting rule", ["k=5", "n=1"])
tchk("embedding and classifier", "n=1")
tchk("Leave-one-disaster-out, T1", "n=1")
tchk("Leave-one-disaster-out, T3", "n=1")

# 4. prose claims
LAD = ["t2_L0", "t2_L1", "t2_L2", "t2_L3", "t2_L3b", "t2_L4a", "t2_L4"]
seedstd = [col(c, "test_macro_f1").std(ddof=1) for c in LAD]
pstd = [(col(b, "test_macro_f1") - col(a, "test_macro_f1")).std(ddof=1) for _, a, b in STEPS[:6]] + \
       [(col("t2_L4", "test_macro_f1") - col("t2_L3", "test_macro_f1")).std(ddof=1)]
pm_ = lambda a, b: (lambda d: f"{sg(d.mean())} ± {d.std(ddof=1):.3f}")(col(b, "test_macro_f1") - col(a, "test_macro_f1"))
tchk("Task 2 rungs are trained with three seeds", "(" + ", ".join(str(r["seed"]) for r in rows("t2_L4")) + ")")
tchk("Across the Task 2 rungs", [f"{min(seedstd):.3f}", f"{max(seedstd):.3f}", f"{min(pstd):.3f}", f"{max(pstd):.3f}", "below about 0.01"])  # last = stated convention
tchk("lists the rungs", f"{int(rows('t2_L4')[0]['n_test'])} images")
tchk("Restoring them raises", [ms(col("t2_L0", "test_macro_f1")), ms(col("t2_L1", "test_macro_f1")), pm_("t2_L0", "t2_L1")])
assert (col("t2_L1", "test_macro_f1") - col("t2_L0", "test_macro_f1") > 0).all()
tchk("Every later step moves", [pm_("t2_L1", "t2_L2"), pm_("t2_L2", "t2_L3"), pm_("t2_L3", "t2_L3b"), pm_("t2_L3b", "t2_L4a"),
                                pm_("t2_L4a", "t2_L4"), f"{rows('t2_L3')[0]['flags']['epochs']}-epoch"])
wall = lambda c: np.array([r["wall_clock_s"] for r in rows(c)]) / 60
ratio = wall("t2_L1").mean() / wall("t2_L4").mean()
assert 6.5 <= ratio < 7.5, ratio                                  # "about one seventh"
tchk("The rest of v2 therefore", [ms(col("t2_L1", "test_macro_f1")), ms(col("t2_L4", "test_macro_f1")),
                                  f"{wall('t2_L4').mean():.1f} instead of {wall('t2_L1').mean():.1f} minutes", "one seventh"])
assert ((col("t2_L4", "test_macro_f1") - col("t2_L3", "test_macro_f1")) < 0).all()
tchk("The whole bundle from L3 to L4", [pm_("t2_L3", "t2_L4")])
if "COS_PRE" in globals():
    tchk("v2's text encoder starts from random", f"{COS_PRE:.4f}")
else:
    tex_bad.append(("embedding cosine", "computed", "pretrained weights unavailable"))
tchk("follows the nearest-neighbour index", f"{T3['L4']['n_test']} test queries")
dk_rows = T3["L0"]["index_rows"] - (T3["L2"]["index_rows"] + T3["L4"]["n_test"])
tchk("The original index held all", [str(T3["L0"]["index_rows"]), str(dk_rows), f"{100 * dk_rows / T3['L0']['index_rows']:.1f}%"])
assert T3["L0"]["test_queries_with_self_in_top5"] == T3["L0"]["n_test"]
tchk("The original notebook's score is not", [f"all {T3['L0']['n_test']} test queries", f"{T3['L0']['test_macro_f1']:.3f}", f"{T3['L0']['test_acc']:.3f}",
                                              f"{T3['L1']['test_macro_f1']:.3f}", f"{T3['L1']['test_acc']:.3f}"])
tchk("L1 is therefore the fair measure", [f"{T3['L2']['test_macro_f1']:.3f}", f"{T3['L3']['test_macro_f1']:.3f}", f"{T3['L4']['test_macro_f1']:.3f}",
                                          f"noise of {T3['L4']['n_test']} queries"])
Pn = rows("t2_L4a", "A")[0]["n_params"]
pairB = lambda c: col(c, "test_macro_f1", "B") - col("t2_L4a", "test_macro_f1", "A")
tchk("A 200k-parameter MLP", [f"{round(rows('t2_image_only', 'B')[0]['n_params'] / 1000)}k-parameter", f"{round(Pn / 1e6)}M-parameter",
                              ms(col("t2_image_only", "test_macro_f1", "B")), ms(col("t2_L4a", "test_macro_f1", "A")),
                              f"{sg(pairB('t2_image_only').mean())} ± {pairB('t2_image_only').std(ddof=1):.3f}",
                              f"{sg(pairB('t2_concat').mean())} ± {pairB('t2_concat').std(ddof=1):.3f}"])
tchk("The text stream alone reaches only", [ms(col("t2_text_only", "test_macro_f1", "B")), f"{sg(pairB('t2_text_only').mean())} ± {pairB('t2_text_only').std(ddof=1):.3f}"])
assert (pairB("t2_no_t2i") < 0).all()
tchk("Removing the text-to-image attention", f"{sg(pairB('t2_no_t2i').mean())} ± {pairB('t2_no_t2i').std(ddof=1):.3f}")
devk = [b3("fused", k, "majority", "dev_macro_f1") for k in KS]
testk = [b3("fused", k, "majority", "test_macro_f1") for k in KS]
assert KS[int(np.argmax(devk))] == 5
tchk("The value k=5 is the best", ["k=5", f"{max(devk):.3f}", f"{min(testk):.3f}", f"{max(testk):.3f}", "\{" + ", ".join(map(str, KS)) + "\}"])
tchk("Similarity-weighted voting gives metrics identical", ["15 combinations", f"{SPREAD:.3f}"])
tchk("Class-balanced voting raises", [f"{b3('fused',5,'class_balanced','dev_macro_f1') - b3('fused',5,'majority','dev_macro_f1'):.3f}",
                                      f"{b3('fused',5,'class_balanced','test_macro_f1') - b3('fused',5,'majority','test_macro_f1'):.3f}",
                                      f"{b3('fused',5,'majority','test_acc'):.3f}", f"{b3('fused',5,'class_balanced','test_acc'):.3f}"])
tchk("Image-only retrieval", [f"{b3('image',5,'majority','test_macro_f1'):.3f}", f"{b3('fused',5,'majority','test_macro_f1'):.3f}",
                              f"{b3('text',5,'majority','test_macro_f1'):.3f}"])
_e1f = json.load(open(RES / "X" / "e1_t3_probe_vs_knn_fused_seed42.json"))
_e1i = json.load(open(RES / "X" / "e1_t3_probe_vs_knn_image_seed42.json"))
assert _e1f["ci_excludes_zero"] and _e1i["ci_excludes_zero"] and _e1f["n_boot"] == _e1i["n_boot"] == 10000      # "both exclude zero"
_iv = lambda r: f"{r['diff']:+.3f} ({r['ci95_low']:+.3f} to {r['ci95_high']:+.3f})"
tchk("A logistic-regression probe", [f"{lr('fused','test_macro_f1'):.3f}", f"{b3('fused',5,'majority','test_macro_f1'):.3f}",
                                     f"{lr('image','test_macro_f1'):.3f}", f"{b3('image',5,'majority','test_macro_f1'):.3f}",
                                     "bootstrap 95%", _iv(_e1f), _iv(_e1i)])
g4, g4a = c1("c1_t2_L4", "image_grey"), c1("c1_t2_L4a", "image_grey")
assert max(np.abs(np.concatenate([g4, g4a]) - maj)) <= 0.002
tchk("A grey image collapses", [f"{g4.mean():.3f}", f"{g4a.mean():.3f}", "within 0.002", f"{maj:.4f}"])
dl4 = c1("c1_t2_L4", "text_empty") - c1("c1_t2_L4", "full")
dl4a = c1("c1_t2_L4a", "full") - c1("c1_t2_L4a", "text_empty")
tchk("Removing the text has a model-dependent effect", [sg(dl4.mean()), f"{c1('c1_t2_L4','full').mean():.3f} to {c1('c1_t2_L4','text_empty').mean():.3f}",
                                                         f"{dl4a.min():.3f} and {dl4a.max():.3f}", f"-{dl4a.mean():.3f}"])
tchk("We read this as redundancy", [f"{col('t2_text_only', 'test_macro_f1', 'B').mean():.3f}"])
tchk("For Task 3 a grey image costs", [f"{t3c['full_macro_f1'] - t3c['image_grey_macro_f1']:.3f}", f"{t3c['full_macro_f1'] - t3c['text_empty_macro_f1']:.3f}"])
nfold = len(glob.glob(str(RES / "C" / "c2_t[13]_*_seed42.json"))) - 2
tchk("which the code asserts for all", f"all {nfold} folds")
tchk("Averaged over the seven events", [f"{a_['t1']['in_dist_macro_f1']:.3f}", f"{a_['t1']['test_macro_f1']:.3f}",
                                        f"{a_['t3']['in_dist_macro_f1']:.3f}", f"{a_['t3']['test_macro_f1']:.3f}"])
lower = lambda t: sum(ev(t, e)["test_macro_f1"] < ev(t, e)["in_dist_macro_f1"] for e in EV)
drops1 = {e: ev("t1", e)["in_dist_macro_f1"] - ev("t1", e)["test_macro_f1"] for e in EV}
big = max(drops1, key=drops1.get)
n3 = sorted(ev("t3", e)["n"] for e in EV)
assert n3[3] > n3[2] and big == "srilanka_floods"
# Task 1: the drop is concentrated in Sri Lanka floods; five of the other six events change by less than 0.03 (stated threshold)
others = [e for e in EV if e != "srilanka_floods"]
small = sum(abs(drops1[e]) < 0.03 for e in others)
assert small == 5 and drops1["srilanka_floods"] > 0.1 and big == "srilanka_floods"
word = {5: "five"}[small]
drops3 = {e: ev("t3", e)["in_dist_macro_f1"] - ev("t3", e)["test_macro_f1"] for e in EV}
big3 = max(drops3, key=drops3.get)
assert min(drops3.values()) < 0 < max(drops3.values()) and big3 == "hurricane_irma"      # mixed signs; largest drop is Irma
pooled = json.load(open(ROOT / "results" / "metrics.json"))
tchk("The pooled headline scores are", [f"{pooled['T1_test']['macro_f1']:.3f}", f"{pooled['T3_test']['macro_f1']:.3f}"])
tchk("We observe that the Task 1 drop", [f"{ev('t1', big)['in_dist_macro_f1']:.3f} to {ev('t1', big)['test_macro_f1']:.3f}", f"{word} of the other six", "less than 0.03"])
tchk("For Task 3 the per-event changes", [f"{ev('t3', big3)['in_dist_macro_f1']:.3f} to {ev('t3', big3)['test_macro_f1']:.3f}", f"{n3[0]} to {n3[2]}"])
gap = lambda r: col(f"t2_{r}", "dev_macro_f1").mean() - col(f"t2_{r}", "test_macro_f1").mean()
tchk("Development macro-F1 for L0 to L2", [f"best of {rows('t2_L2')[0]['flags']['epochs']} epochs", f"{gap('L0'):.3f}", f"{gap('L1'):.3f}", f"{gap('L2'):.3f}",
                                           f"{gap('L3'):+.3f}", f"{rows('t2_L3')[0]['flags']['epochs']}-epoch"])
tchk("we regard only the following as established", [sg(col("t2_L1", "test_macro_f1").mean() - col("t2_L0", "test_macro_f1").mean()), "about 0.01"])
tchk("Differences of about 0.01", "about 0.01")

# 4b. A1 table, the rewritten agreed/disagreed sentences and the E1 bootstrap intervals (all recomputed from results/ablation/X)
E1N = ["t3_probe_vs_knn_fused", "t3_probe_vs_knn_image", "t3_L4_vs_L1", "t2_imageonly_vs_L4a", "t2_agreed_acc_L4a_vs_imageonly"]
E1 = {n: json.load(open(RES / "X" / f"e1_{n}_seed42.json")) for n in E1N}
for n, r in E1.items():
    assert r["n_boot"] == 10_000, n
xs = {sub: x0["subsets"][sub]["n"] for sub in ("agreed", "disagreed", "all")}
ANAME = {"L4a": "L4a (full)", "image_only": "image only", "text_only": "text only"}
for sub, lab in (("agreed", "agreed"), ("disagreed", "disagreed"), ("all", "all rows")):
    tchk(f"{lab} & {xs[sub]} & majority class & -- & {mbx[sub]['acc']:.3f}", [f"{mbx[sub]['macro_f1']:.3f}"], cover_anchor=True)
    for m in ("L4a", "image_only", "text_only"):
        tchk(f"& & {ANAME[m]} & {len(X1[m])} & {ms(xg(m, sub, 'acc'))}", [ms(xg(m, sub, "macro_f1"))], cover_anchor=True)
sg3 = lambda v: f"{v:+.3f}"
agr_acc = xd("L4a", "image_only", "agreed", "acc")
assert (agr_acc > 0).all() and E1["t2_agreed_acc_L4a_vs_imageonly"]["ci_excludes_zero"]          # "positive at all three seeds ... established"
dis_f1 = xd("L4a", "image_only", "disagreed", "macro_f1")
tchk("Over the whole test set the text adds", [sg3(pairB("t2_image_only").mean())])
tchk("The effect depends on whether the two labels agree",
     [f"{x0['n_agreed']} rows", sg3(agr_acc.mean()), f"{x0['n_disagreed']:,} disagreed rows", sg3(dis_f1.mean()),
      f"paired bootstrap 95% interval of {sg3(E1['t2_agreed_acc_L4a_vs_imageonly']['ci95_low'])} to {sg3(E1['t2_agreed_acc_L4a_vs_imageonly']['ci95_high'])}",
      f"at seed {E1['t2_agreed_acc_L4a_vs_imageonly']['seed']}"])
f1b, imb = E1["t3_probe_vs_knn_fused"], E1["t3_probe_vs_knn_image"]
tchk("The bootstrap resamples the", [f"{f1b['n_rows']} test queries"])

# 4c. the L1 - L0 bootstrap sentence and the noise-floor / test-sample-uncertainty sentence
_l = json.load(open(RES / "X" / "e1_t2_L1_vs_L0_seed42.json"))
assert _l["ci_excludes_zero"]
assert abs(_l["regen_macro_f1_L0_crop_seeds"][0] - _l["stored_macro_f1_L0"]) < 0.002 and abs(_l["regen_macro_f1_L1_crop_seeds"][0] - _l["stored_macro_f1_L1"]) < 0.002
tchk("At seed 42 the paired bootstrap 95% interval of this gain", ["At seed 42", "95% interval", f"{sg3(_l['ci95_low'])} to {sg3(_l['ci95_high'])}", "less than 0.002"])
_iv2 = json.load(open(RES / "X" / "e1_t2_imageonly_vs_L4a_seed42.json"))
assert not _iv2["ci_excludes_zero"]
tchk("This noise floor is seed variability", ["95% bootstrap interval", "at seed 42", f"half-width of {(_iv2['ci95_high'] - _iv2['ci95_low']) / 2:.3f}"])

# 4d. "Confidence and calibration" paragraph (recomputed from results/ablation/X/c_calibration_*)
_t3c = json.load(open(RES / "X" / "c_calibration_t3_probe_seed42.json"))
_t2c = [json.load(open(RES / "X" / f"c_calibration_t2_L4a_seed{s_}.json")) for s_ in (42, 43, 44)]
_qq, _aa = _t3c["by_confidence_quintile"], _t3c["by_agreement"]
assert all(_qq[i]["accuracy"] < _qq[i + 1]["accuracy"] for i in range(4))                   # "rises monotonically"
assert all(r["mean_confidence"] > r["accuracy"] for r in _t2c) and _t3c["mean_confidence"] < _t3c["accuracy"]   # over- / under-confident
assert _t3c["auroc_confidence_ci95"][0] > 0.5 and _t3c["auroc_agreement_ci95"][0] > 0.5
_Ts = [r["temperature"] for r in _t2c]
tchk("We measure top-label expected calibration error", ["15 equal-width bins"])
tchk("Task 2 is over-confident", [f"{np.mean([r['mean_confidence'] for r in _t2c]):.3f} against an accuracy of {np.mean([r['accuracy'] for r in _t2c]):.3f}",
                                  f"T between {min(_Ts):.2f} and {max(_Ts):.2f}",
                                  f"from {np.mean([r['ece_before'] for r in _t2c]):.3f} to {np.mean([r['ece_after'] for r in _t2c]):.3f}"])
tchk("The Task 3 probe is under-confident", [f"{_t3c['mean_confidence']:.3f} against an accuracy of {_t3c['accuracy']:.3f}", f"T = {_t3c['temperature']:.2f}",
                                             f"from {_t3c['ece_before']:.3f} to {_t3c['ece_after']:.3f}"])
tchk("Both signals available to the hybrid track correctness",
     [f"{_aa['agree']['accuracy']:.3f} of the {_aa['agree']['n']} test queries", f"{_aa['disagree']['accuracy']:.3f} of the {_aa['disagree']['n']}",
      f"from {_qq[0]['accuracy']:.3f} to {_qq[-1]['accuracy']:.3f}"])
tchk("As predictors of a correct answer",
     [f"AUROC of {_t3c['auroc_confidence']:.3f} (95% bootstrap interval {_t3c['auroc_confidence_ci95'][0]:.3f} to {_t3c['auroc_confidence_ci95'][1]:.3f})",
      f"{_t3c['auroc_agreement']:.3f} ({_t3c['auroc_agreement_ci95'][0]:.3f} to {_t3c['auroc_agreement_ci95'][1]:.3f})"])

# 5. coverage: every numeric token in the .tex must lie inside a verified span (or be structural LaTeX such as labels)
STRUCT = [r"\\ref\{[^}]*\}", r"\\label\{[^}]*\}", r"\\begin\{tabular\}\{[^}]*\}", r"\\begin\{table\}\[[^\]]*\]", r"Tasks? \d\.?"]
uncovered = []
for i, l in TEX:
    masked = l
    for pat in STRUCT:
        masked = re.sub(pat, lambda m: " " * len(m.group(0)), masked)
    for m in re.finditer(r"(?<![A-Za-z\d.\\])[-+]?\d[\d,.]*\d|(?<![A-Za-z\d.\\])\d", masked):
        a, b = m.span()
        if not any(sa <= a and b <= sb for sa, sb in tex_spans[i]):
            uncovered.append((i + 1, m.group(0), l[:110]))
print(f"report/ablation_section.tex: {tex_nchecks} numeric checks, {len(tex_bad)} mismatches, {len(uncovered)} numeric tokens not covered by any check")
for a1, e1, f1 in tex_bad:
    print(f"  TEX MISMATCH [{a1}] expected {e1!r} | line: {f1}")
for ln, tok, ctx in uncovered:
    print(f"  UNCOVERED line {ln}: {tok!r} in: {ctx}")
bad.extend(("tex:" + a1, e1, f1) for a1, e1, f1 in tex_bad)
bad.extend(("tex-uncovered", tok, ctx) for _, tok, ctx in uncovered)

# ── report ────────────────────────────────────────────────────────────────────
print(f"{len(checks)} numeric checks against docs/ablation_findings.md; {len(bad)} mismatches")
for k, e, found in bad:
    print(f"  MISMATCH [{k}] expected {e!r} | line: {found}")
if "--write-log" in sys.argv:
    lines = [f"{'OK ' if ok else 'BAD'} [{k}] {e}" for k, e, ok in checks]
    lines += [f"{'OK ' if ok else 'BAD'} [report/ablation_section.tex: {k}] {e}" for k, e, ok in tex_log]
    (RES / "findings_check.txt").write_text("\n".join(lines) + "\n", encoding="utf8")
sys.exit(1 if bad else 0)
