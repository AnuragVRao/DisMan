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

# ── §9.5 D: Task 2 leave-one-disaster-out (image-only proxy), recomputed from results/ablation/X/d_lodo_t2_* ──────────────
_DE = ["california_wildfires", "hurricane_harvey", "hurricane_irma", "hurricane_maria", "iraq_iran_earthquake", "mexico_earthquake", "srilanka_floods"]
_DR = {e: [json.load(open(f)) for f in sorted(glob.glob(str(RES / "X" / f"d_lodo_t2_{e}_seed*.json")))] for e in _DE}
_dov = json.load(open(RES / "X" / "d_lodo_t2_overall_seed42.json"))
_bt = _dov["bootstrap_lodo_minus_indist_seed42"]
assert all(len(v) == 3 for v in _DR.values()) and sum(v[0]["n_test"] for v in _DR.values()) == _dov["n_test"] == 2237
# design assertions recorded in every row: zero held-out rows in train and dev; no class without training rows; stop rule never fired
assert all(r["heldout_rows_in_train"] == 0 and r["heldout_rows_in_dev"] == 0 and min(r["train_class_counts"].values()) > 0 for v in _DR.values() for r in v)
assert all(r["lodo"]["macro_f1"] >= r["majority_event"]["macro_f1"] for v in _DR.values() for r in v)
_dms = lambda a: f"{np.mean(a):.3f} ± {np.std(a, ddof=1):.3f}"
_dd = {e: np.mean([r["lodo"]["macro_f1"] for r in _DR[e]]) - np.mean([r["indist"]["macro_f1"] for r in _DR[e]]) for e in _DE}
_ie2 = json.load(open(RES / "X" / "e1_t2_imageonly_vs_L4a_seed42.json"))
_pair = col("t2_image_only", "test_macro_f1", "B") - col("t2_L4a", "test_macro_f1", "A")
chk("The image-only model is a proxy for L4a", [f"{sg(_pair.mean())} ± {_pair.std(ddof=1):.3f}", f"{_ie2['ci95_low']:+.3f} to {_ie2['ci95_high']:+.3f}"])
_mc = {e: min(_DR[e][0]["train_class_counts"].items(), key=lambda kv: kv[1]) for e in _DE}
_emin = min(_DE, key=lambda e: _mc[e][1])
assert _emin == "mexico_earthquake" and _mc[_emin][0] == "Missing or found people"
chk("Asserted before training every fold", [f"in any fold's train rows is {_mc[_emin][1]}"])
_bases = [_DR[e][0]["majority_event"]["macro_f1"] for e in _DE]
chk("No fold-seed run fell below", [f"range {min(_bases):.3f} to {max(_bases):.3f}"])
for e in _DE:
    rs = _DR[e]; dl = np.array([r["lodo"]["macro_f1"] - r["indist"]["macro_f1"] for r in rs])
    chk(f"| {e} | {rs[0]['n_test']} | {len(rs)} |",
        [_dms([r["lodo"]["acc"] for r in rs]), _dms([r["lodo"]["macro_f1"] for r in rs]), _dms([r["indist"]["macro_f1"] for r in rs]),
         f"{dl.mean():+.3f} ± {dl.std(ddof=1):.3f}", f"| {rs[0]['majority_event']['macro_f1']:.3f} | {min(rs[0]['train_class_counts'].values())} |"])
_sd = [42, 43, 44]
_avg = lambda k, m: [np.mean([[r[k][m] for r in _DR[e] if r["seed"] == s_][0] for e in _DE]) for s_ in _sd]
_dm = np.array(_avg("lodo", "macro_f1")) - np.array(_avg("indist", "macro_f1"))
chk("| macro average |", [f"| {_dov['n_test']} | 3 |", _dms(_avg("lodo", "acc")), _dms(_avg("lodo", "macro_f1")), _dms(_avg("indist", "macro_f1")),
                          f"{_dm.mean():+.3f} ± {_dm.std(ddof=1):.3f}", f"{np.mean(_bases):.3f}"])
_pl = lambda k, m: [_dov["pooled_by_seed"][str(s_)][k][m] for s_ in _sd]
_dp = np.array(_pl("lodo", "macro_f1")) - np.array(_pl("indist", "macro_f1"))
chk("| pooled (all rows at once) |", [f"| {_dov['n_test']} | 3 |", _dms(_pl("lodo", "acc")), _dms(_pl("lodo", "macro_f1")), _dms(_pl("indist", "macro_f1")), f"{_dp.mean():+.3f} ± {_dp.std(ddof=1):.3f}"])
assert _bt["excludes_zero"] and _bt["n_boot"] == 10000
chk("Pooled comparison (seed 42)", [f"{_dov['n_test']:,}-row set", f"{_bt['lodo_macro_f1']:.4f} for LODO against {_bt['indist_macro_f1']:.4f}", f"{_bt['n_boot']:,} resamples",
                                    f"{_bt['diff']:+.4f}, 95% interval {_bt['ci95_low']:+.4f} to {_bt['ci95_high']:+.4f}"])
assert max(_dd, key=lambda e: -_dd[e]) == "iraq_iran_earthquake" and sorted(_dd, key=lambda e: _dd[e])[1] == "mexico_earthquake"       # the two earthquakes lose the most
assert all(abs(_dd[e]) < 0.02 for e in ("hurricane_harvey", "hurricane_irma", "hurricane_maria", "srilanka_floods"))
chk("Where the loss is", [f"Iraq–Iran {_dd['iraq_iran_earthquake']:+.3f}", f"Mexico {_dd['mexico_earthquake']:+.3f}", f"harvey {_dd['hurricane_harvey']:+.3f}", f"irma {_dd['hurricane_irma']:+.3f}",
                          f"maria {_dd['hurricane_maria']:+.3f}", f"srilanka {_dd['srilanka_floods']:+.3f}", f"changes by {_dd['california_wildfires']:+.3f}"])
chk("Caveats: per-event macro-F1 over rare classes", [f"seed spread is {np.std([r['lodo']['macro_f1'] for r in _DR['california_wildfires']], ddof=1):.3f}", f"{_DR['iraq_iran_earthquake'][0]['n_test']} test rows"])
chk("Run time:", [f"{_dov['total_wall_clock_s'] / 60:.1f} minutes"])

# ── §9.5 per-class pooled F1 (seed 42), recomputed from results/ablation/X/d_lodo_t2_perclass_seed42.json ─────────────
_pc2 = json.load(open(RES / "X" / "d_lodo_t2_perclass_seed42.json"))
for _r in _pc2["classes"]:
    if _r["cls"] == "Missing or found people":
        continue                    # its row is identical in the D and the L0/L1 tables (F1 0.000 everywhere); checked below as appearing exactly twice
    chk(f"| {_r['cls']} | {_r['support']} | {_r['support_earthquake']} | {_r['earthquake_fraction']:.3f} | {_r['f1_lodo']:.3f} | {_r['f1_indist']:.3f} |",
        [f"{_r['diff']:+.3f}", f"{100 * _r['share_of_total_drop']:.1f}%"])
_t = max(_pc2["classes"], key=lambda r: -r["diff"])
_pr = _pc2["macro_f1_lodo"] - _pc2["macro_f1_indist"]
_sd42 = [r["lodo"]["macro_f1"] - r["indist"]["macro_f1"] for r in (next(x for x in _DR_ if x["seed"] == 42) for _DR_ in _DR.values())]
chk("Per-class pooled F1 (seed 42", [f"{_pc2['earthquake_rows']} of the {_pc2['n_test']:,} test rows", f"{100 * _pc2['earthquake_rows'] / _pc2['n_test']:.1f}%"])
chk("Pooled macro-F1 is the mean of these eight F1 values", [f"{_pc2['macro_f1_lodo']:.4f} (LODO) against {_pc2['macro_f1_indist']:.4f}", f"a difference of {_pr:+.4f}"])
chk("The drop is concentrated in one class", [f"{_t['cls']} ({_t['support']} test rows, {_t['support_earthquake']} of them", f"from F1 {_t['f1_indist']:.3f} to {_t['f1_lodo']:.3f}",
                                             f"{100 * _t['share_of_total_drop']:.1f}% of the summed per-class loss", f"{_t['diff'] / 8:+.3f} of {_pr:+.3f}",
                                             f"would be {_pr - _t['diff'] / 8:+.3f}", f"({np.mean(_sd42):+.3f} at seed 42, {np.mean(_dm):+.3f} over three seeds)"])
chk("It is a single class", [f"{_pc2['classes'][6]['diff']:+.3f} with {_pc2['classes'][6]['earthquake_fraction']:.2f}", f"{_pc2['classes'][7]['diff']:+.3f} with {_pc2['classes'][7]['earthquake_fraction']:.2f}",
                             f"is {_pc2['pearson_eqfrac_vs_diff']:+.2f}"])
chk("Caveat: Injured or dead people has only", [f"{_t['support']} test rows", f"{_pc2['classes'][3]['support']}, F1 0 in both settings"])
assert _pc2["classes"][3]["f1_lodo"] == 0 == _pc2["classes"][3]["f1_indist"]

# ── §9.6 L0 vs L1 per class, the D caveat and the limitation, recomputed from results/ablation/X ─────────────────────────
_l01 = json.load(open(RES / "X" / "l0_l1_perclass_seed42.json"))
_e1l = json.load(open(RES / "X" / "e1_t2_L1_vs_L0_seed42.json"))
assert abs(_l01["cpu_macro_f1_L0"] - _e1l["regen_macro_f1_L0_crop_seeds"][0]) < 1e-12 and abs(_l01["cpu_macro_f1_L1"] - _e1l["regen_macro_f1_L1_crop_seeds"][0]) < 1e-12
_c01 = _l01["classes"]
_t01 = max(_c01, key=lambda r: r["diff"])
assert _t01["cls"] == "Injured or dead people" and _t01["share_of_total_gain"] > 0.5                 # "carries the majority of the gain"
_dt = max(_pc2["classes"], key=lambda r: -r["diff"])
assert _dt["cls"] == _t01["cls"]                                                                       # the same class dominates both experiments
chk("Per-class F1, T2 L0 vs L1 (seed 42, crop seed 0", [f"the {_l01['n_test']:,} test rows", f"{_l01['e1_macro_f1_L0']:.6f} and {_l01['e1_macro_f1_L1']:.6f}"])
for _r in _c01:
    if _r["cls"] == "Missing or found people":
        continue
    chk(f"| {_r['cls']} | {_r['support']} | {_r['support_earthquake']} | {_r['earthquake_fraction']:.3f} | {_r['f1_L0']:.3f} | {_r['f1_L1']:.3f} |",
        [f"{_r['diff']:+.3f}", f"{100 * _r['share_of_total_gain']:.1f}%"])
# the "Missing or found people" row: identical in the D table (LODO vs in-dist.) and the L0/L1 table, so it must appear exactly twice
_mr_d = next(r for r in _pc2["classes"] if r["cls"] == "Missing or found people"); _mr_l = next(r for r in _c01 if r["cls"] == "Missing or found people")
_mrow = (f"| Missing or found people | {_mr_d['support']} | {_mr_d['support_earthquake']} | {_mr_d['earthquake_fraction']:.3f} | {_mr_d['f1_lodo']:.3f} | {_mr_d['f1_indist']:.3f} | "
         f"{_mr_d['diff']:+.3f} | {100 * _mr_d['share_of_total_drop']:.1f}% |")
_mrow_l = (f"| Missing or found people | {_mr_l['support']} | {_mr_l['support_earthquake']} | {_mr_l['earthquake_fraction']:.3f} | {_mr_l['f1_L0']:.3f} | {_mr_l['f1_L1']:.3f} | "
           f"{_mr_l['diff']:+.3f} | {100 * _mr_l['share_of_total_gain']:.1f}% |")
_n_rows = sum(1 for _l in LINES if _l.strip() == _mrow) if _mrow == _mrow_l else -1
checks.append(("Missing-or-found row appears in both per-class tables", _mrow, _n_rows == 2))
if _n_rows != 2:
    bad.append(("Missing-or-found row", "appears exactly twice (D table and L0/L1 table)", str(_n_rows)))
chk(f"Macro-F1 {_l01['cpu_macro_f1_L0']:.4f} to {_l01['cpu_macro_f1_L1']:.4f}", [f"a gain of {_l01['cpu_macro_f1_L1'] - _l01['cpu_macro_f1_L0']:+.4f}"])
chk("carries the majority of the gain", [f"{_t01['diff']:+.3f} F1 ({_t01['f1_L0']:.3f} to {_t01['f1_L1']:.3f})", f"{100 * _t01['share_of_total_gain']:.1f}% of the summed per-class gain",
                                        f"({_t01['support']} test rows, {_t01['support_earthquake']} of them", f"({100 * _dt['share_of_total_drop']:.1f}% of the summed loss)"])
chk("The gain is not confined to that class", [f"{_c01[0]['cls']} {_c01[0]['diff']:+.3f} ({100 * _c01[0]['share_of_total_gain']:.1f}% of the gain)",
                                              f"{_c01[6]['cls']} {_c01[6]['diff']:+.3f} ({100 * _c01[6]['share_of_total_gain']:.1f}%)",
                                              f"with {_c01[0]['earthquake_fraction']:.2f} and {_c01[6]['earthquake_fraction']:.2f}"])
_rest2 = _pc2["macro_f1_lodo"] - _pc2["macro_f1_indist"] - _dt["diff"] / 8
chk("The pooled leave-one-disaster-out drop (§9.5)", [f"one {_dt['support']}-row class ({abs(_dt['diff'] / 8):.3f} of the {abs(_pc2['macro_f1_lodo'] - _pc2['macro_f1_indist']):.3f} pooled drop at seed 42)",
                                                       f"is {_rest2:+.3f}", f"over a {_dt['support']}-row class"])
# the limitation: "Missing or found people" scores F1 = 0 in EVERY Task 2 condition
_MC = "Missing or found people"
_zn = 0
for _f in glob.glob(str(RES / "[AB]" / "t2_*_seed*.json")):
    _d = json.load(open(_f))
    if "per_class_f1" in _d:
        _zn += 1
        assert _d["per_class_f1"][_MC] == 0, _f
assert _zn == 33
assert all(r["f1_L0"] == 0 and r["f1_L1"] == 0 for r in _c01 if r["cls"] == _MC)
sys.path.insert(0, str(ROOT / "ablation"))
from common import cc as _cc0, DATA as _D0, SPLIT_DIR as _S0
_te_df = _cc0.load_split("humanitarian", "test", _D0, _S0, verbose=False)[0]
_labels8 = sorted(_cc0.load_split("humanitarian", "train", _D0, _S0, verbose=False)[0]["label"].unique())
_yy = np.array([_labels8.index(l_) for l_ in _te_df["label"]])
_mi = _labels8.index(_MC)
assert int((_yy == _mi).sum()) == 3 == next(r["support"] for r in _c01 if r["cls"] == _MC)
from sklearn.metrics import f1_score as _f1
for _seed_z in (42, 43, 44):
    _lp, _ip = np.full(len(_yy), -1), np.full(len(_yy), -1)
    for _e in sorted(_te_df.event_name.unique()):
        _dd2 = json.load(open(RES / "X" / f"d_lodo_t2_{_e}_seed{_seed_z}.json")); _m = (_te_df.event_name == _e).to_numpy()
        _lp[_m], _ip[_m] = _dd2["pred"], _dd2["indist_pred"]
    for _p in (_lp, _ip):
        assert _f1(_yy, _p, labels=list(range(8)), average=None, zero_division=0)[_mi] == 0.0
chk("Limitation.", [f'"{_MC}" ({_c01[3]["support"]} test rows) has F1 = 0 in every Task 2 condition', f"all {_zn} ladder and component-ablation runs", f"7/8 = {7 / 8:.3f}"])

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
for blk in [m_.group(0) for m_ in re.finditer(r"\\begin\{(table\*?)\}.*?\\end\{\1\}", TEX_RAW, re.S)]:
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
tchk("Averaged over the seven events, Task 1", [f"{a_['t1']['in_dist_macro_f1']:.3f}", f"{a_['t1']['test_macro_f1']:.3f}",
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

# 4e. Task 2 leave-one-disaster-out: table rows, caption and prose (recomputed from results/ablation/X/d_lodo_t2_*)
_bs = chr(92) + "textbf{macro average}"
for e in _DE:
    rs = _DR[e]
    tchk(f"{e} & {rs[0]['n_test']} & {len(rs)} & {_dms([r['lodo']['acc'] for r in rs])}",
         [_dms([r["lodo"]["macro_f1"] for r in rs]), _dms([r["indist"]["macro_f1"] for r in rs]), f"{rs[0]['majority_event']['macro_f1']:.3f}"], cover_anchor=True)
tchk(f"{_bs} & {_dov['n_test']} & 3 & {_dms(_avg('lodo', 'acc'))}", [_dms(_avg("lodo", "macro_f1")), _dms(_avg("indist", "macro_f1")), f"{np.mean(_bases):.3f}"], cover_anchor=True)
tchk(f"pooled & {_dov['n_test']} & 3 & {_dms(_pl('lodo', 'acc'))}", [_dms(_pl("lodo", "macro_f1")), _dms(_pl("indist", "macro_f1"))], cover_anchor=True)
tchk("Leave-one-disaster-out, Task 2, with the image-only model", [f"all {_dov['n_test']} test rows", "the 8 classes"])
_ap = _pair
tchk("Task 2 is evaluated the same way", [f"{sg3(_ap.mean())} ± {_ap.std(ddof=1):.3f}", f"{sg3(_ie2['ci95_low'])} to {sg3(_ie2['ci95_high'])}", "at seed 42"])
tchk("Each fold is run with seeds", ["seeds 42, 43 and 44"])
tchk("Averaged over the seven events the macro-F1 falls", [f"from {np.mean(_avg('indist', 'macro_f1')):.3f} to {np.mean(_avg('lodo', 'macro_f1')):.3f}",
                                                          f"all {_dov['n_test']} test rows at once gives {_bt['lodo_macro_f1']:.3f} against {_bt['indist_macro_f1']:.3f} at seed 42",
                                                          f"a difference of {sg3(_bt['diff'])}", f"95% interval of {sg3(_bt['ci95_low'])} to {sg3(_bt['ci95_high'])}"])
_mi = lambda e, k: np.mean([r[k]["macro_f1"] for r in _DR[e]])
assert all(abs(_dd[e]) < 0.02 for e in ("hurricane_harvey", "hurricane_irma", "hurricane_maria", "srilanka_floods"))
tchk("The loss is concentrated in the two earthquake events",
     [f"Iraq--Iran {_mi('iraq_iran_earthquake', 'indist'):.3f} to {_mi('iraq_iran_earthquake', 'lodo'):.3f}", f"Mexico {_mi('mexico_earthquake', 'indist'):.3f} to {_mi('mexico_earthquake', 'lodo'):.3f}",
      "less than 0.02", f"wildfire change ({sg3(_dd['california_wildfires'])}) is within its seed spread (± {np.std([r['lodo']['macro_f1'] for r in _DR['california_wildfires']], ddof=1):.3f})"])
assert abs(_dd["california_wildfires"]) < np.std([r["lodo"]["macro_f1"] for r in _DR["california_wildfires"]], ddof=1)      # "within its seed spread"
# per-class analysis (seed 42, pooled rows) and the two sentences that rest on it
_pc = json.load(open(RES / "X" / "d_lodo_t2_perclass_seed42.json"))
_top = max(_pc["classes"], key=lambda r: -r["diff"])
_rest = _pc["macro_f1_lodo"] - _pc["macro_f1_indist"] - _top["diff"] / 8
assert _top["cls"] == "Injured or dead people" and _top["earthquake_fraction"] > 0.5 and _top["share_of_total_drop"] > 0.7
assert abs(_pc["macro_f1_lodo"] - _bt["lodo_macro_f1"]) < 1e-12 and abs(_pc["macro_f1_indist"] - _bt["indist_macro_f1"]) < 1e-12
assert abs(_rest) < abs(np.mean(_dm))                                          # "without it the pooled drop ... is smaller than the per-event average"
tchk("The pooled drop is larger than the per-event average",
     [f"({sg3(_dp.mean())} against {sg3(_dm.mean())} over three seeds)", f"({_top['support']} test rows, {_top['support_earthquake']} of them from the two earthquake events)",
      f"falls from {_top['f1_indist']:.3f} to {_top['f1_lodo']:.3f} F1 at seed 42", f"{100 * _top['share_of_total_drop']:.0f}% of the summed per-class loss", f"at seed 42 would be {sg3(_rest)}"])
assert abs(np.mean(_dp)) > abs(np.mean(_dm))                                    # the pooled drop exceeds the per-event average
tchk("L0 was trained without both earthquake events", [f"the L0 to L1 gain ({sg3(np.mean(col('t2_L1', 'test_macro_f1') - col('t2_L0', 'test_macro_f1')))})"])
tchk("No fold falls below its event's majority-class macro-F1", [f"({min(_bases):.3f} to {max(_bases):.3f})"])


# 4f. ladder link by class, the pooled-drop caveat (twice) and the limitation
tchk("In both experiments the same class dominates at seed 42",
     ["at seed 42", f"{_dt['support_earthquake']} of whose {_dt['support']} test rows", f"{100 * _t01['share_of_total_gain']:.0f}% of the summed per-class gain from L0 to L1",
      f"{100 * _dt['share_of_total_drop']:.0f}% of the summed per-class loss"])
tchk("The pooled bootstrap interval is therefore driven mainly by", [f"that one {_dt['support']}-row class ({abs(_dt['diff'] / 8):.3f} of the {abs(_pc2['macro_f1_lodo'] - _pc2['macro_f1_indist']):.3f})",
                                                                      f"over a {_dt['support']}-row class"])
tchk("The pooled Task 2 leave-one-disaster-out drop", [f"({sg3(_pc2['macro_f1_lodo'] - _pc2['macro_f1_indist'])} at seed 42", f"one {_dt['support']}-row class ({abs(_dt['diff'] / 8):.3f} of the {abs(_pc2['macro_f1_lodo'] - _pc2['macro_f1_indist']):.3f})",
                                                       f"without it the drop is {sg3(_rest2)}", f"over a {_dt['support']}-row class"])
assert _bt["excludes_zero"]
tchk("has F1 = 0 in every Task 2 condition", ["F1 = 0", f"({_c01[3]['support']} test rows)", f"about {7 / 8:.3f}"])

# 4g. B: CLIP ViT-L/14 image encoder (recomputed from results/ablation/X/b_* and e1_*)
_bx = RES / "X"
_bT2 = {c_: [json.load(open(f_)) for f_ in sorted(glob.glob(str(_bx / f"b_t2_imageonly_{c_}_seed*.json")))] for c_ in ("B32_raw", "L14_norm", "L14_raw")}
_bT3 = {(e_, m_, c_): json.load(open(_bx / f"b_t3_{e_}_{m_}_{c_}_seed42.json")) for e_ in ("B32", "L14") for m_ in ("fused", "image") for c_ in ("knn5", "probe")}
_bZ = json.load(open(_bx / "b_zeroshot_l14_seed42.json"))
_bS = {n_: json.load(open(_bx / f"e1_{n_}_seed42.json")) for n_ in
       ("t2_imageonlyL14_vs_B32", "t2_imageonlyL14raw_vs_B32", "t2_imageonlyL14_vs_L4a", "t3_probeL14_vs_B32_fused", "t3_knnL14_vs_B32_fused")}
_bf = lambda c_: np.array([r_["test_macro_f1"] for r_ in _bT2[c_]])
_bl4a = np.array([json.load(open(f_))["test_macro_f1"] for f_ in sorted(glob.glob(str(RES / "A" / "t2_L4a_seed*.json")))])
_bpair = _bf("L14_norm") - _bf("B32_raw")
assert len(_bpair) == 3 and (_bpair > 0).all()                                      # "positive at all three seeds"
assert _bS["t2_imageonlyL14_vs_B32"]["ci_excludes_zero"] and _bS["t2_imageonlyL14raw_vs_B32"]["ci_excludes_zero"] and not _bS["t2_imageonlyL14_vs_L4a"]["ci_excludes_zero"]
assert not _bS["t3_probeL14_vs_B32_fused"]["ci_excludes_zero"] and not _bS["t3_knnL14_vs_B32_fused"]["ci_excludes_zero"]
assert abs(_bS["t2_imageonlyL14_vs_B32"]["ci95_low"]) < 0.005                         # "only just excludes zero"
assert _bZ["chosen_l14"] == _bZ["chosen_b32"] and _bZ["test_l14_chosen"]["macro_f1"] > 0.3333 > _bZ["test_b32_zero_shot"]["macro_f1"] > 0.2571
assert _bZ["test_l14_chosen"]["macro_f1"] < _bT3[("L14", "fused", "probe")]["test_macro_f1"]
assert _bT3[("L14", "image", "knn5")]["test_macro_f1"] > _bT3[("B32", "image", "knn5")]["test_macro_f1"] and _bT3[("L14", "image", "probe")]["test_macro_f1"] > _bT3[("B32", "image", "probe")]["test_macro_f1"]
assert _bT3[("L14", "fused", "probe")]["test_macro_f1"] < _bT3[("B32", "fused", "probe")]["test_macro_f1"]
_bd = lambda n_: sg3(_bS[n_]["diff"])
_bci = lambda n_: f"({sg3(_bS[n_]['ci95_low'])} to {sg3(_bS[n_]['ci95_high'])})"
_bcib = lambda n_: f"[{sg3(_bS[n_]['ci95_low'])}, {sg3(_bS[n_]['ci95_high'])}]"
# model sizes from the HF configs (shapes only, no weights)
import torch as _btorch
from transformers import CLIPConfig as _bCC, CLIPModel as _bCM


def _bparams(src_):
    cfg_ = _bCC.from_pretrained(src_)
    with _btorch.device("meta"):
        m_ = _bCM(cfg_)
    return (sum(p_.numel() for p_ in m_.vision_model.parameters()) + sum(p_.numel() for p_ in m_.visual_projection.parameters())) / 1e6, cfg_.vision_config.patch_size, cfg_.vision_config.num_hidden_layers


_bP32, _bpp32, _bll32 = _bparams(_ART / "clip")
_bP14, _bpp14, _bll14 = _bparams("openai/clip-vit-large-patch14")
# the one file under two keys: cached vectors compared
_bvec = lambda root_, task_, split_, key_, raw_=False: np.load(root_ / f"clip_{task_}_{split_}{'_raw' if raw_ else ''}.npy", mmap_mode="r")[json.load(open(root_ / f"clip_{task_}_{split_}.ids.json")).index(key_)].astype("float64")
_bKD, _bKH = "901671684478029825_0", "905274232590004225_0"
_bd32 = float(np.abs(_bvec(_ART / "emb", "damage", "train", _bKD) - _bvec(_ART / "emb", "humanitarian", "test", _bKH)).max())
_bd14 = float(np.abs(_bvec(_ART / "ablation" / "clip_l14", "damage", "train", _bKD, True) - _bvec(_ART / "ablation" / "clip_l14", "humanitarian", "test", _bKH, True)).max())
# findings document
_bex = json.load(open(_bx / "b_clip_l14_extract_seed42.json"))
_btm = json.load(open(_bx / "b_clip_l14_timing_seed42.json"))
chk("Setup. HF `openai/clip-vit-large-patch14`", [f"Embedded {_bex['n_unique_files']:,} unique files", f"{_bex['n_cached_rows']:,} rows", f"{_btm['n_unique_images']:,} unique `image_id` keys",
                                                  f"in {_bex['extraction_minutes']:.1f} minutes", f"{_btm['projected_sequential_min']:.1f}", f"{_btm['gpu_forward_s_median']:.2f} s", f"{_bex['unreadable_replaced_by_grey']} unreadable"])
chk("The encoders differ in size", [f"{_bP14:.1f}M parameters against {_bP32:.1f}M", f"{_bll14} layers against {_bll32}", f"{_bpp14} against {_bpp32} pixels"])
for c_, lab_ in (("B32_raw", "B/32 raw (v2's cached features, 512-d)"), ("L14_norm", "L/14 L2-normalised (768-d, as specified)"), ("L14_raw", "L/14 raw (768-d, control)")):
    r_ = _bT2[c_]
    chk(f"| {lab_} | 3 |", [ms(np.array([x_["dev_macro_f1"] for x_ in r_])), ms(np.array([x_["test_acc"] for x_ in r_])), ms(_bf(c_)), ms(np.array([x_["test_weighted_f1"] for x_ in r_]))])
chk("L/14 minus B/32, paired by seed", [", ".join(sg3(x_) for x_ in _bpair), f"mean {sg3(_bpair.mean())} ± {_bpair.std(ddof=1):.3f}", ms(_bf("L14_raw"))])
for e_ in ("B32", "L14"):
    for m_ in ("fused", "image"):
        for c_ in ("knn5", "probe"):
            r_ = _bT3[(e_, m_, c_)]
            chk(f"| {'B/32' if e_ == 'B32' else 'L/14'} | {'fused' if m_ == 'fused' else 'image only'} | {'5-NN majority' if c_ == 'knn5' else 'logistic probe (C = %g)' % r_['C']} |",
                [f"{r_['dev_macro_f1']:.4f}", f"{r_['test_acc']:.4f}", f"{r_['test_macro_f1']:.4f}"])
chk("Zero-shot Task 3 (identical prompts", [", ".join(f"{n_} {_bZ['dev_candidates_l14'][n_]['macro_f1']:.4f}" for n_ in ("E1", "E2", "E3")), ", ".join(f"{n_} {_bZ['dev_candidates_b32'][n_]:.4f}" for n_ in ("E1", "E2", "E3")),
                                           f"L/14 accuracy {_bZ['test_l14_chosen']['acc']:.4f}, macro-F1 {_bZ['test_l14_chosen']['macro_f1']:.4f}, weighted-F1 {_bZ['test_l14_chosen']['weighted_f1']:.4f}",
                                           f"B/32 accuracy {_bZ['test_b32_zero_shot']['acc']:.4f}, macro-F1 {_bZ['test_b32_zero_shot']['macro_f1']:.4f}, weighted-F1 {_bZ['test_b32_zero_shot']['weighted_f1']:.4f}",
                                           f"L/14 {_bZ['test_l14_dev_calibrated']['macro_f1']:.4f} against B/32 {_bZ['test_b32_dev_calibrated']['macro_f1']:.4f}"])
for n_ in ("t2_imageonlyL14_vs_B32", "t2_imageonlyL14_vs_L4a", "t3_probeL14_vs_B32_fused", "t3_knnL14_vs_B32_fused", "t2_imageonlyL14raw_vs_B32"):
    s_ = _bS[n_]
    chk(f"`{n_}`", [f"{s_['n_rows']:,}", f"{s_['score_a']:.4f}", f"{s_['score_b']:.4f}", f"{s_['diff']:+.4f}", f"[{s_['ci95_low']:+.4f}, {s_['ci95_high']:+.4f}]", "yes" if s_["ci_excludes_zero"] else "no"])
_bm = lambda a_, b_: f"{a_:.3f} to {b_:.3f}"
_bk = lambda m_, c_: _bm(_bT3[("B32", m_, c_)]["test_macro_f1"], _bT3[("L14", m_, c_)]["test_macro_f1"])
chk("Task 3: L/14 helps the image-only paths", [_bk("image", "knn5"), _bk("image", "probe")])
chk("The T2 image-only gain from L/14 is consistent", [f"{_bS['t2_imageonlyL14_vs_B32']['ci95_low']:+.4f} to {_bS['t2_imageonlyL14_vs_B32']['ci95_high']:+.4f}"])
chk("The 18,083rd key", [f"B/32 maximum difference {_bd32:.1e}", f"L/14 {_bd14:.1e}"])
# section prose
tchk("ViT-L/14 differs from ViT-B/32 in both size", [f"{_bP14:.0f}M parameters against {_bP32:.0f}M", f"{_bpp14} against {_bpp32} pixels"])
tchk("With the same head and recipe, the image-only Task 2 model gains",
     [f"gains {_bpair.mean():.3f} macro-F1 on average over three seeds ({_bf('B32_raw').mean():.3f} to {_bf('L14_norm').mean():.3f})", "at seed 42",
      f"{sg3(_bS['t2_imageonlyL14_vs_B32']['ci95_low'])} to {sg3(_bS['t2_imageonlyL14_vs_B32']['ci95_high'])}"])
tchk("Feeding the raw L/14 features", [f"gives {_bf('L14_raw').mean():.3f} on average, and at seed 42 a gain of {_bd('t2_imageonlyL14raw_vs_B32')} {_bci('t2_imageonlyL14raw_vs_B32')}"])
tchk("Against the full B/32 model (L4a)", [f"ahead by {_bS['t2_imageonlyL14_vs_L4a']['diff']:.3f} at seed 42", _bci("t2_imageonlyL14_vs_L4a")])
tchk("For Task 3 the larger encoder helps the image-only paths",
     [f"5-NN {_bk('image', 'knn5')}", f"logistic probe {_bk('image', 'probe')}",
      f"{_bT3[('L14', 'fused', 'probe')]['test_macro_f1']:.3f} against {_bT3[('B32', 'fused', 'probe')]['test_macro_f1']:.3f} ({_bd('t3_probeL14_vs_B32_fused')}, interval {sg3(_bS['t3_probeL14_vs_B32_fused']['ci95_low'])} to {sg3(_bS['t3_probeL14_vs_B32_fused']['ci95_high'])})",
      f"the fused 5-NN {_bT3[('L14', 'fused', 'knn5')]['test_macro_f1']:.3f} against {_bT3[('B32', 'fused', 'knn5')]['test_macro_f1']:.3f} ({_bd('t3_knnL14_vs_B32_fused')}, interval {sg3(_bS['t3_knnL14_vs_B32_fused']['ci95_low'])} to {sg3(_bS['t3_knnL14_vs_B32_fused']['ci95_high'])})"])
tchk("Zero-shot macro-F1, with identical prompts",
     [f"from {_bZ['test_b32_zero_shot']['macro_f1']:.3f} to {_bZ['test_l14_chosen']['macro_f1']:.3f} (accuracy {_bZ['test_b32_zero_shot']['acc']:.3f} to {_bZ['test_l14_chosen']['acc']:.3f})",
      "0.333", "0.257", f"({_bT3[('L14', 'fused', 'probe')]['test_macro_f1']:.3f})"])
tchk("Image encoder scale: CLIP ViT-B/32 against ViT-L/14, test macro-F1", ["three seeds", "n=1", "529 test queries", "at seed 42", "over the 529 queries", "(14 against 32)", "95% interval"])
tchk("Setting & n & B/32 (or L4a) & L/14 & Difference", ["95% interval"])
# section table rows
tchk(f"T2 image-only (L/14 L2-normalised) & 3 & {ms(_bf('B32_raw'))}", [ms(_bf("L14_norm")), _bd("t2_imageonlyL14_vs_B32"), _bcib("t2_imageonlyL14_vs_B32")], cover_anchor=True)
tchk(f"T2 image-only (L/14 raw input, control) & 3 & {ms(_bf('B32_raw'))}", [ms(_bf("L14_raw")), _bd("t2_imageonlyL14raw_vs_B32"), _bcib("t2_imageonlyL14raw_vs_B32")], cover_anchor=True)
tchk(f"T2 image-only L/14 against L4a (full model, B/32) & 3 & {ms(_bl4a)}", [ms(_bf("L14_norm")), _bd("t2_imageonlyL14_vs_L4a"), _bcib("t2_imageonlyL14_vs_L4a")], cover_anchor=True)
for lab_, m_, c_, n_ in (("T3 fused, logistic probe", "fused", "probe", "t3_probeL14_vs_B32_fused"), ("T3 fused, 5-NN", "fused", "knn5", "t3_knnL14_vs_B32_fused"),
                         ("T3 image-only, logistic probe", "image", "probe", None), ("T3 image-only, 5-NN", "image", "knn5", None)):
    tchk(f"{lab_} & 1 & {_bT3[('B32', m_, c_)]['test_macro_f1']:.3f}", [f"{_bT3[('L14', m_, c_)]['test_macro_f1']:.3f}"] + ([_bd(n_), _bcib(n_)] if n_ else []), cover_anchor=True)
tchk(f"T3 zero-shot (prompt set {_bZ['chosen_l14']}) & 1 & {_bZ['test_b32_zero_shot']['macro_f1']:.3f}", [f"{_bZ['test_l14_chosen']['macro_f1']:.3f}"], cover_anchor=True)

# 4h. F: duplicate-image check (results/ablation/X/f_duplicates_seed42.json)
_fd = json.load(open(RES / "X" / "f_duplicates_seed42.json"))
_fr = _fd["rescored"]
_fm = lambda sub_, k_: np.mean([v_[k_] for n_, v_ in _fr.items() if sub_ in n_])
assert max(abs(v_["after_macro_f1"] - v_["before_macro_f1"]) for v_ in _fr.values()) < 0.005
assert _fd["t3_excluded"] == _fr["T3 hybrid (probe label)"]["n_before"] - _fr["T3 hybrid (probe label)"]["n_after"] and _fd["t2_excluded"] == 2237 - _fr["T2 L4a seed 42"]["n_after"]
assert all(not e_["in_t3_index"] for e_ in _fd["demo_examples"]) and len(_fd["demo_examples"]) == 5 and max(e_["top1_image_cosine"] for e_ in _fd["demo_examples"]) < _fd["threshold"]
assert _bZ["test_l14_dev_calibrated"]["macro_f1"] < _bZ["test_b32_dev_calibrated"]["macro_f1"] and _bZ["test_l14_chosen"]["macro_f1"] > _bZ["test_b32_zero_shot"]["macro_f1"]   # ranking reverses
chk("Exact duplicates (md5 of every image file)", [f"{_fd['n_files']:,} image files, {_fd['n_distinct_md5']:,} distinct hashes", f"{_fd['exact_groups']} groups of byte-identical files ({_fd['exact_group_files']} files, largest group {_fd['largest_group']})",
                                                    f"{_fd['exact_groups_cross_tweet']} of the groups spanning different tweet ids ({_fd['exact_group_files_cross_tweet']} files)"])
chk("(a) Task 3:", [f"{_fd['t3_test_exact_in_index']} of {_fd['t3_n_test']} test images", f"({_fd['t3_index_rows']:,} rows", f"for {_fd['t3_exact_same_label_present']} of them"])
chk("(b) Task 2:", [f"{_fd['t2_test_exact_in_train_dev']} of {_fd['t2_n_test']:,} test images", f"({_fd['t2_ref_rows']:,} rows; {_fd['t2_exact_same_tweet']} share the tweet id", f"for {_fd['t2_exact_same_label_present']})"])
chk("Near-duplicates (CLIP B/32 cosine", [f"{_fd['t3_near_dup_queries']} of {_fd['t3_n_test']} ({_fd['t3_near_not_byte_identical']} not byte-identical)", f"{_fd['t2_near_dup_rows']} of {_fd['t2_n_test']:,} ({_fd['t2_near_not_byte_identical']} not byte-identical)"])
chk("Re-scoring with the flagged test rows excluded", [f"{_fd['t3_excluded']} Task 3 and {_fd['t2_excluded']} Task 2 rows"])
for k_, v_ in _fr.items():
    chk(f"| {k_} |", [f"{v_['n_before']} | {v_['n_after']}", f"{v_['before_macro_f1']:.4f} | {v_['after_macro_f1']:.4f}"])
chk("The largest change in any row", [f"{max(abs(v_['after_macro_f1'] - v_['before_macro_f1']) for v_ in _fr.values()):.4f}", f"L4a {_fm('L4a', 'before_macro_f1'):.4f} to {_fm('L4a', 'after_macro_f1'):.4f}",
                                      f"image_only {_fm('image_only', 'before_macro_f1'):.4f} to {_fm('image_only', 'after_macro_f1'):.4f}"])
for e_ in _fd["demo_examples"]:
    chk(f"| `{e_['image_id']}` | {e_["kind"]} (live example) | no |", [f"{e_['top1_fused_similarity']:.4f}", f"{e_['top1_image_cosine']:.4f}"])
for e_ in _fd["examples_t3"] + _fd["examples_t2"]:
    chk(f"* `{e_['test_image_id']}` ({e_['test_label']}", [f"cosine {e_['cosine']:.4f}", f"byte-identical {'yes' if e_['byte_identical'] else 'no'}"])
tchk("Hashing every image file finds", [f"finds {_fd['exact_groups']} groups", f"{_fd['exact_groups_cross_tweet']} of them spanning"])
tchk("Of the 529 Task 3 test images", [f"Of the {_fd['t3_n_test']} Task 3 test images, {_fd['t3_test_exact_in_index']} have", f"and {_fd['t3_near_dup_queries']} have an index image", f"cosine of at least {_fd['threshold']}",
                                       f"of the {_fd['t2_n_test']:,} Task 2 test images, {_fd['t2_test_exact_in_train_dev']} have", f"and {_fd['t2_near_dup_rows']} have a training image"])
_fh, _fk = _fr["T3 hybrid (probe label)"], _fr["T3 5-NN majority"]
tchk("Scoring the stored predictions without the flagged test rows",
     [f"({_fd['t3_excluded']} and {_fd['t2_excluded']})", "by more than 0.005", f"Task 3 hybrid {_fh['before_macro_f1']:.3f} to {_fh['after_macro_f1']:.3f}", f"5-NN {_fk['before_macro_f1']:.3f} to {_fk['after_macro_f1']:.3f}",
      f"L4a {_fm('L4a', 'before_macro_f1'):.3f} to {_fm('L4a', 'after_macro_f1'):.3f}", f"image-only {_fm('image_only', 'before_macro_f1'):.3f} to {_fm('image_only', 'after_macro_f1'):.3f}"])
tchk("Under the development-calibrated bias variant", [f"({_bZ['test_l14_dev_calibrated']['macro_f1']:.3f} for L/14 against {_bZ['test_b32_dev_calibrated']['macro_f1']:.3f} for B/32)"])

# 4i. G: class imbalance, extended metrics and the loss comparison (docs §9.9 and the .tex paragraph).
# Everything is recomputed from the saved logits (artifacts/ablation/imbalance, artifacts/ablation/rescore), the data loaders and the per-run JSONs in
# results/ablation/G, with the same bootstrap helpers (ablation/bootstrap_utils.py, fixed resample indices); stored summary JSONs are cross-checked.
import bootstrap_utils as _gbu
from extra_metrics import macro_f1_ci as _gmci
from sklearn.metrics import (average_precision_score as _gap, balanced_accuracy_score as _gba, f1_score as _gf1, matthews_corrcoef as _gmcc,
                             recall_score as _grec, roc_auc_score as _gauc)

_gD = RES / "G"; _gIMB = _ART / "ablation" / "imbalance"; _gRES = _ART / "ablation" / "rescore"
_gJ = lambda n_: json.load(open(_gD / f"{n_}.json"))
_GS, _GF = (42, 43, 44), ("b32", "l14")
_GARMS = ("none", "inv", "sqrt_inv", "eff_num", "focal1", "focal2", "focal2_inv", "sampler", "logit_adj", "none_posthoc")
_GNAME = {"none": "`none` (unweighted)", "inv": "`inv` (current)", "sqrt_inv": "`sqrt_inv`", "eff_num": "`eff_num`", "focal1": "`focal1`", "focal2": "`focal2`",
          "focal2_inv": "`focal2_inv`", "sampler": "`sampler`", "logit_adj": "`logit_adj`", "none_posthoc": "`none_posthoc`"}
_GBB = {"b32": "B/32", "l14": "L/14"}
_gH = {s_: _cc.load_split("humanitarian", s_, _DATA, _SPLIT, verbose=False)[0] for s_ in ("train", "dev", "test")}
_glab = sorted(_gH["train"].label.unique()); _gl2i = {l_: i_ for i_, l_ in enumerate(_glab)}; _gALL = list(range(len(_glab)))
_gy = {s_: np.array([_gl2i[l_] for l_ in _gH[s_].label]) for s_ in ("dev", "test")}
_gMISS, _gINJ, _gVEH, _gAFF = (_gl2i[x_] for x_ in ("Missing or found people", "Injured or dead people", "Vehicle damage", "Affected individuals"))
_gm = lambda a_, d_=3: f"{np.mean(a_):.{d_}f} ± {np.std(a_, ddof=1):.{d_}f}"
_gsg = lambda x_, d_=4: f"{(0.0 if abs(x_) < 0.5 * 10 ** -d_ else x_):+.{d_}f}"
_gci3 = lambda lo_, hi_: f"[{lo_:.3f}, {hi_:.3f}]"
_gdci = lambda r_: f"{_gsg(r_['diff'])} [{_gsg(r_['ci95_low'])}, {_gsg(r_['ci95_high'])}]"
def _gvd(r_):
    """'better'/'worse' only if the paired 95% CI excludes 0 by more than 0.002; a bound within 0.002 of zero is 'borderline'."""
    if r_["ci95_low"] > 0:
        return "borderline" if r_["ci95_low"] < 0.002 else "better"
    if r_["ci95_high"] < 0:
        return "borderline" if r_["ci95_high"] > -0.002 else "worse"
    return "not distinguishable"
_gnear = lambda a_, b_: abs(a_ - b_) < 1e-9

# -- recomputed per-run quantities from the saved logits; cross-checked against the per-run JSONs --
_gR = {}
for _f in _GF:
    for _a in _GARMS:
        for _s in _GS:
            _z = np.load(_gIMB / f"{_f}_{_a}_seed{_s}.npz"); _pd, _pt = _z["dev"].argmax(1), _z["test"].argmax(1)
            _rj = _gJ(f"g_t2_{_f}_{_a}_seed{_s}")
            _gR[(_f, _a, _s)] = dict(
                pd=_pd, pt=_pt, dev_f1=float(_gf1(_gy["dev"], _pd, labels=_gALL, average="macro", zero_division=0)), dev_bal=float(_gba(_gy["dev"], _pd)),
                test_f1=float(_gf1(_gy["test"], _pt, labels=_gALL, average="macro", zero_division=0)), test_bal=float(_gba(_gy["test"], _pt)),
                rec={c_: float(_grec(_gy["test"], _pt, labels=[c_], average="macro", zero_division=0)) for c_ in (_gMISS, _gINJ, _gVEH, _gAFF)},
                f1c={c_: float(_gf1(_gy["test"], _pt, labels=[c_], average="macro", zero_division=0)) for c_ in (_gMISS, _gAFF)})
            assert _gnear(_gR[(_f, _a, _s)]["test_f1"], _rj["test"]["macro_f1"]) and _gnear(_gR[(_f, _a, _s)]["dev_f1"], _rj["dev_macro_f1"])
            assert _gnear(_gR[(_f, _a, _s)]["test_bal"], _rj["test"]["balanced_acc"])
            assert (_pt == _gMISS).sum() == sum(1 for v_ in [_rj["pred"]] for p_ in v_ if p_ == _gMISS)
# validation claims: 'inv' reproduces the stored image-only runs exactly (B/32: Part B; L/14: the stored L/14 normalised runs)
for _s in _GS:
    _st = json.load(open(RES / "B" / f"t2_image_only_seed{_s}.json")); _r = _gR[("b32", "inv", _s)]
    assert _r["test_f1"] == _st["test_macro_f1"] and _r["dev_f1"] == _st["dev_macro_f1"]
    _st = json.load(open(RES / "X" / f"b_t2_imageonly_L14_norm_seed{_s}.json")); _r = _gR[("l14", "inv", _s)]
    assert _r["test_f1"] == _st["test_macro_f1"] and _r["dev_f1"] == _st["dev_macro_f1"]
_gmean = lambda f_, a_, k_: float(np.mean([_gR[(f_, a_, s_)][k_] for s_ in _GS]))
_gseeds = lambda f_, a_, k_: [_gR[(f_, a_, s_)][k_] for s_ in _GS]

# -- bootstraps (fixed resample indices, the same for every arm and seed) --
_gDI, _gTI = _gbu.resample_idx(len(_gy["dev"]), 2000), _gbu.resample_idx(len(_gy["test"]), 10000)
_gDV = {k_: _gbu.macro_f1_vec(_gy["dev"], v_["pd"], 8, _gDI) for k_, v_ in _gR.items()}
_gTV = {k_: _gbu.macro_f1_vec(_gy["test"], v_["pt"], 8, _gTI) for k_, v_ in _gR.items()}
_gvec = lambda V_, f_, a_: {s_: V_[(f_, a_, s_)] for s_ in _GS}
_gSUM = _gJ("g_imbalance_summary_seed42")["feats"]
_gX = {}
for _f in _GF:
    _o = sorted(_GARMS, key=lambda a_: -_gmean(_f, a_, "dev_f1")); _best = _o[0]; _tied = [_best]; _dd = {}
    for _a in _o[1:]:
        _dd[_a] = _gbu.paired_diff_ci(_gvec(_gDV, _f, _best), _gvec(_gDV, _f, _a), point_diff=_gmean(_f, _best, "dev_f1") - _gmean(_f, _a, "dev_f1"))
        if not (_dd[_a]["ci95_low"] > 0 or _dd[_a]["ci95_high"] < 0):
            _tied.append(_a)
    _pick = max(_tied, key=lambda a_: _gmean(_f, a_, "dev_bal"))
    _tv = {}
    for _a in _GARMS:
        _tv[_a] = dict(ci=_gbu.mean_ci(_gvec(_gTV, _f, _a), point=_gmean(_f, _a, "test_f1")),
                       vs_inv=_gbu.paired_diff_ci(_gvec(_gTV, _f, _a), _gvec(_gTV, _f, "inv"), point_diff=_gmean(_f, _a, "test_f1") - _gmean(_f, "inv", "test_f1")),
                       vs_none=_gbu.paired_diff_ci(_gvec(_gTV, _f, _a), _gvec(_gTV, _f, "none"), point_diff=_gmean(_f, _a, "test_f1") - _gmean(_f, "none", "test_f1")))
    _gX[_f] = dict(order=_o, best=_best, tied=_tied, dd=_dd, pick=_pick, tv=_tv)
    # stored summary == recomputed
    assert _gSUM[_f]["order"] == _o and _gSUM[_f]["tied"] == _tied and _gSUM[_f]["selected"] == _pick and _gSUM[_f]["dev_best"] == _best
    for _a in _GARMS:
        _x = _gSUM[_f]["arms"][_a]
        assert _gnear(_x["dev_f1"], _gmean(_f, _a, "dev_f1")) and _gnear(_x["test_f1"], _gmean(_f, _a, "test_f1")) and _gnear(_x["test_bal_acc"], _gmean(_f, _a, "test_bal"))
        for _k, _r in (("test_ci", _tv[_a]["ci"]), ("test_vs_inv", _tv[_a]["vs_inv"]), ("test_vs_none", _tv[_a]["vs_none"])):
            assert all(_gnear(_x[_k][q_], _r[q_]) for q_ in _r if q_ in ("ci95_low", "ci95_high"))
assert _gSUM["b32"]["arms"]["inv"]["test_f1_seed_std"] == float(np.std(_gseeds("b32", "inv", "test_f1"), ddof=1)) or _gnear(_gSUM["b32"]["arms"]["inv"]["test_f1_seed_std"], float(np.std(_gseeds("b32", "inv", "test_f1"), ddof=1)))
_gSEL = {"b32": _gX["b32"]["pick"], "l14": _gX["l14"]["pick"]}
assert _gSEL == {"b32": "inv", "l14": "focal2_inv"} and _gSEL == _gJ("g_imbalance_summary_seed42")["selected"] and _gJ("g_imbalance_summary_seed42")["adopt"] is None

# -- class counts, from the loaders the models use --
_gCT = {t_: {s_: _cc.load_split(t_, s_, _DATA, _SPLIT, verbose=False)[0] for s_ in ("train", "dev", "test")} for t_ in ("informative", "humanitarian", "damage")}
_gTL = {"informative": "T1", "humanitarian": "T2", "damage": "T3"}
chk("Largest-to-smallest class ratio, train:", [
    "train: " + ", ".join(f"{_gTL[t_]} {max(_gCT[t_]['train'].label.value_counts()) / min(_gCT[t_]['train'].label.value_counts()):.1f}:1" for t_ in _gCT) + "; test: " +
    ", ".join(f"{_gTL[t_]} {max(_gCT[t_]['test'].label.value_counts()) / min(_gCT[t_]['test'].label.value_counts()):.1f}:1" for t_ in _gCT)])
for _t in _gCT:
    _tr = _gCT[_t]["train"].label.value_counts()
    for _c in _tr.index:
        chk(f"| {_gTL[_t]} | {_c} |", [f"{int(_tr[_c]):,} | {int((_gCT[_t]['dev'].label == _c).sum()):,} | {int((_gCT[_t]['test'].label == _c).sum()):,} |"])
_gmiss = lambda f_: sum(int((_gR[(f_, a_, s_)]["pt"] == _gMISS).sum()) for a_ in _GARMS for s_ in _GS)
assert all(_gR[(f_, a_, s_)]["f1c"][_gMISS] == 0.0 and _gR[(f_, a_, s_)]["rec"][_gMISS] == 0.0 for f_ in _GF for a_ in _GARMS for s_ in _GS)
_gmiss_ct = {s_: int((_gy[s_] == _gMISS).sum()) for s_ in ("dev", "test")}; _gmiss_ct["train"] = int((_gH["train"].label == "Missing or found people").sum())
chk("Conventions. Macro-F1 is always computed", [f"({_gmiss_ct['train']} train / {_gmiss_ct['dev']} dev / {_gmiss_ct['test']} test rows)",
                                                f"predicts it {_gmiss('b32')} times on B/32 and {_gmiss('l14')} times on L/14", f"7/8 = {7 / 8:.3f}", "(10,000 resamples;",
                                                "excludes 0 by more than 0.002; a bound within 0.002 of zero is called \"borderline\" (weak evidence)",
                                                "The same rule applies to every comparison, including the recall of the small injured-or-dead class (8 dev and 10 test rows; per-run recall moves in steps of 0.125 on dev and 0.1 on test)",
                                                "a CI that excludes 0 is called \"better\" or \"worse\", and the word \"not resolvable\" is reserved for a CI that includes 0 on that class",
                                                "which is not the same as no effect. Every recall comparison states its row count n.",
                                                "Pairwise comparisons were not corrected for multiple testing (about 10 arms on 2 backbones, several metrics), so isolated borderline exclusions should be read as weak evidence"])
chk("Added after the presentation, in answer to three questions", ["checks every number in this section against the saved results", "not that the analysis design is right"])

# -- extended metrics of the existing T2 models, recomputed from the saved rescore logits --
_gCFG = (("L4a", "T2 full model (L4a)", "A"), ("L4", "T2 v2 (L4)", "A"), ("image_only", "T2 image only", "B"), ("concat", "T2 concat, no attention", "B"),
         ("no_t2i", "T2 no text→image attention", "B"), ("text_only", "T2 text only", "B"))
_gMT = {}
for _cfg, _lab, _part in _gCFG:
    _mm = {k_: [] for k_ in ("acc", "f1", "bal", "mcc", "roc", "pr")}
    for _s in _GS:
        _z = np.load(_gRES / f"{_cfg}_seed{_s}.npy").astype(float); _p = _z.argmax(1); _e = np.exp(_z - _z.max(1, keepdims=True)); _e /= _e.sum(1, keepdims=True)
        _mm["acc"].append(float((_p == _gy["test"]).mean())); _mm["f1"].append(float(_gf1(_gy["test"], _p, labels=_gALL, average="macro", zero_division=0)))
        _mm["bal"].append(float(_gba(_gy["test"], _p))); _mm["mcc"].append(float(_gmcc(_gy["test"], _p)))
        _mm["roc"].append(float(np.mean([_gauc(_gy["test"] == c_, _e[:, c_]) for c_ in _gALL]))); _mm["pr"].append(float(np.mean([_gap(_gy["test"] == c_, _e[:, c_]) for c_ in _gALL])))
        _sj = _gJ(f"g_rescore_{_cfg}_seed{_s}"); _st = json.load(open(RES / _part / f"t2_{_cfg}_seed{_s}.json"))
        assert _sj["max_abs_diff_vs_stored"] == 0.0 and _gnear(_mm["acc"][-1], _st["test_acc"]) and _gnear(_mm["f1"][-1], _st["test_macro_f1"])
        assert _gnear(_sj["test"]["mcc"], _mm["mcc"][-1]) and _gnear(_sj["test"]["macro_pr_auc"], _mm["pr"][-1]) and _gnear(_sj["test"]["macro_roc_auc"], _mm["roc"][-1])
    _ci = _gmci(_gy["test"], np.load(_gRES / f"{_cfg}_seed42.npy").argmax(1), 8); _mm["ci"] = _ci; _gMT[_cfg] = _mm
    chk(f"| {_lab} | {_gm(_mm['acc'])} | {_gm(_mm['f1'])} | {_gm(_mm['bal'])} |", [_gm(_mm["mcc"]), _gm(_mm["roc"]), _gm(_mm["pr"]), f"{_gci3(_ci['ci95_low'], _ci['ci95_high'])} |"])
assert max(_gCFG, key=lambda c_: np.mean(_gMT[c_[0]]["bal"]))[0] == "image_only"
chk("The macro-F1 interval of the full model at seed 42", [_gci3(_gMT["L4a"]["ci"]["ci95_low"], _gMT["L4a"]["ci"]["ci95_high"]), f"({_gm(_gMT['image_only']['bal'])}, against {_gm(_gMT['L4a']['bal'])} for the full model)",
                                                           f"({_gm(_gMT['image_only']['f1'])}) is within {abs(np.mean(_gMT['image_only']['f1']) - np.mean(_gMT['L4a']['f1'])):.3f} of the full model's ({_gm(_gMT['L4a']['f1'])})"])

# -- Tasks 1 and 3 (stored deterministic CPU runs) --
_gT1, _gT3, _gKN = _gJ("g_t1_balance_seed42"), _gJ("g_t3_balance_seed42"), _gJ("g_t3_knn_vote_boot_seed42")
assert _gnear(_gKN["votes"]["majority"]["macro_f1"], _gT3["knn_fused_k5"]["majority"]) and _gnear(_gKN["votes"]["class_balanced"]["macro_f1"], b3("fused", 5, "class_balanced", "test_macro_f1"))
assert _gnear(_gKN["votes"]["majority"]["macro_f1"], b3("fused", 5, "majority", "test_macro_f1"))
for _lab, _arms, _Cs in [("T1 text (TF-IDF)", _gT1["arms"], None)] + [(f"T3 {e_} probe", _gT3["probes"][e_]["arms"], _gT3["probes"][e_]["C"]) for e_ in ("fused", "image", "text")]:
    for _a in ("balanced", "none"):
        _m = _arms[_a]; _c = _m["macro_f1_ci"]
        chk(f"| {_lab} | {_a} | {'1' if _Cs is None else format(_Cs[_a], 'g')} | {_m['acc']:.3f} | {_m['macro_f1']:.3f} | {_gci3(_c['ci95_low'], _c['ci95_high'])} |",
            [f"{_m['balanced_acc']:.3f} | {_m['mcc']:.3f} | {_m['macro_pr_auc']:.3f} | {min(v_['recall'] for v_ in _m['per_class'].values()):.3f} |"])
_gd1 = _gT1["balanced_minus_none"]
chk("Task 1: balanced minus unweighted macro-F1 is", [f"is {_gdci(_gd1)} (paired, 10,000 resamples): {_gvd(_gd1)}"])
for _e in ("fused", "image", "text"):
    _d = _gT3["probes"][_e]["balanced_minus_none"]
    chk(f"Task 3, {_e} probe: balanced minus unweighted", [f"is {_gdci(_d)}: {_gvd(_d)}"])
_gk = _gKN["class_balanced_minus_majority"]
chk("Task 3 5-NN on the fused embedding, class-balanced against majority", [f"(macro-F1 {_gKN['votes']['class_balanced']['macro_f1']:.4f} against {_gKN['votes']['majority']['macro_f1']:.4f}; the two votes differ on {_gKN['n_predictions_differing']} of {_gKN['n_test']} test queries)",
                                                                           f"difference {_gdci(_gk)} (paired, 10,000 resamples over the {_gKN['n_test']} queries): {_gvd(_gk)}"])
chk("Task 2 imbalance treatments (image-only head, the recipe", ["seeds 42, 43, 44", "β = 0.999", "τ = 1", "(maximum difference 0 at all three seeds)"])

# -- the arms tables --
for _f in _GF:
    _X = _gX[_f]
    for _a in _X["order"]:
        _dv, _db = _gmean(_f, _a, "dev_f1"), _gmean(_f, _a, "dev_bal")
        if _a == _X["best"]:
            chk(f"| {_GNAME[_a]} | {_dv:.3f} | {_db:.3f} | best arm |", ["| yes |"])
        else:
            _r = _X["dd"][_a]
            chk(f"| {_GNAME[_a]} | {_dv:.3f} | {_db:.3f} | {_gsg(_r['diff'])} |", [f"[{_gsg(_r['ci95_low'])}, {_gsg(_r['ci95_high'])}] | {'yes' if _a in _X['tied'] else 'no'} |"])
        _t = _X["tv"][_a]
        _vi = "--" if _a == "inv" else f"{_gdci(_t['vs_inv'])} {_gvd(_t['vs_inv'])}"; _vn = "--" if _a == "none" else f"{_gdci(_t['vs_none'])} {_gvd(_t['vs_none'])}"
        chk(f"| {_GNAME[_a]} | {_gmean(_f, _a, 'test_f1'):.3f} ± {np.std(_gseeds(_f, _a, 'test_f1'), ddof=1):.3f} | {_gci3(_t['ci']['ci95_low'], _t['ci']['ci95_high'])} | {_gmean(_f, _a, 'test_bal'):.3f} |",
            [f"{np.mean([_gR[(_f, _a, s_)]['rec'][_gINJ] for s_ in _GS]):.3f} | {np.mean([_gR[(_f, _a, s_)]['rec'][_gVEH] for s_ in _GS]):.3f} | {_vi} | {_vn} |"])
    _tv = _X["tv"]
    chk(f"Dev: the best arm is `{_X['best']}` ({_gmean(_f, _X['best'], 'dev_f1'):.3f})",
        [f"{len(_X['tied'])} of the {len(_GARMS)} arms (the best and {len(_X['tied']) - 1} others) are tied with it", f"selects `{_X['pick']}`", f"`inv` ranks {_X['order'].index('inv') + 1} of {len(_GARMS)} on dev macro-F1"])
    assert max(_GARMS, key=lambda a_: _gmean(_f, a_, "test_f1")) == "eff_num"
    chk(f"Test: the highest point estimate is `eff_num` ({_gmean(_f, 'eff_num', 'test_f1'):.3f})",
        [f"(dev rank {_X['order'].index('eff_num') + 1} of {len(_GARMS)})", f"against `none` it is {_gvd(_tv['eff_num']['vs_none'])} ({_gdci(_tv['eff_num']['vs_none'])})",
         f"against `inv` the paired interval excludes 0 only marginally ({_gdci(_tv['eff_num']['vs_inv'])})", f"(lower bound {_gsg(_tv['eff_num']['vs_inv']['ci95_low'])})",
         "not robust and was not corrected for multiple comparisons across about 10 arms on 2 backbones", "we do not treat it as evidence that `eff_num` outperforms `inv`"])
    assert 0 < _tv["eff_num"]["vs_inv"]["ci95_low"] < 0.002 and _gvd(_tv["eff_num"]["vs_inv"]) == "borderline" and _gvd(_tv["eff_num"]["vs_none"]) == "not distinguishable"
    if _f == "b32":
        _lb = _X["dd"]["inv"]["ci95_low"]; _alt = max((a_ for a_ in _X["tied"] if a_ != "inv"), key=lambda a_: _gmean(_f, a_, "dev_bal")); assert -0.002 < _lb < 0 and _alt == "eff_num"
        chk("`inv` is in the tied set by a narrow margin", [f"the lower bound of the best-minus-`inv` interval is {_gsg(_lb)}", f"without it the rule would select `{_alt}` (dev balanced accuracy {_gmean(_f, _alt, 'dev_bal'):.3f})"])
    else:
        _lb = _X["dd"]["none"]["ci95_low"]; assert 0 < _lb < 0.002 and "none" not in _X["tied"] and _gmean(_f, "none", "dev_bal") < _gmean(_f, _X["pick"], "dev_bal")
        chk("`none` is excluded from the tied set only by a borderline interval", [f"(lower bound {_gsg(_lb)})", f"its dev balanced accuracy ({_gmean(_f, 'none', 'dev_bal'):.3f}) is below that of `{_X['pick']}` ({_gmean(_f, _X['pick'], 'dev_bal'):.3f})"])
assert {f_: sorted(a_ for a_ in _GARMS if a_ != "none" and _gvd(_gX[f_]["tv"][a_]["vs_none"]) != "not distinguishable") for f_ in _GF} == {"b32": ["logit_adj"], "l14": ["focal2_inv", "logit_adj"]}
assert all(_gvd(_gX[f_]["tv"][a_]["vs_inv"]) == "worse" for f_ in _GF for a_ in ("sampler", "focal2_inv", "logit_adj"))
assert all(_gvd(_gX[f_]["tv"][a_]["vs_none"]) == "worse" for f_ in _GF for a_ in ("logit_adj",))
chk("Outcome of the pre-set rule.", [f"B/32 selects `{_gSEL['b32']}`; L/14 selects `{_gSEL['l14']}`", "did not adopt a new loss",
                                     f"dev has {int((_gy['dev'] == _gINJ).sum())} injured-or-dead and {_gmiss_ct['dev']} missing-or-found rows",
                                     f"{len(_gX['b32']['tied'])} of the {len(_GARMS)} arms (the best arm and {len(_gX['b32']['tied']) - 1} others) are tied with the best"])
assert len(_gX["b32"]["tied"]) == len(_gX["l14"]["tied"])

# -- balanced accuracy and minority recall, none / inv / sqrt_inv --
_gBR = _gJ("g_balacc_recall_boot_seed42"); _ginj = []
_gsup = {s_: {n_: int((_gy[s_] == c_).sum()) for n_, c_ in (("inj", _gINJ), ("veh", _gVEH), ("mis", _gMISS))} for s_ in ("dev", "test")}
chk("Balanced accuracy and minority recall, `none` / `inv` / `sqrt_inv`.", [f"{_gsup['dev']['inj']} injured or dead, {_gsup['dev']['veh']} vehicle damage, {_gsup['dev']['mis']} missing or found",
                                                                           f"{_gsup['test']['inj']}, {_gsup['test']['veh']}, {_gsup['test']['mis']}", "(selection-eligible; 2,000 resamples)" if False else "2,000 resamples", "10,000 resamples"])
_gMN = {"balanced_acc": "balanced accuracy", "recall_injured": "recall, injured or dead", "recall_vehicle": "recall, vehicle damage"}
for _f in _GF:
    for _m, _c in (("balanced_acc", None), ("recall_injured", _gINJ), ("recall_vehicle", _gVEH)):
        _vv = {}
        for _split, _idx, _y in (("dev", _gDI, _gy["dev"]), ("test", _gTI, _gy["test"])):
            _P = {a_: {s_: _gR[(_f, a_, s_)]["pd" if _split == "dev" else "pt"] for s_ in _GS} for a_ in ("none", "inv", "sqrt_inv")}
            if _c is None:
                _pt = {a_: [float(_gba(_y, _P[a_][s_])) for s_ in _GS] for a_ in _P}
                _vec = {a_: {s_: _gbu.balanced_acc_vec(_y, _P[a_][s_], 8, _idx) for s_ in _GS} for a_ in _P}
            else:
                _pt = {a_: [float(_grec(_y, _P[a_][s_], labels=[_c], average="macro", zero_division=0)) for s_ in _GS] for a_ in _P}
                _rv = {a_: {s_: _gbu.recall_vec(_y, _P[a_][s_], 8, _idx) for s_ in _GS} for a_ in _P}
                _keep = _rv["none"][42][1][:, _c] > 0; assert _keep.all()
                _vec = {a_: {s_: _rv[a_][s_][0][:, _c][_keep] for s_ in _GS} for a_ in _P}
            for _pr in ("inv-none", "sqrt_inv-none", "inv-sqrt_inv"):
                _a1, _b1 = _pr.split("-")
                _r = _gbu.paired_diff_ci(_vec[_a1], _vec[_b1], point_diff=float(np.mean([_pt[_a1][i_] - _pt[_b1][i_] for i_ in range(3)])))
                _vv[(_split, _pr)] = _r
                _sr = _gBR["feats"][_f][_split]["metrics"][_m]["pairs"][_pr]
                assert all(_gnear(_sr[q_], _r[q_]) for q_ in ("diff", "ci95_low", "ci95_high"))
        for _pr in ("inv-none", "sqrt_inv-none", "inv-sqrt_inv"):
            _a1, _b1 = _pr.split("-")
            _vw = (lambda r_: "not resolvable" if _gvd(r_) == "not distinguishable" else _gvd(r_)) if _m == "recall_injured" else _gvd
            _nn = (lambda sp_: "") if _m == "balanced_acc" else (lambda sp_: f" (n={_gsup[sp_]['inj' if _m == 'recall_injured' else 'veh']})")
            chk(f"| {_GBB[_f]} | {_gMN[_m]} | `{_a1}` − `{_b1}` |", [f"{_gdci(_vv[('dev', _pr)])} {_vw(_vv[('dev', _pr)])}{_nn('dev')} | {_gdci(_vv[('test', _pr)])} {_vw(_vv[('test', _pr)])}{_nn('test')} |"])
            if _m == "recall_injured":
                _ginj.extend([_vv[("dev", _pr)], _vv[("test", _pr)]])
            if _m == "balanced_acc":
                assert _vv[("dev", _pr)]["ci95_low"] > 0 and _vv[("test", _pr)]["ci95_low"] > 0
            if _m == "recall_vehicle" and _pr != "inv-sqrt_inv":
                assert _vv[("dev", _pr)]["ci95_low"] > 0 and _vv[("test", _pr)]["ci95_low"] > 0
            if _m == "recall_injured":
                assert _gvd(_vv[("dev", _pr)]) == "not distinguishable" == _gvd(_vv[("test", _pr)])
        if _m == "recall_vehicle":
            assert not (_vv[("dev", "inv-sqrt_inv")]["ci95_low"] > 0) if _f == "l14" else _vv[("dev", "inv-sqrt_inv")]["ci95_low"] > 0
            assert _vv[("test", "inv-sqrt_inv")]["ci95_low"] > 0
assert len(_ginj) == 12 and sum(1 for r_ in _ginj if r_["ci95_low"] == 0.0) == 11 and all(not (r_["ci95_low"] > 0 or r_["ci95_high"] < 0) for r_ in _ginj)
chk("Against `none`, both `inv` and `sqrt_inv` raise balanced accuracy", [f"all {len(_ginj)} comparisons (dev and test, both backbones, three pairs) have a CI that includes 0 (eleven of them with a lower bound of exactly 0)",
                                                                           f"\"not resolvable\" at this sample size ({_gsup['dev']['inj']} dev / {_gsup['test']['inj']} test rows; "
                                                                           f"per-run recall moves in steps of {1 / _gsup['dev']['inj']:.3f} on dev and {1 / _gsup['test']['inj']:.1f} on test)", "neither evidence of an effect nor of no effect"])
for _f in _GF:
    _rn, _ri = (_gmean(_f, a_, "test_bal") for a_ in ("none", "inv"))
    chk(f"| {_GBB[_f]} | {np.mean([_gR[(_f, 'none', s_)]['rec'][_gAFF] for s_ in _GS]):.3f} | {np.mean([_gR[(_f, 'inv', s_)]['rec'][_gAFF] for s_ in _GS]):.3f} | "
        f"{np.mean([_gR[(_f, 'none', s_)]['f1c'][_gAFF] for s_ in _GS]):.3f} | {np.mean([_gR[(_f, 'inv', s_)]['f1c'][_gAFF] for s_ in _GS]):.3f} |", [])
_gaf = {f_: {a_: (np.mean([_gR[(f_, a_, s_)]["rec"][_gAFF] for s_ in _GS]), np.mean([_gR[(f_, a_, s_)]["f1c"][_gAFF] for s_ in _GS])) for a_ in ("none", "inv")} for f_ in _GF}
assert all(_gaf[f_]["inv"][0] > _gaf[f_]["none"][0] for f_ in _GF) and _gaf["b32"]["inv"][1] < _gaf["b32"]["none"][1]
chk("Weighting raises the recall of this class on both backbones", [f"({_gaf['b32']['none'][1]:.3f} to {_gaf['b32']['inv'][1]:.3f})", f"changes by {_gaf['l14']['inv'][1] - _gaf['l14']['none'][1]:+.3f} on L/14"])

# -- POST-HOC sensitivity, recomputed from the recomputed tied sets --
_gSIMP = ("none", "none_posthoc", "focal1", "focal2", "sqrt_inv", "eff_num", "inv", "focal2_inv", "sampler", "logit_adj")
_gPH = {f_: dict(a=_gX[f_]["best"], b=_gX[f_]["pick"], c=min(_gX[f_]["tied"], key=_gSIMP.index)) for f_ in _GF}
_gPHs = _gJ("g_POSTHOC_selection_sensitivity_seed42")
assert _gPHs["POSTHOC"] is True and all(_gPHs["feats"][f_][k_] == _gPH[f_][{"a_best_dev_macro_f1": "a", "b_original_rule": "b", "c_simplest_among_tied": "c"}[k_]] for f_ in _GF for k_ in ("a_best_dev_macro_f1", "b_original_rule", "c_simplest_among_tied"))
for _key, _lab in (("a", "(a) best mean dev macro-F1"), ("b", "(b) original rule: highest dev balanced accuracy among the tied"), ("c", "(c) simplest arm among the tied (closest to unweighted)")):
    chk(f"| {_lab} |", [f"`{_gPH['b32'][_key]}` | `{_gPH['l14'][_key]}` | {'yes' if _gPH['b32'][_key] == _gPH['l14'][_key] else 'no'} |"])
assert all(_gPH["b32"][k_] != _gPH["l14"][k_] for k_ in "abc")
_gsd = {f_: (_gmean(f_, "sqrt_inv", "dev_f1"), _gmean(f_, _gX[f_]["best"], "dev_f1") - _gmean(f_, "sqrt_inv", "dev_f1"), "sqrt_inv" in _gX[f_]["tied"]) for f_ in _GF}
chk("| (d) `sqrt_inv` as the midpoint reference", [f"(dev macro-F1 {_gsd['b32'][0]:.3f} / {_gsd['l14'][0]:.3f}; gap to the best arm {_gsd['b32'][1]:.3f} / {_gsd['l14'][1]:.3f})",
                                                   f"tied: {'yes' if _gsd['b32'][2] else 'no'} | tied: {'yes' if _gsd['l14'][2] else 'no'}"])
chk("No selection variant agrees across the two backbones", ["none < none_posthoc < focal1 < focal2 < sqrt_inv < eff_num < inv < focal2_inv < sampler < logit_adj", f"on L/14 it picks `{_gPH['l14']['c']}`"])

# -- pre-registered fused check: `none` against `inv` on L4a-L14, recomputed from the stored predictions --
_gFC = _gJ("g_fused_L4a-L14_none_vs_inv_seed42")
_gval, _gsto = _gJ("t2_L4a_validate_seed42"), json.load(open(RES / "A" / "t2_L4a_seed42.json"))
assert all(_gval[k_] == _gsto[k_] for k_ in ("dev_macro_f1", "test_acc", "test_macro_f1", "test_weighted_f1", "per_class_f1", "confusion_matrix", "best_epoch", "n_params"))
_gFP = {}
for _arm in ("inv", "none"):
    for _s in _GS:
        _rj = _gJ(f"t2_L4a-L14_{_arm}_seed{_s}")
        assert _rj["flags"]["feat"] == "l14" and _rj["flags"]["loss"] == _arm and _rj["flags"]["bert"] == "pretrained" and _rj["part"] == "G" and _rj["seed"] == _s and _rj["n_params"] > 67_000_000
        _pdv, _ptt = (np.array(_gFC["arms"][_arm][k_][str(_s)]) for k_ in ("pred_dev", "pred_test"))
        assert len(_pdv) == len(_gy["dev"]) and len(_ptt) == len(_gy["test"])
        # the stored predictions reproduce the training run's own test confusion matrix and its dev and test macro-F1
        assert np.bincount(_gy["test"] * 8 + _ptt, minlength=64).reshape(8, 8).tolist() == _rj["confusion_matrix"]
        assert _gnear(_gf1(_gy["dev"], _pdv, labels=_gALL, average="macro", zero_division=0), _rj["dev_macro_f1"]) and _gnear(_gf1(_gy["test"], _ptt, labels=_gALL, average="macro", zero_division=0), _rj["test_macro_f1"])
        _gFP[(_arm, _s)] = dict(dev=_pdv, test=_ptt)
_gFM = {}
for _arm in ("inv", "none"):
    for _sp in ("dev", "test"):
        _yy = _gy[_sp]; _P = lambda s_: _gFP[(_arm, s_)][_sp]
        _gFM[(_arm, _sp)] = dict(f1=[float(_gf1(_yy, _P(s_), labels=_gALL, average="macro", zero_division=0)) for s_ in _GS], bal=[float(_gba(_yy, _P(s_))) for s_ in _GS],
                                 inj=[float(_grec(_yy, _P(s_), labels=[_gINJ], average="macro", zero_division=0)) for s_ in _GS], veh=[float(_grec(_yy, _P(s_), labels=[_gVEH], average="macro", zero_division=0)) for s_ in _GS],
                                 mis=[float(_grec(_yy, _P(s_), labels=[_gMISS], average="macro", zero_division=0)) for s_ in _GS])
        assert max(_gFM[(_arm, _sp)]["mis"]) == 0.0
_gFD = {}
for _sp, _idx in (("dev", _gDI), ("test", _gTI)):
    _yy = _gy[_sp]; _vec = {}
    for _arm in ("inv", "none"):
        _rv = {s_: _gbu.recall_vec(_yy, _gFP[(_arm, s_)][_sp], 8, _idx)[0] for s_ in _GS}
        _vec[_arm] = dict(f1={s_: _gbu.macro_f1_vec(_yy, _gFP[(_arm, s_)][_sp], 8, _idx) for s_ in _GS}, bal={s_: _gbu.balanced_acc_vec(_yy, _gFP[(_arm, s_)][_sp], 8, _idx) for s_ in _GS},
                          inj={s_: _rv[s_][:, _gINJ] for s_ in _GS}, veh={s_: _rv[s_][:, _gVEH] for s_ in _GS})
    for _m in ("f1", "bal", "inj", "veh"):
        _r = _gbu.paired_diff_ci(_vec["none"][_m], _vec["inv"][_m], point_diff=float(np.mean(_gFM[("none", _sp)][_m]) - np.mean(_gFM[("inv", _sp)][_m])))
        _v = _gvd(_r)
        if _m == "inj" and _v == "not distinguishable":
            _v = "not resolvable"
        _gFD[(_sp, _m)] = (_r, _v)
        _st = _gFC["pairs"][f"{_sp}:{_m}"]
        assert all(_gnear(_st[q_], _r[q_]) for q_ in ("diff", "ci95_low", "ci95_high")) and _st["verdict"] == _v
        if _m in ("inj", "veh"):
            assert _st["n_rows"] == int((_yy == (_gINJ if _m == "inj" else _gVEH)).sum())
assert [_gFD[(s_, m_)][1] for s_ in ("dev", "test") for m_ in ("f1", "bal", "inj", "veh")] == ["not distinguishable", "worse", "worse", "not distinguishable", "not distinguishable", "worse", "worse", "worse"]
_gfn = lambda sp_, m_: f" (n={_gsup[sp_]['inj' if m_ == 'inj' else 'veh']})" if m_ in ("inj", "veh") else ""
chk("Pre-registered fused check: `none` against `inv` on L4a-L14", ["seeds 42, 43, 44", "no other arm was run", "reproduces the stored L4a seed-42 result exactly",
                                                                    "The first launch of the six runs failed on argument parsing (`--tag -L14_inv`) before producing any result; the relaunch used `--tag=...`; no result was seen or discarded"])
for _lab, _m in (("macro-F1", "f1"), ("balanced accuracy", "bal"), ("recall, injured or dead", "inj"), ("recall, vehicle damage", "veh")):
    _cells = {sp_: (_gm(_gFM[("inv", sp_)][_m]), _gm(_gFM[("none", sp_)][_m]), f"{_gdci(_gFD[(sp_, _m)][0])} {_gFD[(sp_, _m)][1]}{_gfn(sp_, _m)}") for sp_ in ("dev", "test")}
    chk(f"| {_lab} | {_cells['dev'][0]} | {_cells['dev'][1]} |", [f"{_cells['dev'][2]} | {_cells['test'][0]} | {_cells['test'][1]} | {_cells['test'][2]} |"])
_gdd = {m_: _gFD[("dev", m_)][0] for m_ in ("f1", "bal", "inj", "veh")}; _gtt = {m_: _gFD[("test", m_)][0] for m_ in ("f1", "bal", "inj", "veh")}
_gsdt = [float(np.std(_gFM[(a_, "test")]["f1"], ddof=1)) for a_ in ("inv", "none")]
chk("Dev (the only version eligible for selection) and test (reporting only", ["shown side by side and neither is preferred", "This is 8 comparisons (4 metrics on 2 splits), none corrected for multiplicity"])
chk("Removing weighting lowers balanced accuracy (dev", [f"(dev {_gsg(_gdd['bal']['diff'])}, test {_gsg(_gtt['bal']['diff'])}; both intervals exclude 0)", f"lowers vehicle recall on test ({_gsg(_gtt['veh']['diff'])})",
                                                         f"(dev {_gsg(_gdd['f1']['diff'])}, test {_gsg(_gtt['f1']['diff'])}; both intervals include 0; the test seed std is {min(_gsdt):.3f} to {max(_gsdt):.3f}, the same size as the {abs(_gtt['f1']['diff']):.3f} gap)",
                                                         "Inverse-frequency weighting is retained as the headline loss, as pre-registered."])
chk("Vehicle recall is not distinguishable on dev", [f"(interval upper bound {_gsg(_gdd['veh']['ci95_high'])})", "but worse on test", "Only dev counts for selection", "with 8 uncorrected comparisons the test result alone is weak evidence"])
chk("Injured-or-dead recall is lower without weighting", [f"(\"worse\", n={_gsup['dev']['inj']} and n={_gsup['test']['inj']})"])
chk("The sign of the macro-F1 difference differs from the image-only study", [f"(`inv` minus `none` {_gsg(_gX['b32']['tv']['inv']['vs_none']['diff'])} on B/32 and {_gsg(_gX['l14']['tv']['inv']['vs_none']['diff'])} on L/14)",
                                                                              f"here `none` scores below `inv` ({_gsg(_gtt['f1']['diff'])})", "and the sign is read as noise"])
assert _gX["b32"]["tv"]["inv"]["vs_none"]["diff"] < 0 and _gX["l14"]["tv"]["inv"]["vs_none"]["diff"] < 0 and _gtt["f1"]["diff"] < 0

# -- bootstrap audit --
_gBC = _gJ("g_bootstrap_check_seed42")
_gzt, _gzd = int((np.sum(_gy["test"][_gTI] == _gMISS, axis=1) == 0).sum()), int((np.sum(_gy["dev"][_gDI] == _gMISS, axis=1) == 0).sum())
assert _gzt == _gBC["zero_missing_resamples"]["test"] and _gzd == _gBC["zero_missing_resamples"]["dev"]
_ggap = {f_: [float(np.mean([_gTV[(f_, a_, s_)] for s_ in _GS], axis=0).mean() - _gmean(f_, a_, "test_f1")) for a_ in _GARMS] for f_ in _GF}
_gchg = max(max(abs(v_[f"vs_{r_}"]["before"][k_] - v_[f"vs_{r_}"]["after"][k_]) for k_ in ("ci95_low", "ci95_high"))
            for f_ in _GF for a_, v_ in _gBC["feats"][f_]["arms"].items() for r_ in ("inv", "none") if f"vs_{r_}" in v_)
assert _gchg == 0.0 and not _gBC["verdict_changes_after_recompute"] and all(_gBC["feats"][f_]["sklearn_crosscheck_max_abs_diff"] == 0.0 for f_ in _GF)
assert all(f"{np.mean([v_['wrong_minus_correct'] for v_ in _gBC['feats'][f_]['arms'].values()]):.3f}" == "0.004" for f_ in _GF)
chk("Bootstrap audit.", [f"{_gzt} of 10,000 test resamples ({_gzt / 100:.2f}%; expected {(1 - 3 / 2237) ** 2237 * 100:.2f}%)", f"{_gzd} of 2,000 dev resamples ({_gzd / 20:.2f}%)",
                         "(maximum difference 0.0)", "raised the bootstrap mean by about 0.004", "(maximum change 0) and 0 verdicts",
                         f"lies {abs(max(_ggap['b32'])):.4f} to {abs(min(_ggap['b32'])):.4f} below the point estimate on B/32 and {abs(max(_ggap['l14'])):.4f} to {abs(min(_ggap['l14'])):.4f} on L/14"])
chk("Limitations. (1)", [f"has {_gmiss_ct['test']} test rows ({_gmiss_ct['dev']} dev, {_gmiss_ct['train']} train)", "cannot exceed 0.875", "about ±0.04 wide", "about 0.01",
                         "rests on one pre-registered check on L4a-L14 (the L4a recipe with L/14 768-d image features; not a re-run of L4a; see `ablation/SELECTION_RULE.md` and the section above), with 3 seeds and 8 uncorrected comparisons"])
assert 0.039 < (_gMT["L4a"]["ci"]["ci95_high"] - _gMT["L4a"]["ci"]["ci95_low"]) / 2 < 0.043

# -- the .tex paragraph --
_gtr = {t_: _gCT[t_]["train"].label.value_counts() for t_ in _gCT}
_grat = lambda t_: f"{max(_gtr[t_]) / min(_gtr[t_]):.1f}:1"
tchk("Task 2 is severely imbalanced", [f"has {int(_gtr['humanitarian'].max()):,} rows", f"has {int(_gtr['humanitarian'].min())}, a ratio of {_grat('humanitarian')}", f"Task 1: {_grat('informative')}", f"Task 3: {_grat('damage')}"])
tchk("compares ten treatments with the image-only head", ["each scored once on the test split"])
tchk("We select on the development split only", ["(2,000 resamples)"])
tchk("The rule has little power", [f"{int((_gy['dev'] == _gINJ).sum())} injured-or-dead and {_gmiss_ct['dev']} missing-or-found rows"])
_gbi = {f_: _gBR["feats"][f_]["test"]["metrics"] for f_ in _GF}
_gdf = {f_: _gX[f_]["tv"]["inv"]["vs_none"] for f_ in _GF}
_gdf = {f_: _gmean(f_, "inv", "test_f1") - _gmean(f_, "none", "test_f1") for f_ in _GF}
assert all(_gbi[f_]["balanced_acc"]["pairs"]["inv-none"]["ci95_low"] > 0 and _gbi[f_]["recall_vehicle"]["pairs"]["inv-none"]["ci95_low"] > 0 for f_ in _GF)
tchk("Against no weighting, inverse-frequency weighting raises test balanced accuracy",
     [f"by +{_gbi['b32']['balanced_acc']['pairs']['inv-none']['diff']:.3f} (B/32) and +{_gbi['l14']['balanced_acc']['pairs']['inv-none']['diff']:.3f} (L/14)",
      f"by +{_gbi['b32']['recall_vehicle']['pairs']['inv-none']['diff']:.3f} and +{_gbi['l14']['recall_vehicle']['pairs']['inv-none']['diff']:.3f}",
      f"({_gdf['b32']:+.3f} and {_gdf['l14']:+.3f})"])
tchk("The effective-number weights have the highest test macro-F1", [f"({_gmean('b32', 'eff_num', 'test_f1'):.3f} and {_gmean('l14', 'eff_num', 'test_f1'):.3f})"])
tchk("Against inverse-frequency weighting their paired interval excludes zero only marginally",
     [f"(lower bounds {_gsg(_gX['b32']['tv']['eff_num']['vs_inv']['ci95_low'])} and {_gsg(_gX['l14']['tv']['eff_num']['vs_inv']['ci95_low'])})", "so we do not read it as an improvement"])
tchk("Pairwise comparisons were not corrected for multiple testing across about ten arms and two encoders", ["weak evidence"])
tchk("Beyond macro-F1 we report balanced accuracy", [])
tchk("For the full model (L4a) the three-seed means are", [f"balanced accuracy {np.mean(_gMT['L4a']['bal']):.3f}", f"MCC {np.mean(_gMT['L4a']['mcc']):.3f}", f"ROC-AUC {np.mean(_gMT['L4a']['roc']):.3f}", f"PR-AUC {np.mean(_gMT['L4a']['pr']):.3f}",
                                                           f"at seed 42 is {_gMT['L4a']['ci']['ci95_low']:.3f} to {_gMT['L4a']['ci']['ci95_high']:.3f}", "95% test-sampling interval", "about ± 0.04 wide", "about 0.01"])
tchk("Balancing helps the Task 1 text model", [f"(macro-F1 {_gT1['arms']['balanced']['macro_f1']:.3f} against {_gT1['arms']['none']['macro_f1']:.3f}", f"(0.{int(round(_gT3['probes']['fused']['arms']['balanced']['macro_f1'] * 1000)):03d} against 0.{int(round(_gT3['probes']['fused']['arms']['none']['macro_f1'] * 1000)):03d})",
                                                 f"gives {_gKN['votes']['class_balanced']['macro_f1']:.3f} against {_gKN['votes']['majority']['macro_f1']:.3f}"])
tchk("has F1 = 0 under every treatment", [f"({_gmiss_ct['train']} training, {_gmiss_ct['dev']} development and {_gmiss_ct['test']} test rows)", "F1 = 0 under", "cap of 0.875"])
_gLAB = {"none": "none", "inv": "inv (current)", "sqrt_inv": "sqrt_inv", "eff_num": "eff_num", "focal1": "focal1", "focal2": "focal2", "focal2_inv": "focal2_inv", "sampler": "sampler", "logit_adj": "logit_adj", "none_posthoc": "none_posthoc"}
for _a in _GARMS:
    _c = [f"{_gmean('b32', _a, 'dev_f1'):.3f}", f"{_gmean('b32', _a, 'test_f1'):.3f} ± {np.std(_gseeds('b32', _a, 'test_f1'), ddof=1):.3f}", f"{_gmean('b32', _a, 'test_bal'):.3f}",
          f"{_gmean('l14', _a, 'dev_f1'):.3f}", f"{_gmean('l14', _a, 'test_f1'):.3f} ± {np.std(_gseeds('l14', _a, 'test_f1'), ddof=1):.3f}", f"{_gmean('l14', _a, 'test_bal'):.3f}"]
    tchk(f"{_gLAB[_a]} & {_c[0]} & ", _c[1:], cover_anchor=True)
tchk("Task 2 class-imbalance treatments, image-only head on cached CLIP features", ["seeds 42, 43, 44"])
tchk("Whether weighting matters inside a fusion model was fixed in advance", ["seeds 42, 43, 44"])
tchk("Removing weighting lowers balanced accuracy (dev", [f"(dev {_gsg(_gdd['bal']['diff'])}, test {_gsg(_gtt['bal']['diff'])}; both intervals exclude zero)", f"on test ({_gsg(_gtt['veh']['diff'])})",
                                                         f"(interval upper bound {_gsg(_gdd['veh']['ci95_high'])})",
                                                         f"(test {_gsg(_gtt['f1']['diff'])}, interval {_gsg(_gtt['f1']['ci95_low'])} to {_gsg(_gtt['f1']['ci95_high'])}; seed std {min(_gsdt):.3f} to {max(_gsdt):.3f} on test, the same size as the {abs(_gtt['f1']['diff']):.3f} gap)"])
tchk("Injured-or-dead recall is also lower without weighting", [f"with {_gsup['dev']['inj']} and {_gsup['test']['inj']} rows"])
_gx4 = lambda x_: f"{x_:+.4f}"
for _lab, _m in (("macro-F1", "f1"), ("balanced acc.", "bal"), ("recall, injured or dead", "inj"), ("recall, vehicle damage", "veh")):
    _c = {sp_: (_gm(_gFM[("inv", sp_)][_m]), _gm(_gFM[("none", sp_)][_m]), f"{_gx4(_gFD[(sp_, _m)][0]['diff'])} [{_gx4(_gFD[(sp_, _m)][0]['ci95_low'])}, {_gx4(_gFD[(sp_, _m)][0]['ci95_high'])}]") for sp_ in ("dev", "test")}
    tchk(f"{_lab} & {_c['dev'][0]} & ", [_c["dev"][1], _c["dev"][2], _c["test"][0], _c["test"][1], _c["test"][2]], cover_anchor=True)
tchk("Metric & Dev inv & Dev none", ["Dev diff. [95% CI]", "Test diff. [95% CI]"])
tchk("Pre-registered check on L4a-L14 (the L4a recipe with CLIP L/14 features", ["seeds 42, 43, 44", "paired 95% interval",
                                                                                 f"Injured or dead has {_gsup['dev']['inj']} dev and {_gsup['test']['inj']} test rows, vehicle damage {_gsup['dev']['veh']} dev and {_gsup['test']['veh']} test rows"])

# 4k. H: Day 2 architecture comparison (docs §9.10 and the .tex paragraph). Everything is recomputed from the stored dev and test predictions in results/ablation/H
# (with the same fixed bootstrap indices as §9.9), the trainable-parameter counts from freshly built heads, and the data loaders.
import itertools as _hit
from d2_common import BACKBONE_M as _hBM, Head as _hHead

_hD = RES / "H"
_hj = lambda a_, s_: json.load(open(_hD / f"{a_}_seed{s_}.json"))
_hLR, _hDO, _hDM = (3e-5, 1e-4, 3e-4), (0.1, 0.2, 0.4), (128, 256, 512)
_hARMS = {"d1": [f"d1_{a_}" for a_ in ("clip_b32", "clip_l14", "dinov2", "resnet50")], "d2": [f"d2_{a_}" for a_ in ("minilm", "mpnet")],
          "d3": [f"d3_{k_}" for k_ in ("image_only", "text_only", "concat", "gated_late", "cross_attention")], "d4": [f"d4_lr{a_:g}_do{b_:g}_dm{c_}" for a_, b_, c_ in _hit.product(_hLR, _hDO, _hDM)]}
_hDIM = {"clip_b32": 512, "clip_l14": 768, "dinov2": 768, "resnet50": 2048, "minilm": 384, "mpnet": 768}
_hLRS = {"3e-05": "3e-5", "0.0001": "1e-4", "0.0003": "3e-4"}
_hLAB = {"d1_clip_b32": "D1 CLIP B/32", "d1_clip_l14": "D1 CLIP L/14", "d1_dinov2": "D1 DINOv2 ViT-B/14", "d1_resnet50": "D1 ResNet-50", "d2_minilm": "D2 MiniLM", "d2_mpnet": "D2 MPNet",
         "d3_image_only": "D3 image-only", "d3_text_only": "D3 text-only", "d3_concat": "D3 concat", "d3_gated_late": "D3 gated-late", "d3_cross_attention": "D3 cross-attention"}


def _hlabel(a_):
    if a_ in _hLAB:
        return _hLAB[a_]
    _, lr_, do_, dm_ = a_.split("_")
    return f"D4 lr {_hLRS[lr_[2:]]}, dropout {do_[2:]}, d_model {dm_[2:]}"


def _hspec(a_):
    """(kind, d_img, d_txt, d_model, dropout, backbones) of an arm, from its name alone (independent of the stored values)."""
    if a_.startswith("d1_"):
        b_ = a_[3:]; return "image_only", _hDIM[b_], None, 256, 0.2, (b_,)
    if a_.startswith("d2_"):
        b_ = a_[3:]; return "text_only", None, _hDIM[b_], 256, 0.2, (b_,)
    if a_.startswith("d3_"):
        k_ = a_[3:]; return k_, 768, 384, 256, 0.2, {"image_only": ("clip_l14",), "text_only": ("minilm",)}.get(k_, ("clip_l14", "minilm"))
    _, lr_, do_, dm_ = a_.split("_"); return "image_only", 768, None, int(dm_[2:]), float(do_[2:]), ("clip_l14",)


_hR = {a_: [_hj(a_, s_) for s_ in _GS] for d_ in _hARMS for a_ in _hARMS[d_]}
_hM = {}       # arm -> recomputed per-seed metrics from the stored predictions
for _a, _rs in _hR.items():
    _k, _di, _dt, _dm, _do, _bb = _hspec(_a)
    _np_ = sum(p_.numel() for p_ in _hHead(_k, 8, _di, _dt, _dm, _do).parameters())
    _m = dict(dev=[], test=[], bal=[], inj=[], veh=[], pd=[], pt=[])
    for _r in _rs:
        _pdv, _ptt = np.array(_r["pred_dev"]), np.array(_r["pred_test"])
        _m["dev"].append(float(_gf1(_gy["dev"], _pdv, labels=_gALL, average="macro", zero_division=0))); _m["test"].append(float(_gf1(_gy["test"], _ptt, labels=_gALL, average="macro", zero_division=0)))
        _m["bal"].append(float(_gba(_gy["test"], _ptt))); _m["inj"].append(float(_grec(_gy["test"], _ptt, labels=[_gINJ], average="macro", zero_division=0)))
        _m["veh"].append(float(_grec(_gy["test"], _ptt, labels=[_gVEH], average="macro", zero_division=0))); _m["pd"].append(_pdv); _m["pt"].append(_ptt)
        assert _gnear(_m["dev"][-1], _r["dev_macro_f1"]) and _gnear(_m["test"][-1], _r["test_macro_f1"]) and _gnear(_m["bal"][-1], _r["test_balanced_acc"])
        assert _gnear(_m["inj"][-1], _r["recall_injured"]) and _gnear(_m["veh"][-1], _r["recall_vehicle"]) and _r["recall_missing"] == 0.0
        assert _r["n_trainable"] == _np_ and abs(_r["backbone_params_m"] - sum(_hBM[b_] for b_ in _bb)) < 1e-9 and _r["n_test_injured"] == 10 and _r["n_test_vehicle"] == 36
    _m["n_params"], _m["bb_m"] = _np_, float(sum(_hBM[b_] for b_ in _bb)); _hM[_a] = _m
assert (int((_gy["test"] == _gINJ).sum()), int((_gy["test"] == _gVEH).sum()), int((_gy["dev"] == _gINJ).sum()), int((_gy["dev"] == _gVEH).sum())) == (10, 36, 8, 38)
# duplicates (same runs): first arm in decision order with identical dev and test predictions on all seeds
_hCAN, _hSEEN = {}, {}
for _d in ("d1", "d2", "d3", "d4"):
    for _a in _hARMS[_d]:
        _hCAN[_a] = _hSEEN.setdefault(tuple((tuple(r_["pred_dev"]), tuple(r_["pred_test"])) for r_ in _hR[_a]), _a)
_hDUP = {a_: c_ for a_, c_ in _hCAN.items() if a_ != c_}
assert _hDUP == {"d3_image_only": "d1_clip_l14", "d3_text_only": "d2_minilm", "d4_lr0.0001_do0.2_dm256": "d1_clip_l14"}
_hN = sum(len(v_) for v_ in _hARMS.values()); _hNd = len(set(_hCAN.values())); assert (_hN, _hNd) == (38, 35)
# validation claims: B/32 and L/14 reproduce the stored image-only runs exactly; D3 image-only == D1 L/14
for _arm, _st in (("d1_clip_b32", "b_t2_imageonly_B32_raw"), ("d1_clip_l14", "b_t2_imageonly_L14_norm")):
    for _s, _r in zip(_GS, _hR[_arm]):
        _x = json.load(open(RES / "X" / f"{_st}_seed{_s}.json"))
        assert _r["dev_macro_f1"] == _x["dev_macro_f1"] and _r["test_macro_f1"] == _x["test_macro_f1"] and _r["pred_test"] == _x["pred"] and _r["best_epoch"] == _x["best_epoch"] and _r["test"]["acc"] == _x["test_acc"]
# dev bootstraps (rule) and test-sampling intervals, same fixed resample indices as 9.9
_hDV = {}
_hTVc = {}
_hsel = {}
for _d in ("d1", "d2", "d3", "d4"):
    _ar = _hARMS[_d]; _mean = {a_: float(np.mean(_hM[a_]["dev"])) for a_ in _ar}; _best = max(_ar, key=lambda a_: _mean[a_])
    _vec = {a_: {s_: _gbu.macro_f1_vec(_gy["dev"], _hM[a_]["pd"][i_], 8, _gDI) for i_, s_ in enumerate(_GS)} for a_ in _ar}
    _df = {a_: _gbu.paired_diff_ci(_vec[_best], _vec[a_], point_diff=_mean[_best] - _mean[a_]) for a_ in _ar if a_ != _best}
    _tied = [_best] + [a_ for a_ in _ar if a_ != _best and not (_df[a_]["ci95_low"] > 0 or _df[a_]["ci95_high"] < 0)]
    if _d == "d4":
        _key = lambda a_: (_hM[a_]["n_params"], abs(_hLR.index(_hR[a_][0]["hp"]["lr"]) - 1) + abs(_hDO.index(_hR[a_][0]["hp"]["dropout"]) - 1), _hR[a_][0]["hp"]["lr"])
    else:
        _key = lambda a_: (_hM[a_]["n_params"], _hM[a_]["bb_m"])
    _pick = min(_tied, key=_key)
    _ss = json.load(open(_hD / f"selection_{_d}_seed42.json"))
    assert _ss["best"] == _best and _ss["tied"] == _tied and _ss["pick"] == _pick
    assert all(_gnear(_ss["best_minus_arm"][a_][q_], _df[a_][q_]) for a_ in _df for q_ in ("diff", "ci95_low", "ci95_high"))
    _hsel[_d] = dict(best=_best, tied=_tied, pick=_pick, df=_df, mean=_mean)
assert [_hsel[d_]["pick"] for d_ in ("d1", "d2", "d3", "d4")] == ["d1_clip_l14", "d2_minilm", "d3_image_only", "d4_lr0.0001_do0.2_dm128"]
_hTV = {}
for _a in _hR:
    _kc = _hCAN[_a]
    if _kc not in _hTVc:
        _hTVc[_kc] = {s_: _gbu.macro_f1_vec(_gy["test"], _hM[_kc]["pt"][i_], 8, _gTI) for i_, s_ in enumerate(_GS)}
    _hTV[_a] = _gbu.mean_ci(_hTVc[_kc], point=float(np.mean(_hM[_a]["test"])))
_hREP = json.load(open(_hD / "report_seed42.json"))
assert _hREP["arm_slots"] == 38 and _hREP["distinct_configurations"] == 35 and _hREP["duplicates"] == _hDUP
for _d in _hARMS:
    for _a in _hARMS[_d]:
        _x = _hREP["decisions"][_d]["rows"][_a]
        assert _gnear(_x["test"], float(np.mean(_hM[_a]["test"]))) and _gnear(_x["ci"][0], _hTV[_a]["ci95_low"]) and _gnear(_x["ci"][1], _hTV[_a]["ci95_high"]) and _x["tied"] == (_a in _hsel[_d]["tied"])
# --- doc lines ---
_hnote = lambda a_: f"same runs as {_hlabel(_hDUP[a_])} (not a separate arm)" if a_ in _hDUP else ""
for _d in ("d1", "d2", "d3", "d4"):
    _s_ = _hsel[_d]
    for _a in sorted(_hARMS[_d], key=lambda a_: -_s_["mean"][a_]):
        _m = _hM[_a]; _c = "best" if _a == _s_["best"] else f"{_gsg(_s_['df'][_a]['diff'])} [{_gsg(_s_['df'][_a]['ci95_low'])}, {_gsg(_s_['df'][_a]['ci95_high'])}]"
        chk(f"| {_hlabel(_a)} | {np.mean(_m['dev']):.4f} ± {np.std(_m['dev'], ddof=1):.4f} | {np.mean(_m['test']):.4f} ± {np.std(_m['test'], ddof=1):.4f} | {_gci3(_hTV[_a]['ci95_low'], _hTV[_a]['ci95_high'])} |",
            [f"{np.mean(_m['bal']):.3f} | {np.mean(_m['inj']):.3f} | {np.mean(_m['veh']):.3f} | {_m['n_params']:,} | {_m['bb_m']:.1f} | {_c} | {'yes' if _a in _s_['tied'] else 'no'} | {'pick' if _a == _s_['pick'] else ''} | {_hnote(_a)} |"])
    _nt = len(_s_["tied"]); _bs = _s_["best"]
    _tt = "no other arm is tied with it" if _nt == 1 else f"{_nt} of the {len(_hARMS[_d])} arms are tied with it (the best and {_nt - 1} other arm{'' if _nt == 2 else 's'})"
    chk(f"{_d.upper()}: the best arm on dev is {_hlabel(_bs)} ({_s_['mean'][_bs]:.4f})", [_tt, f"the rule's pick is {_hlabel(_s_['pick'])}"])
_h1 = _hsel["d1"]["df"]; assert all(_h1[a_]["ci95_low"] > 0.002 for a_ in _h1)
chk("CLIP L/14 is better than each alternative on dev", ["all three paired intervals exclude 0", "none is tied with it", "L/14 is better than B/32"])
_h2 = _hsel["d2"]["df"]["d2_minilm"]
chk("MPNet has the higher dev mean but is not distinguishable from MiniLM", [f"({_gsg(_h2['diff'])} [{_gsg(_h2['ci95_low'])}, {_gsg(_h2['ci95_high'])}])", "so the rule takes the simpler MiniLM"])
_h3 = _hsel["d3"]["df"]["d3_text_only"]
_htxt = float(np.mean(col("t2_text_only", "test_macro_f1", "B"))); _htsd = float(np.std(col("t2_text_only", "test_macro_f1", "B"), ddof=1))
chk("Text-only is far below image-only on dev", [f"(best minus arm {_gsg(_h3['diff'])} [{_gsg(_h3['ci95_low'])}, {_gsg(_h3['ci95_high'])}])", "a frozen 384-d MiniLM sentence embedding with CLIP L/14 image features", "confounds modality with encoder and feature quality",
                                                f"(test macro-F1 {_htxt:.3f} ± {_htsd:.3f}, §4) points the same way, but it is a separate setup"])
chk("Concat, gated-late and cross-attention are tied with image-only on dev", ["no resolvable difference at this n; a power limit, not evidence that fusion or attention adds nothing", "applies to frozen MiniLM token features only; it does not test fine-tuned text"])
assert set(_hsel["d3"]["tied"]) == {"d3_image_only", "d3_concat", "d3_gated_late", "d3_cross_attention"}
_hc = _hM["d3_concat"]
chk("Concat has a higher test macro-F1", [f"({np.mean(_hc['test']):.4f}) and injured-or-dead recall ({np.mean(_hc['inj']):.3f}, test n = 10)", "not selection results (dev is tied; test is never used to select)", "they are not findings"])
assert np.mean(_hc["test"]) > np.mean(_hM["d3_image_only"]["test"])
_h4 = _hsel["d4"]; _hun = [a_ for a_ in _hARMS["d4"] if a_ not in _h4["tied"]]
chk(f"{len(_h4['tied']) - 1} of the other {len(_hARMS['d4']) - 1} configurations are tied with the best on dev", ["a power limit, not evidence that hyperparameters do not matter", f"only {', '.join(_hlabel(a_) for a_ in _hun)} is distinguishable (worse)"])
assert _hun == ["d4_lr3e-05_do0.4_dm128"] and _h4["df"][_hun[0]]["ci95_low"] > 0.002
_hb, _hdf = _hM[_h4["best"]], _hM["d4_lr0.0001_do0.2_dm256"]
assert np.mean(_hb["test"]) < np.mean(_hdf["test"])
chk("The best-on-dev configuration (", [f"has test macro-F1 {np.mean(_hb['test']):.4f}, below the default's {np.mean(_hdf['test']):.4f}", "consistent with selecting on dev noise"])
_hf = _hM[_h4["pick"]]
chk("Picks and what they support.", ["the simplest model within noise of the best under the pre-registered rule; no benefit from text, attention or tuning was detected; the test is underpowered",
                                     f"scores test macro-F1 {np.mean(_hf['test']):.4f} ± {np.std(_hf['test'], ddof=1):.4f} (test-sampling interval {_gci3(_hTV[_h4['pick']]['ci95_low'], _hTV[_h4['pick']]['ci95_high'])})",
                                     f"This is not a win over v2 or L4a ({np.mean(col('t2_L4', 'test_macro_f1')):.3f} and {np.mean(col('t2_L4a', 'test_macro_f1')):.3f})", "the intervals overlap (they are about ±0.04 wide)"])
chk("Multiplicity and limitations.", [f"{_hN} arm slots ({len(_hARMS['d4'])} + {len(_hARMS['d3'])} + {len(_hARMS['d2'])} + {len(_hARMS['d1'])}), {_hNd} distinct trained configurations, no multiplicity correction",
                                      "8 injured-or-dead and 2 missing-or-found dev rows", "about ±0.04 wide against dev differences of about 0.01 to 0.03"])
chk("What \"tied\" means.", [f"({len(_h4['tied']) - 1} of the other {len(_hARMS['d4']) - 1} grid configurations, and {len(_hsel['d3']['tied']) - 1} of the other {len(_hARMS['d3']) - 1} heads) are a power limit",
                            "2,237 dev rows, 3 seeds; 8 injured-or-dead and 2 missing-or-found dev rows", "not that the arms are equal"])
_htm = json.load(open(_ART / "ablation" / "d2_feats" / "minilm" / "tokens_meta.json"))
assert _htm["dev"]["truncated_fraction"] == 0.0 and _htm["test"]["truncated_fraction"] == 0.0
_hmeta = {n_: json.load(open(_ART / "ablation" / "d2_feats" / n_ / "meta.json")) for n_ in ("dinov2", "resnet50", "minilm", "mpnet")}
assert all(round(_hmeta[n_]["params_m"], 1) == _hBM[n_] for n_ in _hmeta) and round(_bP32, 1) == _hBM["clip_b32"] and round(_bP14, 1) == _hBM["clip_l14"]
chk("Candidates (frozen features, cached", [f"CLIP ViT-B/32 (512-d raw, {_hBM['clip_b32']}M parameters)", f"CLIP ViT-L/14 (768-d L2-normalised, {_hBM['clip_l14']}M)", f"{_hBM['dinov2']}M) and ResNet-50 (`microsoft/resnet-50`, 2048-d pooled, {_hBM['resnet50']}M)",
                                           f"(384-d sentence embedding, {_hBM['minilm']}M)", f"(768-d, {_hBM['mpnet']}M)", f"{_htm['train']['truncated_fraction'] * 100:.2f}% of the training tweets are longer, none in dev or test"])
chk("Validation. Under the Day 2 trainer", ["6 of 6 runs", "(3 of 3)"])
chk("Counting. The decision tables contain", [f"{_hN} arm slots (D1 4, D2 2, D3 5, D4 27) but {_hNd} distinct trained configurations"])
chk("Table conventions (as in 9.9).", ["n = 10 injured-or-dead and n = 36 vehicle-damage rows (8 and 38 on dev)", "10,000 resamples of the test rows", "2,000 resamples of the dev rows"])
# --- tex ---
for _lab, _a in (("D1 CLIP L/14", "d1_clip_l14"), ("D1 DINOv2", "d1_dinov2"), ("D1 CLIP B/32", "d1_clip_b32"), ("D1 ResNet-50", "d1_resnet50"), ("D2 MPNet", "d2_mpnet"), ("D2 MiniLM", "d2_minilm"),
                 ("D3 concat", "d3_concat"), ("D3 gated-late", "d3_gated_late"), ("D3 cross-attention", "d3_cross_attention"), ("D4 picked (d_model 128)", "d4_lr0.0001_do0.2_dm128"), ("D4 best on dev (d_model 512)", "d4_lr0.0003_do0.1_dm512")):
    _d = _a[:2]; _m = _hM[_a]
    tchk(f"{_lab} & {np.mean(_m['dev']):.4f} & ", [f"{np.mean(_m['test']):.4f} ± {np.std(_m['test'], ddof=1):.4f}", f"{np.mean(_m['bal']):.3f}", f"{_m['n_params']:,}", "yes" if _a in _hsel[_d]["tied"] else "no", "yes" if _a == _hsel[_d]["pick"] else "no"], cover_anchor=True)
tchk("Architecture comparison on frozen features (Task 2), seeds", ["seeds 42, 43, 44", "95% interval is about ± 0.04 wide"])
tchk("The picks are CLIP L/14 (every alternative is worse on dev)", [f"learning rate 0.0001, dropout 0.2 and d_model 128 ({len(_h4['tied'])} of {len(_hARMS['d4'])} grid configurations tied)"])
tchk("The text-only against image-only comparison confounds modality", ["a frozen 384-d sentence embedding", f"(0.{int(round(_htxt * 1000)):03d})"])
tchk("Concatenation has a higher test macro-F1", [f"({np.mean(_hc['test']):.4f})", f"({np.mean(_hc['inj']):.3f}, test n = 10)"])
tchk("The configuration with the best dev score in the grid", [f"(d_model 512, test {np.mean(_hb['test']):.4f})", f"on test ({np.mean(_hdf['test']):.4f})"])
tchk("The final frozen pipeline (test macro-F1", [f"test macro-F1 {np.mean(_hf['test']):.4f}, test-sampling interval {_hTV[_h4['pick']]['ci95_low']:.3f} to {_hTV[_h4['pick']]['ci95_high']:.3f}", "about ± 0.04 wide"])
tchk("There are 38 arm slots (27, 5, 2 and 4 per decision)", [f"There are {_hN} arm slots (27, 5, 2 and 4 per decision) and {_hNd} distinct trained configurations"])

# 4l. Fine-tuned fusion against the frozen image-only head (docs 9.10.1 and the .tex paragraph), recomputed from the stored predictions of the three arms.
from transformers import AutoConfig as _lAC, AutoModel as _lAM, DistilBertConfig as _lDC, DistilBertModel as _lDM
from t2_models import FusionModel as _lFM

_lFT = _gJ("g_finetune_compare_seed42")
_lARMS = ("frozen", "f1_distilbert", "f2_minilm")
_lLAB = {"frozen": "FT frozen image-only head (Day 2 pick)", "f1_distilbert": "FT DistilBERT fine-tuned (L4a-L14, reused runs)", "f2_minilm": "FT MiniLM fine-tuned (L4a-L14-MiniLM)"}
_lML = {"f1": "macro-F1", "bal": "balanced accuracy", "inj": "recall, injured or dead", "veh": "recall, vehicle damage"}
_lP = {a_: {} for a_ in _lARMS}
_lwc = []
for _i, _s in enumerate(_GS):
    _fr = _hR["d4_lr0.0001_do0.2_dm128"][_i]
    _lP["frozen"][_s] = (np.array(_fr["pred_dev"]), np.array(_fr["pred_test"]))
    _lP["f1_distilbert"][_s] = (_gFP[("inv", _s)]["dev"], _gFP[("inv", _s)]["test"])
    _rj = _gJ(f"t2_L4a-L14-MiniLM_inv_seed{_s}")
    assert _rj["flags"]["feat"] == "l14" and _rj["flags"]["loss"] == "inv" and _rj["flags"]["text"] == "minilm" and _rj["flags"]["bert"] == "pretrained" and _rj["part"] == "G" and _rj["seed"] == _s
    _lP["f2_minilm"][_s] = (np.array(_lFT["pred"]["f2_minilm"]["dev"][str(_s)]), np.array(_lFT["pred"]["f2_minilm"]["test"][str(_s)]))
    assert np.bincount(_gy["test"] * 8 + _lP["f2_minilm"][_s][1], minlength=64).reshape(8, 8).tolist() == _rj["confusion_matrix"]
    assert _gnear(_gf1(_gy["dev"], _lP["f2_minilm"][_s][0], labels=_gALL, average="macro", zero_division=0), _rj["dev_macro_f1"]) and _gnear(_gf1(_gy["test"], _lP["f2_minilm"][_s][1], labels=_gALL, average="macro", zero_division=0), _rj["test_macro_f1"])
    _lwc.append(_rj["wall_clock_s"])
_lbc = _lDC.from_pretrained(_ART / "distilbert_config")
_lref, _lnew = _cc.DualStreamClassifier(8, bert_config=_lbc, img_dim=768), _lFM(8, _lDM(_lbc), _lbc.dim, img_dim=768)
assert [(k_, tuple(v_.shape)) for k_, v_ in _lref.state_dict().items()] == [(k_, tuple(v_.shape)) for k_, v_ in _lnew.state_dict().items()] and len(_lnew.state_dict()) == 124
_lmenc = _lAM.from_config(_lAC.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"))
_lN = {"frozen": _hM["d4_lr0.0001_do0.2_dm128"]["n_params"], "f1_distilbert": sum(p_.numel() for p_ in _lnew.parameters()), "f2_minilm": sum(p_.numel() for p_ in _lFM(8, _lmenc, _lmenc.config.hidden_size, img_dim=768).parameters())}
assert _lN == _lFT["n_params"] and _gJ("t2_L4a-L14_inv_seed42")["n_params"] == _lN["f1_distilbert"] and _gJ("t2_L4a-L14-MiniLM_inv_seed42")["n_params"] == _lN["f2_minilm"]
assert len(_gy["dev"]) == len(_gy["test"]) == 2237
_lM = {a_: {sp_: dict(f1=[float(_gf1(_gy[sp_], _lP[a_][s_][k_], labels=_gALL, average="macro", zero_division=0)) for s_ in _GS], bal=[float(_gba(_gy[sp_], _lP[a_][s_][k_])) for s_ in _GS],
                      inj=[float(_grec(_gy[sp_], _lP[a_][s_][k_], labels=[_gINJ], average="macro", zero_division=0)) for s_ in _GS],
                      veh=[float(_grec(_gy[sp_], _lP[a_][s_][k_], labels=[_gVEH], average="macro", zero_division=0)) for s_ in _GS]) for sp_, k_ in (("dev", 0), ("test", 1))} for a_ in _lARMS}
assert all(_gnear(_lM[a_][sp_][m_][i_], _lFT["point"][a_][sp_][m_][i_]) for a_ in _lARMS for sp_ in ("dev", "test") for m_ in ("f1", "bal", "inj", "veh") for i_ in range(3))
assert _lP["f1_distilbert"][42][1].tolist() != _lP["frozen"][42][1].tolist()
# rule on dev, pairs on dev and test, test-sampling CIs
_lvd = {a_: {s_: _gbu.macro_f1_vec(_gy["dev"], _lP[a_][s_][0], 8, _gDI) for s_ in _GS} for a_ in _lARMS}
_lmean = {a_: float(np.mean(_lM[a_]["dev"]["f1"])) for a_ in _lARMS}; _lbest = max(_lARMS, key=lambda a_: _lmean[a_])
_ldf = {a_: _gbu.paired_diff_ci(_lvd[_lbest], _lvd[a_], point_diff=_lmean[_lbest] - _lmean[a_]) for a_ in _lARMS if a_ != _lbest}
_ltied = [_lbest] + [a_ for a_ in _lARMS if a_ != _lbest and not (_ldf[a_]["ci95_low"] > 0 or _ldf[a_]["ci95_high"] < 0)]
_lpick = min(_ltied, key=lambda a_: (_lN[a_], 304.0))
assert _lbest == "f2_minilm" and set(_ltied) == set(_lARMS) and _lpick == "frozen" and _lFT["rule"]["pick"] == "frozen" and _lFT["rule"]["tied"] == _ltied
assert all(_gnear(_lFT["rule"]["best_minus_arm"][a_][q_], _ldf[a_][q_]) for a_ in _ldf for q_ in ("diff", "ci95_low", "ci95_high"))
_lPR = {}
for _sp, _idx in (("dev", _gDI), ("test", _gTI)):
    _vec = {}
    _k = 0 if _sp == "dev" else 1
    for _a in _lARMS:
        _rv = {s_: _gbu.recall_vec(_gy[_sp], _lP[_a][s_][_k], 8, _idx)[0] for s_ in _GS}
        _vec[_a] = dict(f1={s_: _gbu.macro_f1_vec(_gy[_sp], _lP[_a][s_][_k], 8, _idx) for s_ in _GS}, bal={s_: _gbu.balanced_acc_vec(_gy[_sp], _lP[_a][s_][_k], 8, _idx) for s_ in _GS},
                        inj={s_: _rv[s_][:, _gINJ] for s_ in _GS}, veh={s_: _rv[s_][:, _gVEH] for s_ in _GS})
    for _a in ("f1_distilbert", "f2_minilm"):
        for _m in ("f1", "bal", "inj", "veh"):
            _r = _gbu.paired_diff_ci(_vec[_a][_m], _vec["frozen"][_m], point_diff=float(np.mean(_lM[_a][_sp][_m]) - np.mean(_lM["frozen"][_sp][_m])))
            _v = _gvd(_r)
            if _m == "inj" and _v == "not distinguishable":
                _v = "not resolvable"
            _lPR[(_sp, _a, _m)] = (_r, _v)
            _st = _lFT["pairs"][f"{_sp}:{_a}:{_m}"]
            assert all(_gnear(_st[q_], _r[q_]) for q_ in ("diff", "ci95_low", "ci95_high")) and _st["verdict"] == _v
    if _sp == "test":
        _ltci = {a_: _gbu.mean_ci(_vec[a_]["f1"], point=float(np.mean(_lM[a_]["test"]["f1"]))) for a_ in _lARMS}
        assert all(_gnear(_lFT["test_ci"][a_]["ci95_low"], _ltci[a_]["ci95_low"]) and _gnear(_lFT["test_ci"][a_]["ci95_high"], _ltci[a_]["ci95_high"]) for a_ in _lARMS)
assert len(_lPR) == 16 == len(_lFT["pairs"])
_lnn = lambda sp_, m_: f" (n={_gsup[sp_]['inj' if m_ == 'inj' else 'veh']})" if m_ in ("inj", "veh") else ""
_ldc = lambda r_: f"{_gsg(r_['diff'])} [{_gsg(r_['ci95_low'])}, {_gsg(r_['ci95_high'])}]"
assert all(_lPR[(sp_, a_, "f1")][1] == "not distinguishable" for sp_ in ("dev", "test") for a_ in ("f1_distilbert", "f2_minilm"))
assert [_lPR[("dev", "f1_distilbert", "bal")][1], _lPR[("dev", "f1_distilbert", "veh")][1], _lPR[("test", "f1_distilbert", "veh")][1], _lPR[("test", "f1_distilbert", "bal")][1]] == ["worse", "worse", "worse", "not distinguishable"]
assert 0 < _lPR[("test", "f1_distilbert", "bal")][0]["ci95_high"] < 0.002 and all(_lPR[(sp_, a_, "inj")][1] == "not resolvable" for sp_ in ("dev", "test") for a_ in ("f1_distilbert", "f2_minilm"))
chk("Pre-registered in the fine-tuned fusion addendum", ["The DistilBERT arm reuses the existing L4a-L14 `inv` runs of 9.9", "so this is not a blind comparison", "only the MiniLM arm (3 new runs) was run after it"])
chk("Setup: CLIP L/14 768-d L2-normalised cached features", [f"trainable parameters ({_lN['frozen']:,} for the frozen head against {_lN['f2_minilm']:,} and {_lN['f1_distilbert']:,})", "The arms differ in more than the text", "The comparison does not isolate the text."])
chk("Validation: `FusionModel` built with DistilBERT", ["the same 124 state-dict tensors (names and shapes) as `DualStreamClassifier`", "on all 2,237 rows each", f"took {min(_lwc):.0f} to {max(_lwc):.0f} s each inside the training script, about {sum(_lwc) / 60:.0f} minutes in total"])
for _a in _lARMS:
    _d, _t = _lM[_a]["dev"], _lM[_a]["test"]
    chk(f"| {_lLAB[_a]} | {_lN[_a]:,} | {_gm(_d['f1'], 4)} | {np.mean(_d['bal']):.3f} | {_gm(_t['f1'], 4)} | {_gci3(_ltci[_a]['ci95_low'], _ltci[_a]['ci95_high'])} |",
        [f"{np.mean(_t['bal']):.3f} | {np.mean(_t['inj']):.3f} | {np.mean(_t['veh']):.3f} |"])
chk("Rule (dev only): the best arm on dev is", [f"({_lmean[_lbest]:.4f}); its paired dev difference from the frozen head is {_ldc(_ldf['frozen'])} and from DistilBERT {_ldc(_ldf['f1_distilbert'])}",
                                                "so all three arms are tied (no difference resolvable at this n, not equal)", f"the frozen image-only head ({_lN['frozen']:,} trainable parameters)"])
for _a in ("f1_distilbert", "f2_minilm"):
    for _m in ("f1", "bal", "inj", "veh"):
        _cells = " | ".join(f"{_ldc(_lPR[(sp_, _a, _m)][0])} {_lPR[(sp_, _a, _m)][1]}{_lnn(sp_, _m)}" for sp_ in ("dev", "test"))
        chk(f"| {_lLAB[_a]} | {_lML[_m]} |", [f"{_cells} |"])
chk("In these two setups fine-tuned text fusion showed no detectable benefit", ["This does not show that text is useless", "the arms differ in more than the text (trainable parameters, head and recipe, see above), and the test is underpowered"])
chk("Macro-F1 is not distinguishable from the frozen head for both fine-tuned models", [f"on dev (DistilBERT {_ldc(_lPR[('dev', 'f1_distilbert', 'f1')][0])}, MiniLM {_ldc(_lPR[('dev', 'f2_minilm', 'f1')][0])})",
                                                                                           f"on test (DistilBERT {_ldc(_lPR[('test', 'f1_distilbert', 'f1')][0])}, MiniLM {_ldc(_lPR[('test', 'f2_minilm', 'f1')][0])})"])
chk("For DistilBERT, balanced accuracy (", [f"balanced accuracy ({_ldc(_lPR[('dev', 'f1_distilbert', 'bal')][0])}) and vehicle recall ({_ldc(_lPR[('dev', 'f1_distilbert', 'veh')][0])}, n = {_gsup['dev']['veh']}) are worse than the frozen head on dev",
                                            f"on test vehicle recall is also worse ({_ldc(_lPR[('test', 'f1_distilbert', 'veh')][0])}, n = {_gsup['test']['veh']})",
                                            f"upper bound of {_gsg(_lPR[('test', 'f1_distilbert', 'bal')][0]['ci95_high'])} (it touches zero; not distinguishable)", "These come from 16 uncorrected comparisons, so they are weak evidence in either direction"])
chk("All four injured-or-dead recall comparisons", [f"(n = {_gsup['dev']['inj']} dev, n = {_gsup['test']['inj']} test) have a CI that includes 0", "\"not resolvable\" under the existing rule"])
chk("The MiniLM test columns in the table are reporting only", ["They are not selection results, and no claim here rests on them."])
chk("The DistilBERT arm and the frozen head share the same test mean", [f"({np.mean(_lM['f1_distilbert']['test']['f1']):.4f}) by coincidence"])
assert f"{np.mean(_lM['f1_distilbert']['test']['f1']):.4f}" == f"{np.mean(_lM['frozen']['test']['f1']):.4f}"
chk("Multiplicity: 9.10 has", [f"{_hN} arm slots ({_hNd} distinct trained configurations)", f"{len(_lPR)} pairwise comparisons (2 arms x 4 metrics x 2 splits)", "neither set is corrected for multiplicity"])
chk("Under the pre-registered rule the pick is the frozen image-only head", ["the simplest model within noise of the best under the pre-registered rule; fine-tuned text fusion showed no detectable benefit; the test is underpowered"])
# --- tex ---
_lTL = {"frozen": "Frozen image-only head", "f1_distilbert": "DistilBERT fine-tuned (reused runs)", "f2_minilm": "MiniLM fine-tuned"}
for _a in _lARMS:
    _d, _t = _lM[_a]["dev"], _lM[_a]["test"]
    tchk(f"{_lTL[_a]} & {_lN[_a]:,} & ", [_gm(_d["f1"], 4), f"{np.mean(_d['bal']):.3f}", _gm(_t["f1"], 4), f"{np.mean(_t['bal']):.3f}"], cover_anchor=True)
tchk("Fine-tuned fusion against the frozen image-only head (Task 2, CLIP L/14 features", ["seeds 42, 43, 44"])
tchk("Fine-tuned fusion was compared with the frozen image-only head under a rule fixed in advance", ["seeds 42, 43, 44"])
tchk("The arms differ in more than the text: the frozen head has", [f"has {_lN['frozen']:,} trainable parameters and the fine-tuned models {_lN['f2_minilm']:,} and {_lN['f1_distilbert']:,}"])
tchk("On dev all three arms are tied", [f"by {_gsg(_ldf['frozen']['diff'])} with interval {_gsg(_ldf['frozen']['ci95_low'])} to {_gsg(_ldf['frozen']['ci95_high'])}"])
_lt = {a_: _lPR[("test", a_, "f1")][0] for a_ in ("f1_distilbert", "f2_minilm")}
tchk("Macro-F1 is not distinguishable from the frozen head for either fine-tuned model", [f"(test differences {_gsg(_lt['f1_distilbert']['diff'])} and {_gsg(_lt['f2_minilm']['diff'])}, intervals {_gsg(_lt['f1_distilbert']['ci95_low'])} to {_gsg(_lt['f1_distilbert']['ci95_high'])} and {_gsg(_lt['f2_minilm']['ci95_low'])} to {_gsg(_lt['f2_minilm']['ci95_high'])})"])
tchk("For DistilBERT, balanced accuracy (", [f"balanced accuracy ({_gsg(_lPR[('dev', 'f1_distilbert', 'bal')][0]['diff'])}) and vehicle recall ({_gsg(_lPR[('dev', 'f1_distilbert', 'veh')][0]['diff'])}, n = {_gsup['dev']['veh']})",
                                           f"upper bound of {_gsg(_lPR[('test', 'f1_distilbert', 'bal')][0]['ci95_high'])}"])
tchk("These come from 16 uncorrected comparisons", ["16 uncorrected comparisons"])
tchk("The test columns are reporting only and none is a selection result", [f"({_gsup['dev']['inj']} dev and {_gsup['test']['inj']} test rows)"])
tchk("There are 38 arm slots (35 distinct configurations) and, separately", [f"There are {_hN} arm slots ({_hNd} distinct configurations) and, separately, {len(_lPR)} pairwise comparisons here"])
tchk("The three MiniLM runs took about", [f"about {sum(_lwc) / 60:.0f} minutes in total"])

# 4m. Day 3: consolidated best set of choices (docs 9.11 and the .tex paragraph and table). Every number is repeated from 9.9, 9.10 and 9.10.1 and is re-derived above (recomputed dev means, paired dev CIs, ties, picks).
_mh = lambda d_, a_: _hsel[d_]["mean"][a_]
_mdf = lambda d_, a_: _hsel[d_]["df"][a_]
_LOSSN = {"none": "unweighted", "inv": "inverse-frequency", "sqrt_inv": "sqrt-inverse", "eff_num": "effective-number", "focal1": "focal loss (gamma 1)", "focal2": "focal loss (gamma 2)",
          "focal2_inv": "focal loss (gamma 2) with inverse-frequency weights", "sampler": "weighted sampler", "logit_adj": "logit-adjusted loss", "none_posthoc": "post-hoc logit adjustment"}
_gl, _gb = _gX["l14"], _gX["b32"]
assert _gl["best"] == "focal1" and _gb["best"] == "focal2" and _gSEL == {"b32": "inv", "l14": "focal2_inv"} and len(_gl["tied"]) == len(_gb["tied"]) == 7 and len(_gl["order"]) == 10
_d4 = _hsel["d4"]["mean"]; _d4p, _d4b, _d4u, _d4d = "d4_lr0.0001_do0.2_dm128", "d4_lr0.0003_do0.1_dm512", "d4_lr3e-05_do0.4_dm128", "d4_lr0.0001_do0.2_dm256"
assert _hsel["d4"]["pick"] == _d4p and _hsel["d4"]["best"] == _d4b and [a_ for a_ in _d4 if a_ not in _hsel["d4"]["tied"]] == [_d4u]
_d3m = _hsel["d3"]["df"]
_ROWS = [("Image backbone (9.10, D1)",
          f"CLIP L/14 {_mh('d1', 'd1_clip_l14'):.4f}; DINOv2 {_mh('d1', 'd1_dinov2'):.4f}; CLIP B/32 {_mh('d1', 'd1_clip_b32'):.4f}; ResNet-50 {_mh('d1', 'd1_resnet50'):.4f}", "CLIP L/14",
          f"best arm; best minus DINOv2 {_ldc(_mdf('d1', 'd1_dinov2'))}, CLIP B/32 {_ldc(_mdf('d1', 'd1_clip_b32'))}, ResNet-50 {_ldc(_mdf('d1', 'd1_resnet50'))}", "resolvable: every alternative is worse on dev"),
         ("Text encoder (9.10, D2)", f"MPNet {_mh('d2', 'd2_mpnet'):.4f}; MiniLM {_mh('d2', 'd2_minilm'):.4f}", "MiniLM", f"best (MPNet) minus MiniLM {_ldc(_mdf('d2', 'd2_minilm'))}",
          "tie-break (simplicity): no resolvable difference; the picked head does not use a text encoder"),
         ("Head (9.10, D3)", f"image-only {_mh('d3', 'd3_image_only'):.4f}; concat {_mh('d3', 'd3_concat'):.4f}; gated-late {_mh('d3', 'd3_gated_late'):.4f}; cross-attention {_mh('d3', 'd3_cross_attention'):.4f}; text-only {_mh('d3', 'd3_text_only'):.4f}",
          "image-only head", f"best (image-only) minus concat {_ldc(_d3m['d3_concat'])}, gated-late {_ldc(_d3m['d3_gated_late'])}, cross-attention {_ldc(_d3m['d3_cross_attention'])}, text-only {_ldc(_d3m['d3_text_only'])}",
          "tie-break: image-only is the best arm and three heads are tied with it; text-only is resolvably worse (confounded: frozen MiniLM against CLIP L/14)"),
         ("Loss / imbalance treatment (9.9)",
          f"{len(_gl['order'])} arms; on L/14 the best is {_LOSSN[_gl['best']]} {_gmean('l14', _gl['best'], 'dev_f1'):.4f}, unweighted {_gmean('l14', 'none', 'dev_f1'):.4f}, inverse-frequency {_gmean('l14', 'inv', 'dev_f1'):.4f}; on B/32 the best is {_LOSSN[_gb['best']]} {_gmean('b32', _gb['best'], 'dev_f1'):.4f} and inverse-frequency {_gmean('b32', 'inv', 'dev_f1'):.4f}",
          "inverse-frequency (retained; no new loss adopted)",
          f"best minus inverse-frequency: B/32 {_ldc(_gb['dd']['inv'])}, L/14 {_ldc(_gl['dd']['inv'])}; {len(_gl['tied'])} of {len(_gl['order'])} arms tied with the best on both backbones",
          "not resolvable: under the 9.9 rule (tie-break by dev balanced accuracy, not simplicity) the backbones pick different arms (inverse-frequency on B/32, focal loss (gamma 2) with inverse-frequency weights on L/14), so the default is kept (note b)"),
         ("Hyperparameters (9.10, D4)",
          f"{len(_d4)} configurations (lr x dropout x d_model); dev {min(_d4.values()):.4f} to {max(_d4.values()):.4f}; best lr 3e-4, dropout 0.1, d_model 512 {_d4[_d4b]:.4f}; default (lr 1e-4, dropout 0.2, d_model 256) {_d4[_d4d]:.4f}",
          f"lr 1e-4, dropout 0.2, d_model 128 (dev {_d4[_d4p]:.4f})",
          f"best minus pick {_ldc(_mdf('d4', _d4p))}; {len(_hsel['d4']['tied'])} of {len(_d4)} configurations tied; only lr 3e-5, dropout 0.4, d_model 128 is worse ({_ldc(_mdf('d4', _d4u))})",
          "tie-break (simplicity): no resolvable difference; the pick is nominally below the default, not an improvement (note a)"),
         ("Fine-tuned vs frozen (9.10.1)", f"frozen image-only head {_lmean['frozen']:.4f}; DistilBERT fine-tuned {_lmean['f1_distilbert']:.4f}; MiniLM fine-tuned {_lmean['f2_minilm']:.4f}", "frozen image-only head",
          f"best (MiniLM) minus frozen {_ldc(_ldf['frozen'])}, minus DistilBERT {_ldc(_ldf['f1_distilbert'])}", "tie-break (simplicity): no resolvable difference")]
for _dn, _c, _p, _e, _b in _ROWS:
    chk(f"| {_dn} |", [f"| {_c} | {_p} | {_e} | {_b} |"])
chk("One row per architectural decision", ["Nothing new was run for this table", "mean dev macro-F1 over three seeds", "paired dev bootstrap of the best arm minus the candidate (95% CI)"])
# notes (a) and (b)
_np_ = lambda a_: (float(np.mean(_hM[a_]["test"])), float(np.std(_hM[a_]["test"], ddof=1)), _hTV[a_]["ci95_low"], _hTV[a_]["ci95_high"])
_pk_t, _df_t = _np_(_d4p), _np_(_d4d)
assert _d4[_d4p] < _d4[_d4d] and _pk_t[0] < _df_t[0]
chk("Note (a): The picked configuration", [f"(lr 1e-4, dropout 0.2, d_model 128; dev {_d4[_d4p]:.4f}, test {_pk_t[0]:.4f} ± {_pk_t[1]:.4f}, test-sampling 95% CI [{_pk_t[2]:.3f}, {_pk_t[3]:.3f}])",
                                           f"is nominally below the default d_model 256 (dev {_d4[_d4d]:.4f}, test {_df_t[0]:.4f} ± {_df_t[1]:.4f}, test-sampling 95% CI [{_df_t[2]:.3f}, {_df_t[3]:.3f}])", "It was chosen on simplicity, not improvement."])
assert all(_gmean("b32", "inv", "dev_f1") <= _gmean("b32", a_, "dev_f1") for a_ in _gb["tied"]) and [a_ for a_ in _gl["tied"] if _gmean("l14", a_, "dev_f1") < _gmean("l14", "inv", "dev_f1")] == ["focal2_inv"]
assert all(not (_gx_["dd"]["inv"]["ci95_low"] > 0 or _gx_["dd"]["inv"]["ci95_high"] < 0) for _gx_ in (_gb, _gl))
chk("Note (b): On dev, inverse-frequency is nominally", ["the lowest of the arms tied with the best on B/32 and the second lowest on L/14 (only focal loss (gamma 2) with inverse-frequency weights is lower)",
                                                         f"with gaps to the best of {_gsg(_gb['dd']['inv']['diff'])} (B/32) and {_gsg(_gl['dd']['inv']['diff'])} (L/14), but the paired CIs include 0; this is a power limit",
                                                         "Inverse-frequency is \"not shown worse\", not \"as good\"."])
chk("What the picks add up to.", [f"The picked text encoder (MiniLM) is not part of this final pipeline: the picked head is image-only and uses no text.",
                                  f"Its test macro-F1 is {np.mean(_hf['test']):.4f} ± {np.std(_hf['test'], ddof=1):.4f} (test-sampling 95% CI [{_hTV[_h4['pick']]['ci95_low']:.3f}, {_hTV[_h4['pick']]['ci95_high']:.3f}]",
                                  "D1 (the image backbone) is the only pick that is resolvable on dev", "no benefit from text, attention, tuning or fine-tuned text fusion was detected; the test is underpowered",
                                  "The loss row used the earlier rule of 9.9 (tie-break by dev balanced accuracy)"])
chk("Limitations of this table.", [f"has {_gmiss_ct['train']} train / {_gmiss_ct['dev']} dev / {_gmiss_ct['test']} test rows", "capped at 0.875", "about ±0.04 wide, against gaps between arms of about 0.01", "(3) Three seeds per arm",
                                   f"{_hN} arm slots ({_hNd} distinct configurations) in 9.10, {len(_lPR)} comparisons in 9.10.1", f"has {_gsup['dev']['inj']} injured-or-dead and {_gsup['dev']['mis']} missing-or-found rows",
                                   "(6) The simplicity tie-break favours the simplest model whenever power is low, so \"no benefit detected\" means \"not resolvable at this n\", not \"absent\"."])
# --- tex ---
_g4 = lambda r_: f"{r_['diff']:+.4f} [{r_['ci95_low']:+.4f}, {r_['ci95_high']:+.4f}]"
tchk(f"Image backbone & CLIP L/14 & CLIP L/14 ({_mh('d1', 'd1_clip_l14'):.4f}) &", ["best arm", "resolvable"], cover_anchor=True)
tchk(f"Text encoder & MiniLM & MPNet ({_mh('d2', 'd2_mpnet'):.4f}) &", [_g4(_mdf("d2", "d2_minilm")), "tie-break"], cover_anchor=True)
tchk(f"Head & image-only & image-only ({_mh('d3', 'd3_image_only'):.4f}) &", ["best arm", "tie-break (heads tied)"], cover_anchor=True)
tchk(f"Loss & inverse-frequency & focal loss ({_gmean('l14', 'focal1', 'dev_f1'):.4f}, L/14) &", [_g4(_gl["dd"]["inv"]), "no resolvable difference, default kept, not shown worse"], cover_anchor=True)
tchk(f"Hyperparameters & lr 0.0001, dropout 0.2, d_model 128 & lr 0.0003, dropout 0.1, d_model 512 ({_d4[_d4b]:.4f}) &", [_g4(_mdf("d4", _d4p)), "tie-break, not an improvement (a)"], cover_anchor=True)
tchk(f"Fine-tuned vs frozen & frozen head & MiniLM fine-tuned ({_lmean['f2_minilm']:.4f}) &", [_g4(_ldf["frozen"]), "tie-break"], cover_anchor=True)
tchk("Decision & Pick & Best arm on dev", ["[95% CI]"])
tchk("Consolidated choices (Task 2): for each decision", ["paired dev 95% interval", f"(dev {_d4[_d4p]:.4f}, test {_pk_t[0]:.4f}) is nominally below the default d_model 256 (dev {_d4[_d4d]:.4f}, test {_df_t[0]:.4f})"])
tchk("For the loss, inverse-frequency weighting is nominally", [f"(gaps {_gsg(_gb['dd']['inv']['diff'])} on B/32 and {_gsg(_gl['dd']['inv']['diff'])} on L/14)"])
tchk("The hyperparameter pick is nominally below the default", [f"below the default d_model 256 (dev {_d4[_d4p]:.4f} against {_d4[_d4d]:.4f}, test {_pk_t[0]:.4f} against {_df_t[0]:.4f})"])
tchk("The picks add up to frozen CLIP L/14 features", [f"whose test macro-F1 is {np.mean(_hf['test']):.4f} with test-sampling interval {_hTV[_h4['pick']]['ci95_low']:.3f} to {_hTV[_h4['pick']]['ci95_high']:.3f}"])
tchk("Limitations: the minority class has", [f"has {_gmiss_ct['train']} training, {_gmiss_ct['dev']} development and {_gmiss_ct['test']} test rows", "F1 = 0 in every arm", "cannot exceed 0.875", "about ± 0.04 wide against gaps between arms of about 0.01"])

# 5. coverage: every numeric token in the .tex must lie inside a verified span (or be structural LaTeX such as labels)
STRUCT = [r"[LB]/\d+", r"\\ref\{[^}]*\}", r"\\label\{[^}]*\}", r"\\begin\{tabular\}\{.*", r"\\begin\{table\*?\}\[[^\]]*\]", r"Tasks? \d\.?", r"\\setlength\{\\tabcolsep\}\{\d+pt\}"]
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
