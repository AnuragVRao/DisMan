"""Day 4 part of verify_findings.py: re-derive every number of docs/ablation_findings.md section 9.12 from the stored run JSONs (results/ablation/D4/*.json: y and dev/test probabilities)
with code that is independent of metrics_all.py and of the analysis scripts: sklearn metrics, and a bootstrap built on a bincount confusion matrix per resample (the analysis scripts use boolean masks).
Row ids (`d4.<...>`) locate one document line each; every recomputed number string must occur on that line. Called from verify_findings.py: run(LINES, checks, bad)."""
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

from bootstrap_utils import N_BOOT, resample_idx
from common import DATA, SPLIT_DIR, cc

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "ablation"
SEEDS = (42, 43, 44)
_cache = {}


def run(task, name, seed):
    k = (task, name, seed)
    if k not in _cache:
        r = json.load(open(RES / "D4" / f"{task}_{name}_seed{seed}.json"))
        _cache[k] = dict(y={s: np.array(r[f"y_{s}"]) for s in ("dev", "test")}, p={s: np.array(r[f"probs_{s}"]) for s in ("dev", "test")}, raw=r)
    return _cache[k]


def seeds_of(task, name):
    return [s for s in SEEDS if (RES / "D4" / f"{task}_{name}_seed{s}.json").exists()]


def met(task, name, seed, sp):
    """Metrics recomputed from the stored labels and probabilities (sklearn)."""
    r = run(task, name, seed); y, p = r["y"][sp], r["p"][sp]; K = p.shape[1]; pred = p.argmax(1); L = list(range(K))
    o = dict(acc=float((pred == y).mean()), wf1=float(f1_score(y, pred, labels=L, average="weighted", zero_division=0)), mf1=float(f1_score(y, pred, labels=L, average="macro", zero_division=0)),
             bal=float(balanced_accuracy_score(y, pred)), mcc=float(matthews_corrcoef(y, pred)))
    if K == 2:
        o["auc"] = float(roc_auc_score(y, p[:, 1]))
    else:
        o["auc"] = float(np.mean([roc_auc_score(y == c, p[:, c]) for c in L if 0 < (y == c).sum() < len(y)]))
        o["top2"] = float((np.argsort(-p, 1)[:, :2] == y[:, None]).any(1).mean())
    return o


def avg(task, name, key, sp):
    v = np.array([met(task, name, s, sp)[key] for s in seeds_of(task, name)]); return v


def ms(v):
    return f"{v.mean():.4f}" + (f" ± {v.std(ddof=1):.4f}" if len(v) > 1 else "")


def sgs(v):
    return f"{v:+.4f}"


def cm_stats(y, pred, idx, K):
    """Per-resample macro-F1, weighted F1 and balanced accuracy from a bincount confusion matrix (independent of the analysis scripts' masks)."""
    nb, n = idx.shape; yt, yp = y[idx], pred[idx]
    flat = (yt * K + yp) + (np.arange(nb) * K * K)[:, None]
    cm = np.bincount(flat.ravel(), minlength=nb * K * K).reshape(nb, K, K)
    tp = np.einsum("bii->bi", cm); sup, prd = cm.sum(2), cm.sum(1)
    f1 = np.where(sup + prd > 0, 2 * tp / np.maximum(sup + prd, 1), 0.0)
    return dict(macro=f1.mean(1), weighted=(f1 * sup).sum(1) / n, bal=(tp / np.maximum(sup, 1)).mean(1))


def paired(task, a, b, sp, key):
    """Seed-averaged paired bootstrap of a - b for macro / weighted / bal: (point, lo, hi, verdict)."""
    sa, sb = seeds_of(task, a), seeds_of(task, b); assert sa == sb == list(SEEDS), (a, b)
    y = run(task, a, 42)["y"][sp]; K = run(task, a, 42)["p"][sp].shape[1]; idx = resample_idx(len(y), N_BOOT)
    d = np.mean([cm_stats(y, run(task, a, s)["p"][sp].argmax(1), idx, K)[key] - cm_stats(y, run(task, b, s)["p"][sp].argmax(1), idx, K)[key] for s in SEEDS], 0)
    pt = np.mean([{"macro": met(task, a, s, sp)["mf1"] - met(task, b, s, sp)["mf1"], "weighted": met(task, a, s, sp)["wf1"] - met(task, b, s, sp)["wf1"],
                   "bal": met(task, a, s, sp)["bal"] - met(task, b, s, sp)["bal"]}[key] for s in SEEDS])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(pt), float(lo), float(hi), ("better" if lo > 0 else "worse" if hi < 0 else "not distinguishable")


def h_met(name, s, sp, y):
    """Stored (labels-only) Day 2 runs in results/ablation/H."""
    pr = np.array(json.load(open(RES / "H" / f"{name}_seed{s}.json"))[f"pred_{sp}"]); L = list(range(8))
    return dict(mf1=float(f1_score(y, pr, labels=L, average="macro", zero_division=0)), wf1=float(f1_score(y, pr, labels=L, average="weighted", zero_division=0)))


def subset(y, p, m):
    yt, pr = y[m], p[m].argmax(1); pres = sorted(set(yt) | set(pr)); K = p.shape[1]
    return dict(acc=float((yt == pr).mean()), wf1=float(f1_score(yt, pr, labels=list(range(K)), average="weighted", zero_division=0)), mf1=float(f1_score(yt, pr, labels=pres, average="macro", zero_division=0)),
                all=float(f1_score(yt, pr, labels=list(range(K)), average="macro", zero_division=0)))


def ece(p, y, nb=15):
    conf, pred = p.max(1), p.argmax(1); e = np.linspace(0, 1, nb + 1); t = 0.0
    for lo, hi in zip(e[:-1], e[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            t += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(t)


def softmax_T(logp, T):
    z = logp / T; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def run_checks(LINES, checks, bad):
    def line(key):
        hits = [l for l in LINES if key in l]
        return hits[0] if len(hits) == 1 else None

    def ck(key, strings):
        l = line(key)
        if l is None:
            bad.append((key, "row id not found exactly once in section 9.12", "")); return
        for s in ([strings] if isinstance(strings, str) else strings):
            ok = s in l; checks.append((key, s, ok))
            if not ok:
                bad.append((key, s, l.strip()[:240]))

    ids = lambda prefix: sorted({m for l in LINES for m in re.findall(r"`d4\." + re.escape(prefix) + r"\.([A-Za-z0-9_+.]+)`", l)})
    # ---- Task 1 table
    for n in [i for i in ids("t1") if i not in ("switch", "bilstm_tie", "weighting")]:
        ck(f"`d4.t1.{n}`", [f"| {len(seeds_of('T1', n))} |", ms(avg("T1", n, "wf1", "dev"))] + [ms(avg("T1", n, k, "test")) for k in ("acc", "wf1", "mf1", "bal", "mcc", "auc")])
    # ---- switch rule, BiLSTM tie, weighting
    base = [met("T1", "tfidf_lr", 42, "dev")["wf1"]] * 3
    y = run("T1", "tfidf_lr", 42)["y"]["dev"]; idx = resample_idx(len(y), N_BOOT); fb = cm_stats(y, run("T1", "tfidf_lr", 42)["p"]["dev"].argmax(1), idx, 2)["weighted"]
    per = [met("T1", "bertweet", s, "dev")["wf1"] - base[0] for s in SEEDS]
    g = np.mean([cm_stats(y, run("T1", "bertweet", s)["p"]["dev"].argmax(1), idx, 2)["weighted"] - fb for s in SEEDS], 0); lo, hi = np.percentile(g, [2.5, 97.5])
    ck("`d4.t1.switch`", [f"{base[0]:.4f}", ", ".join(sgs(x) for x in per), sgs(np.mean(per)), f"[{lo:+.4f}, {hi:+.4f}]", "positive at every seed: " + str(all(x > 0 for x in per)),
                          "switch the deployed Task 1 model to BERTweet" if (lo > 0 and all(x > 0 for x in per)) else "keep TF-IDF + LR"])
    g = np.mean([cm_stats(y, run("T1", "bilstm_glove", s)["p"]["dev"].argmax(1), idx, 2)["weighted"] - fb for s in SEEDS], 0); lo, hi = np.percentile(g, [2.5, 97.5])
    pt = np.mean([met("T1", "bilstm_glove", s, "dev")["wf1"] for s in SEEDS]) - base[0]
    ck("`d4.t1.bilstm_tie`", [sgs(pt), f"[{lo:+.4f}, {hi:+.4f}]", "tied / not resolvable" if lo <= 0 <= hi else "resolved"])
    pt, lo, hi, _ = paired("T1", "bertweet", "bertweet_noweight", "dev", "bal")
    rp = lambda arm, sp: {k: np.mean([cm_rates(run("T1", arm, s)["y"][sp], run("T1", arm, s)["p"][sp].argmax(1))[k] for s in SEEDS]) for k in ("ni_recall", "inf_recall", "ni_precision")}
    w_, u_ = {sp: rp("bertweet", sp) for sp in ("dev", "test")}, {sp: rp("bertweet_noweight", sp) for sp in ("dev", "test")}
    ck("`d4.t1.weighting`", [f"{sgs(pt)} [{lo:+.4f}, {hi:+.4f}]"] + [f"{w_['dev'][k]:.4f} / {w_['test'][k]:.4f} vs {u_['dev'][k]:.4f} / {u_['test'][k]:.4f}" for k in ("ni_recall", "inf_recall", "ni_precision")])
    # ---- Task 1 fusion
    FUSED1 = [n for n in ids("t1f")]
    def best_uni(s):
        t, i = met("T1", "bertweet", s, "dev")["wf1"], met("T1", "img_clip_l14", s, "dev")["wf1"]; return "bertweet" if t >= i else "img_clip_l14"
    for n in FUSED1:
        gs = {(sp, k): np.array([met("T1", n, s, sp)[k] - met("T1", best_uni(s), s, sp)[k] for s in SEEDS]) for sp in ("dev", "test") for k in ("wf1", "mf1")}
        ck(f"`d4.t1f.{n}`", [f"{v.mean():+.4f} ± {v.std(ddof=1):.4f}" for v in gs.values()])
    dev_means = {n: np.mean([met("T1", n, s, "dev")["wf1"] for s in SEEDS]) for n in FUSED1}; best_fused = max(dev_means, key=dev_means.get)
    for key, (a, b) in {"best_fused_vs_text_only": (best_fused, "bertweet"), "guides_core_vs_deployed": ("early_bertweet_resnet50", "bertweet")}.items():
        for sp in ("dev", "test"):
            for k, kk in (("weighted_f1", "weighted"), ("macro_f1", "macro")):
                pt, lo, hi, vd = paired("T1", a, b, sp, kk)
                ck(f"`d4.t1b.{key}.{sp}.{k}`", [f"{a} − {b}", sgs(pt), f"[{lo:+.4f}, {hi:+.4f}]", vd])
    # ---- Task 1 test subsets
    tdf = cc.load_split("informative", "test", DATA, SPLIT_DIR, verbose=False)[0]
    raw = pd.read_csv(SPLIT_DIR / "task_informative_text_img_test.tsv", sep="\t", dtype=str).drop_duplicates(subset=["image_id"]).set_index("image_id")
    ag1 = (raw.loc[tdf.image_id, "label_text"].str.strip().values == raw.loc[tdf.image_id, "label_image"].str.strip().values)
    for n in ids("t1s"):
        yv = run("T1", n, 42)["y"]["test"]
        v = {g: [subset(yv, run("T1", n, s)["p"]["test"], m) for s in seeds_of("T1", n)] for g, m in (("agree", ag1), ("disagree", ~ag1))}
        ck(f"`d4.t1s.{n}`", [f"{np.mean([x[k] for x in v[g]]):.4f}" for g in v for k in ("wf1", "mf1", "acc")])
    # ---- deployment, train/serve, latency
    o, d = run("T1", "bertweet", 42)["raw"], run("T1", "bertweet_deploy", 42)["raw"]
    T = json.load(open(ROOT / "artifacts" / "task1_bertweet" / "t1_meta.json"))["temperature_dev"]
    agr = float((run("T1", "bertweet", 42)["p"]["test"].argmax(1) == run("T1", "bertweet_deploy", 42)["p"]["test"].argmax(1)).mean())
    ece_s = {sp: (ece(run("T1", "bertweet_deploy", 42)["p"][sp], run("T1", "bertweet_deploy", 42)["y"][sp]), ece(softmax_T(np.log(np.clip(run("T1", "bertweet_deploy", 42)["p"][sp], 1e-9, 1)), T), run("T1", "bertweet_deploy", 42)["y"][sp])) for sp in ("dev", "test")}
    ck("`d4.deploy`", [f"dev weighted F1 {met('T1', 'bertweet_deploy', 42, 'dev')['wf1']:.4f} (original run {met('T1', 'bertweet', 42, 'dev')['wf1']:.4f})", f"test weighted F1 {met('T1', 'bertweet_deploy', 42, 'test')['wf1']:.4f} (original {met('T1', 'bertweet', 42, 'test')['wf1']:.4f})",
                       f"best epoch {d['best_epoch']} (original {o['best_epoch']})", f"agreement {agr:.4f}", f"dev temperature {T:.3f}", f"ECE dev {ece_s['dev'][0]:.4f} to {ece_s['dev'][1]:.4f}", f"test {ece_s['test'][0]:.4f} to {ece_s['test'][1]:.4f}"])
    tsj = json.load(open(RES / "D4_t1_trainserve.json")); rr = tsj["rows"]
    ck("`d4.trainserve`", [f"{len(rr)} fixed-seed test tweets", f"on {sum(r['argmax_stored_raw'] == r['argmax_demo_raw'] for r in rr)} of {len(rr)} (unscaled)", f"{sum(r['argmax_stored_cal'] == r['argmax_demo_cal'] for r in rr)} of {len(rr)} (calibrated)",
                           f"largest probability difference {max(r['max_abs_diff_raw'] for r in rr):.4f}"])
    lat = json.load(open(RES / "D4_t1_latency.json"))
    for grp in ("model_call", "end_to_end", "http_round_trip"):
        for n, v in lat[grp].items():
            ck(f"`d4.lat.{grp}.{n}`", [f"| {v['median_ms']:.2f} | {v['p95_ms']:.2f} |"])
    # ---- overlap
    S = {t: {s: cc.load_split(task, s, DATA, SPLIT_DIR, verbose=False)[0] for s in ("train", "dev", "test")} for t, task in (("T1", "informative"), ("T2", "humanitarian"), ("T3", "damage"))}
    tw = {t: {s: set(S[t][s].tweet_id.astype(str)) for s in S[t]} for t in S}
    for sp in ("train", "dev", "test"):
        ck(f"`d4.ov.{sp}`", [" | ".join(str(len(tw["T1"][sp] & tw[t][s])) for t, s in (("T2", "train"), ("T2", "dev"), ("T2", "test"), ("T3", "train"), ("T3", "dev"), ("T3", "test")))])
    meta = pd.read_csv(ROOT / "artifacts" / "task3_meta.csv", dtype=str, keep_default_na=False)
    ck("`d4.ov.index`", [f"{len(set(meta.tweet_id) & tw['T1']['test'])} and {len(set(meta.rel_image) & set(S['T1']['test'].rel_image.astype(str)))}",
                         f"{len(tw['T3']['train'] & tw['T3']['test'])} train-test and {len(tw['T3']['dev'] & tw['T3']['test'])} dev-test"] + [f"(0 shared tweets)" if not (tw['T1']['train'] & tw['T1']['test']) and not (tw['T2']['train'] & tw['T2']['test']) else "NOT DISJOINT"])
    # ---- Task 2 tables
    for n in ids("t2"):
        K = "t2"
        ck(f"`d4.t2.{n}`", [ms(avg("T2", n, "mf1", "dev"))] + [ms(avg("T2", n, k, "test")) for k in ("mf1", "wf1", "acc", "bal", "mcc", "top2", "auc")])
    for n in ids("t2t"):
        if n in ("arm", "vs_distilbert"):
            continue
        ck(f"`d4.t2t.{n}`", [ms(avg("T2", n, "mf1", "dev"))] + [ms(avg("T2", n, k, "test")) for k in ("mf1", "wf1", "acc", "bal", "mcc", "top2", "auc")])
    pt, lo, hi, vd = paired("T2", "bertweet_text_matched", "bertweet_text_rule8", "dev", "macro")
    ck("`d4.t2t.arm`", [f"{sgs(pt)} [{lo:+.4f}, {hi:+.4f}]", "tied" if lo <= 0 <= hi else "resolved", "bertweet_text_rule8 (tied on dev -> simpler, rule 13)" if lo <= 0 <= hi else "pick"])
    pd_ = {sp: {k: paired("T2", "bertweet_text_matched", "t2_distilbert_text", sp, kk) for k, kk in (("mf1", "macro"), ("wf1", "weighted"))} for sp in ("dev", "test")}
    ck("`d4.t2t.vs_distilbert`", [", ".join(f"{met('T2', 'bertweet_text_matched', s, 'dev')['mf1']:.4f}" for s in SEEDS), ", ".join(f"{met('T2', 't2_distilbert_text', s, 'dev')['mf1']:.4f}" for s in SEEDS),
                                  f"{ci_(pd_['dev']['mf1'])} (point {sgs(pd_['dev']['mf1'][0])})", f"test macro-F1 {sgs(pd_['test']['mf1'][0])} {ci_(pd_['test']['mf1'])}",
                                  f"weighted F1 dev {sgs(pd_['dev']['wf1'][0])} {ci_(pd_['dev']['wf1'])}, test {sgs(pd_['test']['wf1'][0])} {ci_(pd_['test']['wf1'])}"])
    PAIRS2 = {"core_vs_L4a": ("t2_early_bertweet_resnet50", "t2_L4a"), "core_vs_image_only_clip_b32": ("t2_early_bertweet_resnet50", "t2_img_clip_b32"),
              "bertweet_matched_vs_distilbert_text": ("bertweet_text_matched", "t2_distilbert_text"), "posthoc_clip_b32_vs_L4a": ("t2_early_bertweet_clip_b32", "t2_L4a"),
              "posthoc_clip_b32_vs_core": ("t2_early_bertweet_clip_b32", "t2_early_bertweet_resnet50")}
    for key in sorted({i.rsplit(".", 2)[0] for i in ids("t2b")}):
        a, b = PAIRS2[key]
        for sp in ("dev", "test"):
            for k, kk in (("macro_f1", "macro"), ("weighted_f1", "weighted")):
                pt, lo, hi, vd = paired("T2", a, b, sp, kk)
                ck(f"`d4.t2b.{key}.{sp}.{k}`", [f"{a} − {b}", sgs(pt), f"[{lo:+.4f}, {hi:+.4f}]", vd])
    pcj = json.load(open(RES / "D4_t2_param_counts.json"))
    for key in ("core_vs_L4a", "core_vs_image_only_clip_b32"):
        a, b = PAIRS2[key]; _, lo, hi, _ = paired("T2", a, b, "dev", "macro"); tied = lo <= 0 <= hi; _, wl, wh, wv = paired("T2", a, b, "dev", "weighted"); wp = paired("T2", a, b, "dev", "weighted")[0]
        other = pcj["t2_L4a"] if key == "core_vs_L4a" else pcj["t2_img_clip_b32_head"]
        # the stored counts of the non-core models must equal the n_params stored with the original runs
        assert other == json.load(open(RES / ("A" if key == "core_vs_L4a" else "B") / ("t2_L4a_seed42.json" if key == "core_vs_L4a" else "t2_image_only_seed42.json")))["n_params"]
        ck(f"`d4.t2tb.{key}`", [("yes" if tied else "no"), f"{pcj['core_bertweet_resnet50']:,} / {other:,}", (b if other < pcj["core_bertweet_resnet50"] else a), f"{sgs(wp)} [{wl:+.4f}, {wh:+.4f}] ({wv})"])
    yD = {sp: run("T2", "t2_L4a", 42)["y"][sp] for sp in ("dev", "test")}
    def best_single2(s):
        t, i = met("T2", "bertweet_text_rule8", s, "dev")["mf1"], met("T2", "t2_img_clip_l14_inv", s, "dev")["mf1"]; return "bertweet_text_rule8" if t >= i else "t2_img_clip_l14_inv"
    for n in ids("t2g"):
        g = [np.array([met("T2", n, s, sp)[k] - met("T2", best_single2(s), s, sp)[k] for s in SEEDS]) for sp in ("dev", "test") for k in ("mf1", "wf1")]
        ck(f"`d4.t2g.{n}`", [f"{v.mean():+.4f} ± {v.std(ddof=1):.4f}" for v in g])
    OWN = {"t2_early_bertweet_resnet50": ("bertweet_text_matched", None), "late_bertweet_text_rule8+t2_img_clip_l14_inv": ("bertweet_text_rule8", "t2_img_clip_l14_inv"), "t2_L4a": ("t2_distilbert_text", "t2_img_clip_b32"),
           "t2_early_bertweet_clip_b32": ("bertweet_text_matched", "t2_img_clip_b32")}
    for n in ids("t2o"):
        tx, im = OWN[n]; gs = []
        for sp in ("dev", "test"):
            for k in ("mf1", "wf1"):
                vals = []
                for s in SEEDS:
                    c1 = met("T2", tx, s, "dev")["mf1"], met("T2", tx, s, sp)[k]
                    c2 = (met("T2", im, s, "dev")["mf1"], met("T2", im, s, sp)[k]) if im else (h_met("d1_resnet50", s, "dev", yD["dev"])["mf1"], h_met("d1_resnet50", s, sp, yD[sp])[k])
                    best = c1 if c1[0] >= c2[0] else c2; vals.append(met("T2", n, s, sp)[k] - best[1])
                gs.append(np.array(vals))
        ck(f"`d4.t2o.{n}`", [f"{v.mean():+.4f} ± {v.std(ddof=1):.4f}" for v in gs])
    # ---- Task 2 subsets (Table VIII convention)
    tdf2 = cc.load_split("humanitarian", "test", DATA, SPLIT_DIR, verbose=False)[0]
    raw2 = pd.read_csv(SPLIT_DIR / "task_humanitarian_text_img_test.tsv", sep="\t", dtype=str).drop_duplicates(subset=["image_id"]).set_index("image_id")
    ag2 = raw2.loc[tdf2.image_id, "label_text"].map(cc.t2_normalise).values == raw2.loc[tdf2.image_id, "label_image"].map(cc.t2_normalise).values
    for n in ids("t2s"):
        if n == "check":
            continue
        sd = {g: [subset(yD["test"], run("T2", n, s)["p"]["test"], m) for s in SEEDS] for g, m in (("agreed", ag2), ("disagreed", ~ag2))}
        exp = []
        for g in sd:
            for k in ("mf1", "all", "wf1", "acc"):
                v = np.array([x[k] for x in sd[g]]); exp.append(f"{v.mean():.3f} ± {v.std(ddof=1):.3f}")
        ck(f"`d4.t2s.{n}`", exp)
    a1 = {g: np.mean([json.load(open(RES / "X" / f"a1_L4a_seed{s}.json"))["subsets"][g]["macro_f1"] for s in SEEDS]) for g in ("agreed", "disagreed")}
    l4 = {g: np.mean([subset(yD["test"], run("T2", "t2_L4a", s)["p"]["test"], m)["mf1"] for s in SEEDS]) for g, m in (("agreed", ag2), ("disagreed", ~ag2))}
    l4a8 = np.mean([subset(yD["test"], run("T2", "t2_L4a", s)["p"]["test"], ag2)["all"] for s in SEEDS])
    assert all(abs(l4[g] - a1[g]) < 1e-3 for g in l4), "L4a subsets do not reproduce Table VIII"
    ck("`d4.t2s.check`", [f"agreed {l4['agreed']:.3f}, disagreed {l4['disagreed']:.3f}", f"gives {l4a8:.3f} for agreed"])
    # ---- error analysis totals
    for tag, (task, nm) in (("T1", ("T1", "bertweet_deploy")), ("T2", ("T2", "t2_L4a"))):
        r = run(task, nm, 42); n_err = int((r["p"]["test"].argmax(1) != r["y"]["test"]).sum())
        ck(f"`d4.err.{tag}`", [f"{n_err} errors on {len(r['y']['test'])} test rows"])
    # ---- justification rows (numbers repeated from the rows above or from the older sections)
    dv = lambda t, n: np.mean([met(t, n, s, "dev")["wf1" if t == "T1" else "mf1"] for s in SEEDS])
    ck("`d4.j.t1_model`", [f"BERTweet {dv('T1', 'bertweet'):.4f}", f"BiLSTM {dv('T1', 'bilstm_glove'):.4f}", f"TF-IDF + LR {met('T1', 'tfidf_lr', 42, 'dev')['wf1']:.4f}", f"NB (balanced) {met('T1', 'tfidf_nb_bal', 42, 'dev')['wf1']:.4f}",
                          f"Random Forest {met('T1', 'tfidf_rf', 42, 'dev')['wf1']:.4f}", f"Linear SVM {met('T1', 'tfidf_svm', 42, 'dev')['wf1']:.4f}", f"majority {met('T1', 'majority', 42, 'dev')['wf1']:.4f}", sgs(np.mean(per))])
    ck("`d4.j.t2_text`", [f"BERTweet matched {dv('T2', 'bertweet_text_matched'):.4f}", f"rule-8 {dv('T2', 'bertweet_text_rule8'):.4f}", f"DistilBERT {dv('T2', 't2_distilbert_text'):.4f}", f"TF-IDF + LR {met('T2', 't2_tfidf_lr', 42, 'dev')['mf1']:.4f}",
                          f"{sgs(pd_['dev']['mf1'][0])} {ci_(pd_['dev']['mf1'])}"])
    h = {a: np.mean([h_met(a, s, "dev", yD["dev"])["mf1"] for s in SEEDS]) for a in ("d1_clip_l14", "d1_dinov2", "d1_clip_b32", "d1_resnet50")}
    ck("`d4.j.t2_image`", [f"CLIP L/14 {h['d1_clip_l14']:.4f}", f"DINOv2 {h['d1_dinov2']:.4f}", f"CLIP B/32 {h['d1_clip_b32']:.4f}", f"ResNet-50 {h['d1_resnet50']:.4f}"])
    src = [l for l in LINES if l.startswith("| Image backbone (9.10, D1)")][0]
    for s_ in ("+0.0366 [+0.0126, +0.0618]", "+0.0445 [+0.0111, +0.0797]", "+0.0706 [+0.0372, +0.1084]"):
        checks.append(("d4.j.t2_image vs 9.11 row", s_, s_ in src and s_ in line("`d4.j.t2_image`")))
        if not (s_ in src and s_ in line("`d4.j.t2_image`")):
            bad.append(("d4.j.t2_image vs 9.11 row", s_, ""))
    pc_, pb_ = paired("T2", "t2_early_bertweet_resnet50", "t2_L4a", "dev", "macro"), paired("T2", "t2_early_bertweet_resnet50", "t2_img_clip_b32", "dev", "macro")
    ck("`d4.j.t2_fusion`", [f"{sgs(pc_[0])} [{pc_[1]:+.4f}, {pc_[2]:+.4f}]", f"{sgs(pb_[0])} [{pb_[1]:+.4f}, {pb_[2]:+.4f}]", f"{pcj['core_bertweet_resnet50']:,} (core)", f"{pcj['t2_L4a']:,} (L4a)", f"{pcj['t2_img_clip_b32_head']:,} (image-only head)"])
    loss_src = [l for l in LINES if l.startswith("| Loss / imbalance treatment (9.9)")][0]
    for s_ in ("0.6321", "0.6237", "0.6124", "0.6016", "0.5679", "+0.0197 [-0.0122, +0.0480]", "+0.0337 [-0.0019, +0.0651]"):
        okk = s_ in loss_src and s_ in line("`d4.j.t2_loss`"); checks.append(("d4.j.t2_loss vs 9.11 row", s_, okk))
        if not okk:
            bad.append(("d4.j.t2_loss vs 9.11 row", s_, ""))
    t3 = [l for l in LINES if l.startswith("| Hybrid (probe label)")][0] + [l for l in LINES if l.startswith("| v2 5-NN (fused, k = 5)")][0] + [l for l in LINES if "Bootstrap, hybrid minus 5-NN macro-F1" in l][0]
    for s_ in ("0.6146", "0.4914", "+0.1231", "+0.0615 to +0.1861"):
        okk = s_ in t3 and s_ in line("`d4.j.t3_probe`"); checks.append(("d4.j.t3_probe vs 9.3", s_, okk))
        if not okk:
            bad.append(("d4.j.t3_probe vs 9.3", s_, ""))
    dd = [met("T1", "bertweet", s, "dev")["wf1"] for s in SEEDS]
    ck("`d4.j.limits`", [f"{met('T1', 'bertweet_deploy', 42, 'dev')['wf1']:.4f} against {dd[0]:.4f}", f"{agr * 100:.1f}%", f"({abs(dd[0] - met('T1', 'bertweet_deploy', 42, 'dev')['wf1']):.4f})", f"spread ({max(dd) - min(dd):.4f})",
                         f"{len(tw['T1']['test'] & tw['T2']['train']):,} of the {len(tw['T1']['test']):,}"])
    # t1_fusion / t1_loss justification rows
    ck("`d4.j.t1_loss`", [f"{sgs(pt_w)} [{lo_w:+.4f}, {hi_w:+.4f}]" for pt_w, lo_w, hi_w, _ in [paired("T1", "bertweet", "bertweet_noweight", "dev", "bal")]] + [f"{w_['dev']['ni_recall']:.4f} vs {u_['dev']['ni_recall']:.4f}", f"{w_['dev']['inf_recall']:.4f} vs {u_['dev']['inf_recall']:.4f}"])
    bf = paired("T1", best_fused, "bertweet", "dev", "weighted"); gc = paired("T1", "early_bertweet_resnet50", "bertweet", "dev", "weighted")
    ck("`d4.j.t1_fusion`", [f"{sgs(bf[0])} [{bf[1]:+.4f}, {bf[2]:+.4f}]", f"{sgs(gc[0])} [{gc[1]:+.4f}, {gc[2]:+.4f}]"])


def ci_(t):
    return f"[{t[1]:+.4f}, {t[2]:+.4f}]"


def cm_rates(y, pred):
    """not_informative is class 1, informative class 0 (cc.T1_LABELS order)."""
    tp1 = ((y == 1) & (pred == 1)).sum(); tp0 = ((y == 0) & (pred == 0)).sum()
    return dict(ni_recall=tp1 / (y == 1).sum(), inf_recall=tp0 / (y == 0).sum(), ni_precision=tp1 / max((pred == 1).sum(), 1))
