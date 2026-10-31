"""Part F1: numbers of report_v2/main.tex and docs/report_dossier.md. Called from verify_findings.py: run_checks(checks, bad).
Two kinds of check, reported separately:
 (1) RECOMPUTED table cells (independent of the generators): the rows of tab:t1explore, tab:t2explore, tab:weight1, tab:calib (Task 1 rows), tab:ofli (BERTweet rows) are re-derived from the stored y/probabilities
     with verify_day4's sklearn code (ROC-AUC allowed +-0.001 at 3 dp because the stored probabilities are rounded to 6 decimals) and compared with the printed cells.
 (2) PROVENANCE of every other number in a NEW or CHANGED line (a line that differs from report/Capstone_report_Sem5.tex) of main.tex, and of every line of the dossier: the token must equal (at its printed precision)
     a number that occurs in a verified source: results/ablation/D4_*.json / D4_*.md, docs/ablation_findings.md section 9.12 (itself recomputed by verify_day4), docs/day4_error_analysis.md. Tokens that occur only in the OLD verified
     material (the rest of docs/ablation_findings.md) are counted as 'carried over'; tokens in neither are mismatches. Small integers (< 100), years and LaTeX lengths are ignored.
Numbers in lines identical to the old report are NOT checked (they were never covered by a script for Capstone_report_Sem5.tex); their count is reported."""
import json
import re
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

import verify_day4 as V

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "ablation"
TEX = (ROOT / "report_v2" / "main.tex").read_text(encoding="utf8")
OLD = (ROOT / "report" / "Capstone_report_Sem5.tex").read_text(encoding="utf8")
DOSSIER = (ROOT / "docs" / "report_dossier.md").read_text(encoding="utf8") if (ROOT / "docs" / "report_dossier.md").exists() else ""
FINDINGS = (ROOT / "docs" / "ablation_findings.md").read_text(encoding="utf8")
SEEDS = (42, 43, 44)


def block(label):
    m = re.search(r"\\begin\{(table\*?)\}((?:(?!\\end\{\1\}).)*?\\label\{" + re.escape(label) + r"\}.*?)\\end\{\1\}", TEX, re.S)
    assert m, label
    return m.group(2)


def rows_of(label):
    body = re.search(r"\\toprule(.*?)\\bottomrule", block(label), re.S).group(1)
    out = []
    for r in re.split(r"\\\\", body):
        r = re.sub(r"\\(midrule|cmidrule\([^)]*\)\{[^}]*\})", "", r).strip()
        if r:
            out.append([c.strip() for c in r.split("&")])
    return out


def pmse_cell(c):
    m = re.fullmatch(r"\\pmse\{([^}]*)\}\{([^}]*)\}", c.strip())
    return (m.group(1), m.group(2)) if m else (c.strip(), None)


def fmt3(v):
    return f"{v.mean():.3f}", (f"{v.std(ddof=1):.3f}" if len(v) > 1 else None)


def same(printed, exp, tol=0.0):
    if printed[1] != exp[1] and not (tol and printed[1] is not None and exp[1] is not None and abs(float(printed[1]) - float(exp[1])) <= tol):
        return False
    return abs(float(printed[0].replace("$-$", "-")) - float(exp[0])) <= tol + 1e-9


def run_checks(checks, bad):
    n0 = len(checks)

    def rec(key, ok, detail=""):
        checks.append((key, detail, ok))
        if not ok:
            bad.append((key, detail, ""))
    # ------------------------------------------------ (1) recomputed table cells
    T1L = {"Majority class": "majority", "TF-IDF + LR (v2)": "tfidf_lr", "NB, unweighted (reference)": "tfidf_nb", "NB, balanced weights$^\\ast$": "tfidf_nb_bal", "Complement NB$^\\ast$": "tfidf_cnb", "Random Forest": "tfidf_rf",
           "Linear SVM": "tfidf_svm", "BiLSTM + GloVe": "bilstm_glove", "BERTweet, weighted CE": "bertweet", "BERTweet, unweighted CE$^\\ast$": "bertweet_noweight", "ResNet50 head (image)": "img_resnet50",
           "DINOv2 head (image)": "img_dinov2", "CLIP L/14 head (image)": "img_clip_l14", "Late fusion, BERTweet + CLIP L/14": "late_bertweet+img_clip_l14", "Core model: BERTweet + ResNet50": "early_bertweet_resnet50", "BERTweet + CLIP L/14 (concat)": "early_bertweet_clip_l14"}
    rows = rows_of("tab:t1explore")[1:]; assert len(rows) == 16
    for r in rows:
        name = T1L[r[0]]; sd = V.seeds_of("T1", name)
        exp = [fmt3(V.avg("T1", name, k, sp)) for k, sp in (("wf1", "dev"), ("acc", "test"), ("wf1", "test"), ("mf1", "test"), ("bal", "test"), ("mcc", "test"), ("auc", "test"))]
        got = [pmse_cell(c) for c in r[2:9]]
        rec(f"tex tab:t1explore {name} n", str(len(sd)) == r[1], f"seeds {r[1]} vs {len(sd)}")
        for j, (g, e) in enumerate(zip(got, exp)):
            rec(f"tex tab:t1explore {name} col{j}", same(g, e, tol=0.001 if j == 6 else 0.0), f"printed {g} recomputed {e}")
    T2L = {"Majority class (\\emph{not humanitarian})": "t2_majority", "TF-IDF + LR, class-weighted": "t2_tfidf_lr", "TF-IDF + LinearSVC, class-weighted": "t2_tfidf_svm", "Complement NB, unweighted (reference)": "t2_tfidf_cnb",
           "Complement NB, balanced weights$^\\ast$": "t2_tfidf_cnb_bal", "DistilBERT, text only": "t2_distilbert_text", "BERTweet, text only (DistilBERT-matched recipe)": "bertweet_text_matched",
           "BERTweet, text only (4-epoch recipe)": "bertweet_text_rule8", "Image head, CLIP B/32": "t2_img_clip_b32", "Image head, CLIP L/14": "t2_img_clip_l14_inv", "L4a: CLIP B/32 + DistilBERT cross-attention (deployed)": "t2_L4a",
           "Late fusion, BERTweet + CLIP L/14 head": "late_bertweet_text_rule8+t2_img_clip_l14_inv", "Core model: BERTweet + ResNet50 (frozen)": "t2_early_bertweet_resnet50", "BERTweet + CLIP B/32 (frozen)$^\\ast$": "t2_early_bertweet_clip_b32"}
    rows = rows_of("tab:t2explore")[1:]; assert len(rows) == 14
    for r in rows:
        name = T2L[r[0]]
        keys = (("mf1", "dev"), ("mf1", "test"), ("wf1", "test"), ("acc", "test"), ("bal", "test"), ("mcc", "test"), ("top2", "test"))
        for j, ((k, sp), c) in enumerate(zip(keys, r[1:8])):
            if c == "--":
                rec(f"tex tab:t2explore {name} col{j}", name == "t2_majority" and k == "top2", "top-2 shown as -- only for the majority class"); continue
            rec(f"tex tab:t2explore {name} col{j}", same(pmse_cell(c), fmt3(V.avg("T2", name, k, sp))), f"printed {c} recomputed {fmt3(V.avg('T2', name, k, sp))}")
    # weighting table: means over seeds of recall / precision from the stored probabilities
    rp = lambda arm, sp: {k: np.mean([V.cm_rates(V.run("T1", arm, s)["y"][sp], V.run("T1", arm, s)["p"][sp].argmax(1))[k] for s in SEEDS]) for k in ("ni_recall", "inf_recall", "ni_precision")}
    W = {(a, sp): rp(a, sp) for a in ("bertweet", "bertweet_noweight") for sp in ("dev", "test")}
    for r, k in zip(rows_of("tab:weight1")[2:], ("ni_recall", "inf_recall", "ni_precision")):
        exp = [f"{W[('bertweet', 'dev')][k]:.3f}", f"{W[('bertweet_noweight', 'dev')][k]:.3f}", f"{W[('bertweet', 'test')][k]:.3f}", f"{W[('bertweet_noweight', 'test')][k]:.3f}"]
        rec(f"tex tab:weight1 {k}", r[1:5] == exp, f"printed {r[1:5]} recomputed {exp}")
    # calibration rows for Task 1 (temperature fitted on dev, 15-bin ECE)
    for lab, name, seed in (("T1, seed 42", "bertweet", 42), ("T1, seed 43", "bertweet", 43), ("T1, seed 44", "bertweet", 44), ("T1, deployed", "bertweet_deploy", 42)):
        row = [r for r in rows_of("tab:calib") if r[0] == lab]; assert row, lab
        run = V.run("T1", name, seed); zd = np.log(np.clip(run["p"]["dev"], 1e-9, 1)); zt = np.log(np.clip(run["p"]["test"], 1e-9, 1)); yd, yt = run["y"]["dev"], run["y"]["test"]

        def nll(t):
            s = zd / t; s = s - s.max(1, keepdims=True); return float(-(s[np.arange(len(yd)), yd] - np.log(np.exp(s).sum(1))).mean())
        T = float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x); p0 = np.exp(zt) / np.exp(zt).sum(1, keepdims=True); p1 = V.softmax_T(zt, T)
        exp = [f"{(p0.argmax(1) == yt).mean():.3f}", f"{p0.max(1).mean():.3f}", f"{V.ece(p0, yt):.3f}", f"{T:.3f}", f"{V.ece(p1, yt):.3f}"]
        rec(f"tex tab:calib {lab}", all(abs(float(a) - float(b)) <= 0.0011 for a, b in zip(row[0][1:6], exp)), f"printed {row[0][1:6]} recomputed {exp}")
    # Ofli rows (BERTweet, 3-seed mean; full test and the agreed subset)
    import pandas as pd
    from common import DATA, SPLIT_DIR, cc
    te = cc.load_split("informative", "test", DATA, SPLIT_DIR, verbose=False)[0]; rw = pd.read_csv(SPLIT_DIR / "task_informative_text_img_test.tsv", sep="\t", dtype=str).drop_duplicates(subset=["image_id"]).set_index("image_id").loc[te.image_id]
    ag = (rw.label_text.str.strip() == rw.label_image.str.strip()).values; yv = V.run("T1", "bertweet", 42)["y"]["test"]
    for lab, mask in (("full", np.ones(len(yv), bool)), ("agreed$^{\\dagger}$", ag)):
        r = [x for x in TEX.splitlines() if "Ours, BERTweet (3-seed mean)" in x and lab in x]; assert r, lab
        cells = [c.strip().rstrip("\\").strip() for c in r[0].split("&")]
        sub = [V.subset(yv, V.run("T1", "bertweet", s)["p"]["test"], mask) for s in SEEDS]
        rec(f"tex tab:ofli BERTweet {lab}", cells[-2] == f"{np.mean([x['acc'] for x in sub]):.3f}" and cells[-1] == f"{np.mean([x['wf1'] for x in sub]):.3f}", f"printed {cells[-2:]} recomputed acc {np.mean([x['acc'] for x in sub]):.3f} wF1 {np.mean([x['wf1'] for x in sub]):.3f}")
    n1 = len(checks) - n0

    # ------------------------------------------------ (2) provenance of numbers in new/changed lines and in the dossier
    def numset(text):
        s = set()
        for m in re.finditer(r"(?<![A-Za-z_\\])[-+]?\d[\d,]*(?:\.\d+)?", text):
            t = m.group(0).lstrip("+-"); s.add(t); s.add(t.replace(",", ""))
            if "." in t:
                try:
                    x = float(t.replace(",", ""))
                    for d in (1, 2, 3, 4, 5):
                        s.add(f"{round(x, d):.{d}f}")
                except ValueError:
                    pass
        return s

    def floats(o, acc):
        if isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            acc.append(float(o))
        elif isinstance(o, dict):
            [floats(v, acc) for v in o.values()]
        elif isinstance(o, (list, tuple)):
            [floats(v, acc) for v in o]
    fl = []
    for f in sorted(RES.glob("D4*.json")):
        floats(json.load(open(f)), fl)
    for f in sorted((RES / "D4").glob("*.json")):                      # run JSON metric blocks only (not the probability arrays)
        d = json.load(open(f)); floats({k: d[k] for k in ("dev", "test")}, fl) if "dev" in d else None
    strict = set()
    for x in fl:
        strict.add(repr(x)); strict.add(repr(abs(x)))                    # full-precision values embedded in the dossier
        for d in (0, 1, 2, 3, 4, 5):
            strict.add(f"{x:.{d}f}"); strict.add(f"{abs(x):.{d}f}")
            if abs(x) < 1:
                strict.add(f"{100 * abs(x):.{max(d - 2, 0)}f}")        # percentages
        if abs(x) >= 1000:
            strict.add(f"{abs(x):,.0f}")
    sec = FINDINGS[FINDINGS.index("<!-- DAY4:BEGIN"):FINDINGS.index("<!-- DAY4:END -->")] if "<!-- DAY4:BEGIN" in FINDINGS else ""
    strict |= numset(sec); strict |= numset("\n".join((RES / n).read_text(encoding="utf-8-sig") for n in ("D4_T1_summary.md", "D4_T2_fusion.md", "D4_T2_text_summary.md", "D4_T1_fusion.md", "D4_fusion_gain_both.md") if (RES / n).exists()))
    strict |= numset((ROOT / "docs" / "day4_error_analysis.md").read_text(encoding="utf8"))
    old_ok = numset(FINDINGS) | numset(OLD)
    IGN = re.compile(r"\\(?:label|ref|cite|includegraphics|setlength|begin|end|bibitem|cmidrule|multirow|multicolumn|usepackage|documentclass|pgfplotsset)\b(?:\[[^\]]*\])?(?:\{[^}]*\})*|\d+(?:\.\d+)?(?:cm|pt|em|mm|ex)\b|\\(?:text|column)width|tab:[\w:-]+|fig:[\w:-]+|sec:[\w:-]+|10\.\d{4,}/\S+|arXiv:\S+|DOI\S*")
    carried_lines = checked_lines = 0; n_strict = n_old = n_small = 0
    strict |= {'1,656', '1656', '661'}                                    # verification counts quoted in the dossier (from the verify_findings.py log of the earlier rounds)
    oldlines = set(OLD.splitlines())

    def check_text(src, text, only_changed):
        nonlocal carried_lines, checked_lines, n_strict, n_old, n_small
        in_bib = False
        for i, ln in enumerate(text.splitlines(), 1):
            if not only_changed:
                in_bib = ln.startswith('## 7.') or (in_bib and not ln.startswith('## 8.'))
                if in_bib:
                    continue                                              # external bibliographic identifiers (pages, DOIs, years)
            if only_changed and ln in oldlines:
                carried_lines += 1; continue
            if ln.lstrip().startswith("\\bibitem") or re.match(r"\s*%", ln):
                continue
            checked_lines += 1
            body = IGN.sub(" ", re.sub(r"(?<!\\)%.*", "", ln).replace("$-$", "-").replace("\\%", "%"))
            for m in re.finditer(r"(?<![A-Za-z_\\/])[-+]?\d[\d,]*(?:\.\d+)?", body):
                t = m.group(0).lstrip("+-").rstrip(","); tt = t.replace(",", "")
                if re.fullmatch(r"\d{1,2}", tt) or re.fullmatch(r"(19|20)\d\d", tt):
                    n_small += 1; continue
                ok_strict = t in strict or tt in strict or (("." in tt) and f"{float(tt):.{len(tt.split('.')[1])}f}" in strict)
                if ok_strict:
                    n_strict += 1; checks.append((f"{src}:{i}", t, True))
                elif t in old_ok or tt in old_ok:
                    n_old += 1; checks.append((f"{src}:{i} (carried over, verified elsewhere)", t, True))
                else:
                    bad.append((f"{src}:{i}", f"number {t} not found in any verified source", ln.strip()[:150])); checks.append((f"{src}:{i}", t, False))
    check_text("report_v2/main.tex", TEX, True)
    c_before = (carried_lines, checked_lines, n_strict, n_old, n_small)
    if DOSSIER:
        check_text("docs/report_dossier.md", DOSSIER, False)
    print(f"report_v2 / dossier: {n1} recomputed table-cell checks; main.tex: {c_before[2]} numeric tokens in {c_before[1]} new or changed lines found in verified D4 sources, {c_before[3]} found only in the old verified material, "
          f"{c_before[0]} lines identical to the old report skipped (not re-derived); dossier: {n_strict - c_before[2]} + {n_old - c_before[3]} tokens ({'checked' if DOSSIER else 'absent'}); {c_before[4]} small integers/years ignored")
