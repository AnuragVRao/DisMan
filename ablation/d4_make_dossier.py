"""Day 4 final round, Part E: docs/report_dossier.md, the single data source for rewriting the report prose and building the slides.
Assembled from results/ablation/D4_*.json / D4_*.md (full precision, verified by ablation/verify_findings.py) and from report_v2/main.tex (every table is converted to markdown exactly as printed, 3 dp).
Nothing is typed by hand except the classification (verdict, rule, split, n), the changed-claims list, the caveats and the slide map; all numbers come from the files."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "ablation"
J = lambda n: json.load(open(RES / n))
MD = lambda n: (RES / n).read_text(encoding="utf-8-sig").replace("\r\n", "\n").strip()
TEX = (ROOT / "report_v2" / "main.tex").read_text(encoding="utf8")


# ---------------------------------------------------------------- LaTeX table -> markdown (as printed in the report)
def balanced(s, i):
    d = 0
    for j in range(i, len(s)):
        d += (s[j] == "{") - (s[j] == "}")
        if d == 0:
            return j
    return len(s) - 1


def clean(c):
    c = re.sub(r"\\pmse\{([^}]*)\}\{([^}]*)\}", r"\1 ± \2", c)
    c = re.sub(r"\\multirow\{[^}]*\}\{[^}]*\}\{([^}]*)\}", r"\1", c)
    c = re.sub(r"\\multicolumn\{[^}]*\}\{[^}]*\}\{([^}]*)\}", r"\1", c)
    for a, b in (("$-$", "−"), ("$\\pm$", "±"), ("$\\rightarrow$", "→"), ("$^\\ast$", "*"), ("$^{\\dagger}$", "†"), ("$^\\dagger$", "†"), ("\\,", " "), ("\\%", "%"), ("\\&", "&"), ("\\_", "_"), ("~", " "), ("$\\geq$", "≥"), ("$\\times$", "×")):
        c = c.replace(a, b)
    for _ in range(3):
        c = re.sub(r"\\(?:emph|textbf|textit|texttt|footnotesize)\{([^{}]*)\}", r"\1", c)
    c = re.sub(r"\$([^$]*)\$", r"\1", c)
    return re.sub(r"\s+", " ", c).strip().replace("|", "/")


def tables(tex):
    out = []
    for m in re.finditer(r"\\begin\{(table\*?)\}(.*?)\\end\{\1\}", tex, re.S):
        blk = m.group(2); cap = ""; lab = ""
        i = blk.find("\\caption{")
        if i >= 0:
            j = balanced(blk, i + len("\\caption"))
            cap = clean(blk[i + len("\\caption{"):j])
        lm = re.search(r"\\label\{([^}]+)\}", blk); lab = lm.group(1) if lm else ""
        body = re.search(r"\\toprule(.*?)\\bottomrule", blk, re.S)
        rows = []
        if body:
            for r in re.split(r"\\\\(?:\[[^\]]*\])?", body.group(1)):
                r = re.sub(r"\\(?:midrule|cmidrule\([^)]*\)\{[^}]*\}|cmidrule\{[^}]*\})", "", r).strip()
                r = re.sub(r"\\cmidrule\(lr\)\{[^}]*\}", "", r).strip()
                if r:
                    rows.append([clean(c) for c in r.split("&")])
        out.append((lab, cap, rows))
    return out


def md_table(rows):
    if not rows:
        return "(no tabular body parsed)"
    w = max(len(r) for r in rows); rows = [r + [""] * (w - len(r)) for r in rows]
    return "\n".join(["| " + " | ".join(rows[0]) + " |", "|" + "---|" * w] + ["| " + " | ".join(r) + " |" for r in rows[1:]])


# ---------------------------------------------------------------- facts
t1 = {r["model"]: r for r in J("D4_T1_summary.json")}; sw = J("D4_switch_rule_T1.json"); wt = J("D4_T1_weighting_posthoc.json"); tie1 = J("D4_T1_bilstm_vs_tfidf_dev.json")
fu1 = J("D4_T1_fusion.json"); t2 = J("D4_T2_fusion.json"); t2t = J("D4_T2_text_summary.json"); ov = J("D4_cross_task_overlap.json"); dups = J("D4_T1_dups.json"); lat = J("D4_t1_latency.json")
ofl = J("D4_T1_ofli.json"); cal = J("D4_T1_calibration.json"); c3 = J("D4_fig_c3_data.json"); er = J("D4_error_analysis.json"); chk = J("D4_t1_deploy_check.json"); gb = J("D4_fusion_gain_both.json")
dep = json.load(open(RES / "D4" / "T1_bertweet_deploy_seed42.json")); lodo = J("D4_T1_lodo.json") if (RES / "D4_T1_lodo.json").exists() else None
s4 = lambda x: f"{x:.4f}"; sg4 = lambda x: f"{x:+.4f}"; ms4 = lambda v: f"{v[0]:.4f}" + (f" ± {v[1]:.4f}" if v[1] is not None else ""); ci4 = lambda c: f"[{c[0]:+.4f}, {c[1]:+.4f}]"
bw, lr = t1["bertweet"], t1["tfidf_lr"]; cm = t2["comparisons"]; ph = t2["posthoc"]; T2m = t2["models"]; pcs = J("D4_t2_param_counts.json")
old_tables = {lab for lab, _, _ in tables((ROOT / "report" / "Capstone_report_Sem5.tex").read_text(encoding="utf8"))}

HEAD = [
    (f"Task 1, deployed BERTweet checkpoint (test): weighted F1 {s4(dep['test']['weighted_f1'])}, macro-F1 {s4(dep['test']['macro_f1'])}, accuracy {s4(dep['test']['acc'])}", "D4/T1_bertweet_deploy_seed42.json"),
    (f"Task 1, BERTweet 3-seed mean (test): weighted F1 {ms4(bw['test_weighted_f1'])}, macro-F1 {ms4(bw['test_macro_f1'])}, accuracy {ms4(bw['test_acc'])}; the TF-IDF + LR baseline of v2: {s4(lr['test_weighted_f1'][0])}, {s4(lr['test_macro_f1'][0])}, {s4(lr['test_acc'][0])}", "D4_T1_summary.json"),
    (f"Switch rule (dev weighted F1, BERTweet minus TF-IDF + LR): {sg4(sw['seed_avg_gain'])} {ci4(sw['ci95'])}, per seed {', '.join(sg4(x) for x in sw['per_seed_point_gain'])} -> switch", "D4_switch_rule_T1.json"),
    (f"Task 1 majority-class baseline (always informative): weighted F1 {s4(t1['majority']['test_weighted_f1'][0])}, accuracy {s4(t1['majority']['test_acc'][0])}, macro-F1 {s4(t1['majority']['test_macro_f1'][0])}", "D4_T1_summary.json"),
    (f"Task 1 fusion is not distinguishable from text-only on dev: best fused minus BERTweet {sg4(fu1['bootstrap']['best_fused_vs_text_only']['dev']['weighted_f1']['point'])} {ci4(fu1['bootstrap']['best_fused_vs_text_only']['dev']['weighted_f1']['ci95'])} (weighted F1)", "D4_T1_fusion.json"),
    (f"Task 1 loss weighting: not-informative recall {s4(wt['trade_off_means']['dev']['ni_recall']['weighted'])} vs {s4(wt['trade_off_means']['dev']['ni_recall']['unweighted'])} (dev), informative recall {s4(wt['trade_off_means']['dev']['inf_recall']['weighted'])} vs {s4(wt['trade_off_means']['dev']['inf_recall']['unweighted'])}; dev balanced-accuracy gain {sg4(wt['point_gain'])} {ci4(wt['ci95'])}", "D4_T1_weighting_posthoc.json"),
    (f"Task 2, deployed L4a (test macro-F1, 3 seeds): {ms4(T2m['t2_L4a']['test_macro_f1'])}; image-only CLIP L/14 head {ms4(T2m['t2_img_clip_l14_inv']['test_macro_f1'])}; CLIP B/32 head {ms4(T2m['t2_img_clip_b32']['test_macro_f1'])}", "D4_T2_fusion.json"),
    (f"Guide's core model BERTweet + ResNet50 (test macro-F1): {ms4(T2m['t2_early_bertweet_resnet50']['test_macro_f1'])}; vs L4a on dev macro-F1 {sg4(cm['core_vs_L4a']['dev']['macro_f1']['point'])} {ci4(cm['core_vs_L4a']['dev']['macro_f1']['ci95'])} (not distinguishable; tie -> fewer parameters: L4a)", "D4_T2_fusion.json"),
    (f"Backbone-only swap (BERTweet + CLIP B/32, test macro-F1 {ms4(T2m['t2_early_bertweet_clip_b32']['test_macro_f1'])}) vs core model: dev macro-F1 {sg4(ph['posthoc_clip_b32_vs_core']['dev']['macro_f1']['point'])} {ci4(ph['posthoc_clip_b32_vs_core']['dev']['macro_f1']['ci95'])}, dev weighted F1 {sg4(ph['posthoc_clip_b32_vs_core']['dev']['weighted_f1']['point'])} {ci4(ph['posthoc_clip_b32_vs_core']['dev']['weighted_f1']['ci95'])}", "D4_T2_fusion.json"),
    (f"Task 2 text encoder: BERTweet (matched recipe) {ms4(T2m['bertweet_text_matched']['test_macro_f1'])} vs DistilBERT {ms4(T2m['t2_distilbert_text']['test_macro_f1'])} test macro-F1; paired dev interval {ci4(ph['bertweet_matched_vs_distilbert_text']['dev']['macro_f1']['ci95'])} (marginally includes zero)", "D4_T2_fusion.json"),
    (f"Agreed/disagreed, Task 1 test (weighted F1): BERTweet {c3['T1|bertweet|agreed'][0]:.4f} / {c3['T1|bertweet|disagreed'][0]:.4f}; CLIP L/14 head {c3['T1|img_clip_l14|agreed'][0]:.4f} / {c3['T1|img_clip_l14|disagreed'][0]:.4f}", "D4_fig_c3_data.json"),
    (f"Agreed/disagreed, Task 2 test (macro-F1 over classes present): BERTweet text {c3['T2|bertweet_text_rule8|agreed'][0]:.4f} / {c3['T2|bertweet_text_rule8|disagreed'][0]:.4f}; CLIP L/14 head {c3['T2|t2_img_clip_l14_inv|agreed'][0]:.4f} / {c3['T2|t2_img_clip_l14_inv|disagreed'][0]:.4f}; L4a {c3['T2|t2_L4a|agreed'][0]:.4f} / {c3['T2|t2_L4a|disagreed'][0]:.4f}", "D4_fig_c3_data.json"),
    (f"Latency (median ms): Task 1 model call TF-IDF {lat['model_call']['tfidf_lr_cpu']['median_ms']:.2f} / BERTweet GPU {lat['model_call']['bertweet_cuda']['median_ms']:.2f}; whole cascade in-process {lat['end_to_end']['tfidf_cuda']['median_ms']:.1f} / {lat['end_to_end']['bertweet_cuda']['median_ms']:.1f}; HTTP round trip {lat['http_round_trip']['tfidf_gpu_server']['median_ms']:.0f} / {lat['http_round_trip']['bertweet_gpu_server']['median_ms']:.0f}", "D4_t1_latency.json"),
    (f"Cross-task overlap: {ov['by_tweet_id']['T1_test']['T2_train']:,} of the 2,237 Task 1 test rows are in the Task 2 train split ({ov['by_tweet_id']['T1_test']['T2_dev']} dev, {ov['by_tweet_id']['T1_test']['T2_test']} test); {ov['t3_deployed_index']['T1_overlap_by_tweet_id']['test']} in the Task 3 index", "D4_cross_task_overlap.json"),
    (f"Deployed Task 1 checkpoint is a retrain: dev weighted F1 {s4(chk['dev_wf1']['deployed'])} vs {s4(chk['dev_wf1']['original'])} originally; test-prediction agreement {100 * chk['prediction_agreement']['test']:.1f}%", "D4_t1_deploy_check.json"),
    (f"Indicative Ofli row: BERTweet on the {ofl['bertweet']['n']:,} agreed rows of our test split: accuracy {ms4(ofl['bertweet']['acc'])}, weighted F1 {ms4(ofl['bertweet']['weighted_f1'])} (Ofli multimodal 0.844 / 0.842; not like-for-like)", "D4_T1_ofli.json"),
]

META = {  # label -> (what, source files, script, n, decision split / rule, verdict)
    "tab:t1explore": ("Task 1 model comparison", "D4_T1_summary.{json,md}; runs results/ablation/D4/T1_*", "d4_t1_classical.py, d4_t1_step2.py, d4_t1_bilstm.py, d4_t1_bertweet.py, d4_t1_heads.py, d4_t1_fusion.py, d4_t1_early_fusion.py; summary: d4_t1_step2.py", "2,237 test rows; 3 seeds (TF-IDF models and majority: 1 deterministic run)", "dev weighted F1 selects; test reported afterwards; tie -> fewest trainable parameters", "descriptive table; the BERTweet-vs-TF-IDF row is established by the dev rule, the rest see tab:prereg"),
    "tab:t2explore": ("Task 2 model comparison", "D4_T2_fusion.{json,md}, D4_T2_text_summary.{json,md}", "d4_t2_classical.py, d4_t2_bertweet_text.py, d4_t2_reinfer.py, d4_t2_early_fusion.py, d4_fusion_both.py; analysis: d4_t2_fusion_analysis.py, d4_t2_summary.py", "2,237 test rows; 3 seeds (TF-IDF models and majority: 1 run)", "dev macro-F1 selects; test afterwards", "descriptive; comparisons in tab:prereg"),
    "tab:prereg": ("Pre-registered and post-hoc comparisons", "D4_switch_rule_T1.json, D4_T1_fusion.json, D4_T2_fusion.json", "d4_t1_switch.py, d4_t1_fusion_gain.py, d4_t2_fusion_analysis.py", "paired bootstrap, 10,000 resamples (seed 20241025), seed-averaged over 3 seeds, dev n = 2,237", "dev decides (CI excludes 0 -> better/worse; else not distinguishable; tie -> fewest trainable parameters); test reporting only", "per row (see table); post-hoc rows are descriptive"),
    "tab:weight1": ("Task 1 loss weighting trade-off", "D4_T1_weighting_posthoc.json", "d4_t1_step2.py", "3 seeds, dev n = 2,237 (1,612 informative / 625 not informative)", "dev balanced accuracy, paired bootstrap", "post-hoc; shown to change minority recall (interval excludes 0); not a safety claim; weighted/macro-F1 not tested"),
    "tab:fgain": ("Fusion gain, both tasks, both baselines", "D4_fusion_gain_both.{json,md}", "d4_fusion_both.py", "3 seeds per row", "baseline (better component / best unimodal) picked per seed on dev; test afterwards", "descriptive (no interval)"),
    "tab:latency": ("Latency", "D4_t1_latency.json", "d4_t1_latency.py, d4_t1_http_latency.py", "100 requests (30 for HTTP) after 5 warm-ups, 3 fixed examples, one RTX 4050 laptop", "none (measurement)", "descriptive; one machine"),
    "tab:errcounts": ("Error counts by category", "D4_error_analysis.json", "d4_error_analysis.py", "all test errors (T1 340, T2 440)", "rule-based categories (rule 22)", "descriptive; categories are rule-based, not human-verified"),
    "tab:errex": ("Error examples", "D4_error_analysis.json (sample), docs/day4_error_analysis.md", "d4_error_analysis.py", "12 per task (4 shown)", "pre-set sampling rule (rule 22)", "descriptive"),
    "tab:justify": ("Justification table", "docs/ablation_findings.md section 9.12.7; docs/day4_justification_table.tex", "d4_make_findings.py", "per row", "as per row basis", "per row"),
    "tab:main": ("Table I (headline test results): Task 1 rows changed", "D4_T1_summary.json", "d4_t1_step2.py", "see rows", "-", "descriptive; the T2/T3 rows are carried over from the old report"),
    "tab:ofli": ("Ofli et al. comparison: two BERTweet rows added", "D4_T1_ofli.json", "d4_t1_calib_ofli.py", "full test (2,237) and agreed subset (1,534)", "-", "indicative only (the agreed subset of datasplit_all is not Ofli's agreed-label split)"),
    "tab:lodo": ("Leave-one-disaster-out: Task 1 BERTweet column added", "D4_T1_lodo.{json,md}", "d4_t1_lodo.py", "7 folds, one seed (42) for BERTweet; TF-IDF deterministic", "per fold: train on the other 6 events' train rows, test on the held-out event's test rows", "descriptive; single seed, per-event n small for three events"),
    "tab:calib": ("Calibration: Task 1 rows added", "D4_T1_calibration.{json,md}", "d4_t1_calib_ofli.py", "test n = 2,237", "T fitted on dev", "descriptive"),
}
FIGS = [("fig:pipeline", "Fig. 1 pipeline (TikZ, Task 1 box changed)", "report_v2/main.tex (TikZ); matplotlib PNG for slides: report_v2/figures/fig_c6_pipeline.png", "d4_make_pipeline_fig.py", "diagram", "-"),
        ("fig:ladder", "Fig. 2 Task 2 ladder (unchanged)", "carried over from the old report", "-", "3 seeds", "carried over, unverified"),
        ("fig:t1models", "Task 1 models (dev/test weighted F1)", "report_v2/figures/fig_c4_t1_models.{pdf,png}", "d4_make_figures.py (c4); data D4_T1_summary.json", "3 seeds / single runs", "descriptive"),
        ("fig:t2models", "Task 2 models (dev/test macro-F1)", "report_v2/figures/fig_c5_t2_models.{pdf,png}", "d4_make_figures.py (c5)", "3 seeds / single runs", "descriptive"),
        ("fig:fgain", "Fusion gain over own components, both tasks", "report_v2/figures/fig_c8_fusion_gain.{pdf,png}", "d4_make_figures.py (c8); data D4_fusion_gain_both.json", "3 seeds", "descriptive"),
        ("fig:agree", "Agreed/disagreed bars, both tasks (headline)", "report_v2/figures/fig_c3_agreed_disagreed.{pdf,png}; data D4_fig_c3_data.json", "d4_make_figures.py (c3)", "test: T1 agreed 1,534 / disagreed 703; T2 955 / 1,282; 3 seeds", "descriptive (no interval)"),
        ("fig:conf12", "Confusion matrices T1 (deployed) and T2 (L4a seed 42)", "report_v2/figures/fig_c1_t1_confusion.*, fig_c2_t2_confusion.*", "d4_make_figures.py (c1, c2)", "single checkpoints, test n = 2,237", "descriptive"),
        ("(supplementary)", "Contact sheets of the 12 + 12 sampled errors", "report_v2/figures/contact_T1.*, contact_T2.*", "d4_error_analysis.py", "12 per task", "descriptive; for slides")]

OUT = ["# Report dossier (the single source for rewriting the prose and building the slides)\n",
       "Generated by `ablation/d4_make_dossier.py` from `results/ablation/D4_*` (full precision) and `report_v2/main.tex` (tables as printed). Every number in `report_v2/main.tex` that is new or changed comes from these files; numbers in text carried over unchanged from `report/Capstone_report_Sem5.tex` are **not** re-derived (section 6 lists them). "
       "Verdict vocabulary: *established* (the decision rule's condition holds), *not distinguishable* (the paired 95% interval includes zero), *descriptive only* (no interval). Seeds are 42/43/44 unless stated; dev decides, test is reported afterwards.\n",
       "## 1. Headline numbers (candidates for slides)\n"]
OUT += [f"{i}. {t}  \n   *source:* `results/ablation/{s}`" for i, (t, s) in enumerate(HEAD, 1)]

OUT.append("\n## 2. Every table and figure of report_v2\n")
for lab, cap, rows in tables(TEX):
    what, src, script, n, rule, verdict = META.get(lab, ("Table carried over from the old report (unchanged)", "carried over from report/Capstone_report_Sem5.tex; the same numbers are in docs/ablation_findings.md and report/ablation_section.tex, which verify_findings.py checks, but they are not checked against the Sem5 text itself", "ablation/ scripts of the earlier rounds", "see caption", "see caption and the old report", "carried over, unverified against report_v2"))
    changed = lab in META
    OUT.append(f"\n### Table `{lab}`: {what}\n\n*Caption:* {cap}\n\n*As printed in report_v2 (3 dp):*\n\n{md_table(rows)}\n\n- **Source:** {src}\n- **Script:** {script}\n- **n / seeds:** {n}\n- **Decision split and rule:** {rule}\n- **Verdict:** {verdict}\n- **Status:** {'new or changed in report_v2' if changed else 'unchanged from the old report'}")
OUT.append("\n### Figures\n\n| Label | What | File / source | Script | n | Verdict |\n|---|---|---|---|---|---|")
OUT += [f"| `{a}` | {b} | {c} | {d} | {e} | {f} |" for a, b, c, d, e, f in FIGS]

OUT.append("\n## 3. Full-precision source tables (4 dp, as produced and verified)\n")
for title, name in (("Task 1 models", "D4_T1_summary.md"), ("Task 1 weighting, BiLSTM tie and switch rule", None), ("Task 1 fusion, bootstrap and subsets", "D4_T1_fusion.md"), ("Task 2 text-only models", "D4_T2_text_summary.md"), ("Task 2 fusion, comparisons, tie-break, subsets", "D4_T2_fusion.md"),
                    ("Fusion gain, both tasks", "D4_fusion_gain_both.md"), ("Task 1 calibration", "D4_T1_calibration.md"), ("Task 1 duplicates", "D4_T1_dups.md"), ("Task 1 leave-one-disaster-out", "D4_T1_lodo.md")):
    if name is None:
        OUT.append(f"\n### {title}\n\n- Switch rule: gain {sg4(sw['seed_avg_gain'])} {ci4(sw['ci95'])}; per seed {', '.join(sg4(x) for x in sw['per_seed_point_gain'])}; base dev weighted F1 {s4(sw['dev_weighted_f1_tfidf_lr'])}; decision: {'switch' if sw['switch'] else 'keep'}.\n"
                   f"- BiLSTM minus TF-IDF + LR (dev weighted F1): {sg4(tie1['point_gain'])} {ci4(tie1['ci95'])} -> {tie1['verdict']} (post-hoc).\n- Weighting (post-hoc): dev balanced-accuracy gain {sg4(wt['point_gain'])} {ci4(wt['ci95'])}; {wt['verdict']}.")
        continue
    if not (RES / name).exists():
        OUT.append(f"\n### {title}\n\n*(file {name} not available when the dossier was built)*")
        continue
    OUT.append(f"\n### {title}\n\n*File:* `results/ablation/{name}`\n\n" + MD(name).split("\n", 1)[1].replace("\n# ", "\n#### ").replace("\n## ", "\n#### "))
OUT.append("\n### Task 1 Ofli indicative row and error-analysis counts\n\n```json\n" + json.dumps(ofl, indent=1) + "\n```\n\n```json\n" + json.dumps({k: {x: v[x] for x in ("model", "n_test", "n_errors", "by_category", "cap_per_pair", "sample_categories", "ranked_pairs")} for k, v in er.items()}, indent=1) + "\n```")

OUT.append("""
## 4. Changed claims (old report -> report_v2)

| Old claim (`Capstone_report_Sem5.tex`) | New | Why |
|---|---|---|""")
OUT += [
    f"| Abstract / Table I: Task 1 macro-F1 **0.751** (TF-IDF + LR) | BERTweet: weighted F1 {ms4(bw['test_weighted_f1'])}, macro-F1 {ms4(bw['test_macro_f1'])} (3-seed means; deployed checkpoint {s4(dep['test']['weighted_f1'])} / {s4(dep['test']['macro_f1'])}); TF-IDF + LR kept as the v2 baseline | pre-registered dev rule triggered ({sg4(sw['seed_avg_gain'])} {ci4(sw['ci95'])}) |",
    "| IV-A: Task 1 = TF-IDF (50k features) + balanced logistic regression | BERTweet fine-tuned, raw text, weighted CE, T = %.3f; TF-IDF kept as CPU fallback | same |" % chk["temperature"],
    "| IV-E: \"raw logistic-regression probability for the Task 1 badge\" | calibrated BERTweet probability | the demo now serves BERTweet |",
    f"| IV-E: \"A model call takes 0.1-0.2 s and an end-to-end request about 1.8 s\" | model call {lat['model_call']['tfidf_lr_cpu']['median_ms']:.2f} ms (TF-IDF) / {lat['model_call']['bertweet_cuda']['median_ms']:.1f} ms (BERTweet); whole cascade in-process {lat['end_to_end']['tfidf_cuda']['median_ms']:.1f} / {lat['end_to_end']['bertweet_cuda']['median_ms']:.1f} ms; HTTP round trip {lat['http_round_trip']['tfidf_gpu_server']['median_ms']:.0f} / {lat['http_round_trip']['bertweet_gpu_server']['median_ms']:.0f} ms | the old figures have no benchmark script or log (first stated in README.md, commit e1afcb4); the 1.8 s matches the HTTP round trip, the 0.1-0.2 s matches no measurement |",
    "| IV-F: \"A linear text model for Task 1 ... within about 1.5 points of a published text CNN\" | rewritten: BERTweet replaced TF-IDF; the 1.5-point statement now describes the v2 model | switch rule |",
    "| V-C: \"Task 1 and Task 3 involve no stochastic training\" | false for Task 1: BERTweet, the BiLSTM and the heads are trained with 3 seeds; only TF-IDF models, the majority class and Task 3 are deterministic | Task 1 is now a trained neural model |",
    "| III-C: \"The informativeness and humanitarian splits are tweet-disjoint\" | within each task only; across tasks %s of 2,237 Task 1 test rows are in the Task 2 train split | cross-task overlap matrix (D4_cross_task_overlap.json) |" % f"{ov['by_tweet_id']['T1_test']['T2_train']:,}",
    "| III-D: duplicates analysed for Tasks 2 and 3 only | Task 1 added: %d exact + %d near-duplicate test tweets, %d byte-identical + %d flagged test images; re-scoring changes <= 0.0051 weighted F1 | D4_T1_dups.json |" % (dups["text"]["n_exact"], dups["text"]["n_near_incl_exact"] - dups["text"]["n_exact"], dups["image"]["n_exact_md5"], dups["image"]["n_near_incl_exact"]),
    "| VIII: \"Evaluating on text labels, or on the agreed subset as Ofli et al. did, would favour fusion\" | partly contradicted: on the full text-labelled Task 1, fusion is not distinguishable from text-only on dev (best gain %s); fusion gains appear on agreed rows only | D4_T1_fusion.json (marked \\todo in report_v2) |" % sg4(fu1["bootstrap"]["best_fused_vs_text_only"]["dev"]["weighted_f1"]["point"]),
    "| VII-D: calibration table covered Tasks 2 and 3 | Task 1 BERTweet rows added; the Task 1 model is over-confident (deployed ECE %.3f -> %.3f) | D4_T1_calibration.json |" % (cal[-1]["test"]["ece"], cal[-1]["test"]["ece_after"]),
    "| VII-C: Task 1 LODO column was TF-IDF | BERTweet column (single seed per fold) added beside it | D4_T1_lodo.json |" if lodo else "| VII-C: Task 1 LODO column was TF-IDF | **not updated: the LODO job had not finished when the dossier was built** | rerun d4_make_dossier.py after D4_T1_lodo.json exists |",
    ("| VII-C: \"Text and image fail on different unseen events: text on the flood, images on the earthquakes\" | with BERTweet (single seed) the flood is the BEST held-out Task 1 event (%.3f held out vs %.3f in distribution, %d rows); the largest losses are on the two earthquakes (Mexico %.3f -> %.3f, Iraq-Iran %.3f -> %.3f); marked \\todo in report_v2 | D4_T1_lodo.json (the TF-IDF column still shows the old pattern) |"
     % tuple([next(r for r in lodo["rows"] if r["event"] == "srilanka_floods")["held_out"][k] for k in ("macro_f1",)] + [next(r for r in lodo["rows"] if r["event"] == "srilanka_floods")["in_dist"]["macro_f1"], next(r for r in lodo["rows"] if r["event"] == "srilanka_floods")["held_out"]["n"]]
             + [next(r for r in lodo["rows"] if r["event"] == "mexico_earthquake")["in_dist"]["macro_f1"], next(r for r in lodo["rows"] if r["event"] == "mexico_earthquake")["held_out"]["macro_f1"],
                next(r for r in lodo["rows"] if r["event"] == "iraq_iran_earthquake")["in_dist"]["macro_f1"], next(r for r in lodo["rows"] if r["event"] == "iraq_iran_earthquake")["held_out"]["macro_f1"]])) if lodo else "| VII-C: flood/earthquake sentence | not updated (LODO missing) | - |",
    "| VIII Limitations: \"Task 1 and Task 3 results are single deterministic runs\" | Task 3 and the classical Task 1 models only; GPU non-determinism and the retrain are stated | D4_t1_deploy_check.json |",
    "| Implicit: the deployed Task 1 model = the v2 model | the deployed Task 1 checkpoint is a retrain of seed 42 | the original runs saved no weights |"]

OUT.append("""
## 5. Deviations and caveats

- **Retrained deployed checkpoint.** Seed 42 was retrained with `--save` because the original runs saved no weights: dev weighted F1 %s vs %s, best epoch %d vs %d, %.1f%% test-prediction agreement. Same-seed run-to-run variation (0.0032) exceeds the inter-seed dev spread (0.0030), so seed selection is within noise. Quote the deployed numbers from `bertweet_deploy`, architecture claims from the 3-seed means.
- **Post-hoc arms** (labelled * in the tables): balanced NB, Complement NB, majority class, unweighted BERTweet, BERTweet + CLIP B/32 (backbone swap), the paired BERTweet-vs-DistilBERT bootstrap, the BiLSTM-vs-TF-IDF bootstrap. Pre-registered rules: `ablation/SELECTION_RULE_DAY4.md` (rules 1-24).
- **Borderline results.** Switch rule lower bound only %s; BERTweet vs DistilBERT dev macro-F1 interval %s (marginally includes zero); core model vs L4a / vs the CLIP B/32 head on dev macro-F1 are not distinguishable (weighted F1 is worse, secondary).
- **Dev/test disagreement.** Task 1 fusion and the core model are better than text-only on test intervals (lower bounds about +0.001) but not on dev; the rule uses dev. Task 2 core model is worse than L4a on test macro-F1 (interval excludes 0) but not on dev macro-F1.
- **Tie-break basis.** Rules 14-19 of SELECTION_RULE_DAY4.md contain no tie wording; rule 2 ("Ties go to the simplest model") governs, with "simplest" = fewest trainable parameters as in the Day 2 addendum: core model %s, L4a %s, image-only B/32 head %s.
- **Agreed/disagreed numbers are descriptive** (no interval); the Task 2 subset macro-F1 averages the classes present (Table VIII definition; L4a reproduces 0.745 and 0.549).
- **Error analysis categories are rule-based** (written criteria, not reading each example); the ambiguity criterion fires rarely (T1 %d, T2 %d errors); one sampled Task 2 tweet contains a vulgar word verbatim from the data.
- **Optional candidates not run:** DeBERTa-v3, ModernBERT, SigLIP, EfficientNet-B0, zero-shot VLM/LLM, ResNet50 fine-tuning, Task 1 gated fusion, BERTweet inside the L4a cross-attention.
- **Not compiled.** No LaTeX engine is installed; the compiler binary I downloaded was blocked by the sandbox. `report_v2/main.tex` passes `ablation/d4_check_tex.py` (matched environments, labels, citations, figure files) but overfull boxes and undefined-reference warnings from a real compile are unchecked.
- **Wrong draft corrected.** The first `docs/day4_report_changes.md` targeted `report/main.tex`; it was rewritten for `Capstone_report_Sem5.tex`.
""" % (s4(dep["dev"]["weighted_f1"]), s4(chk["dev_wf1"]["original"]), chk["best_epoch"]["deployed"], chk["best_epoch"]["original"], 100 * chk["prediction_agreement"]["test"], ci4(sw["ci95"])[1:-1].split(",")[0], ci4(ph["bertweet_matched_vs_distilbert_text"]["dev"]["macro_f1"]["ci95"]),
       f"{pcs['core_bertweet_resnet50']:,}", f"{pcs['t2_L4a']:,}", f"{pcs['t2_img_clip_b32_head']:,}", er["T1"]["by_category"].get("label ambiguity", 0), er["T2"]["by_category"].get("label ambiguity", 0)))

OUT.append("""
## 6. Numbers carried over from the old report (not re-derived)

Tables and text that report_v2 inherits unchanged from `Capstone_report_Sem5.tex`: Table `tab:classes` (class distribution), `tab:ladder2`, Fig. `fig:ladder`, `tab:ladder3`, `tab:comp`, `tab:agree`, `tab:emb3`, `tab:scale`, `tab:imb`, `tab:arch`, `tab:zeroshot`, the Task 3 text, the Task 2 ladder text, the component-ablation, robustness (except the Task 1 LODO column), image-scale, class-imbalance and architecture paragraphs, and the T2/T3 rows of Table I. Their values match `docs/ablation_findings.md` and `report/ablation_section.tex`, which `verify_findings.py` checks (1,656 + 661 numeric checks, 0 mismatches), but `Capstone_report_Sem5.tex` itself was never covered by that script.

## 7. Bibliography added (each entry checked against its source)

| Key | Entry | Checked against |
|---|---|---|
| nguyen2020 | BERTweet, EMNLP 2020 System Demonstrations, pp. 9-14 (Nguyen, Vu, Tuan Nguyen) | ACL Anthology record 2020.emnlp-demos.2 |
| pennington2014 | GloVe, EMNLP 2014, pp. 1532-1543 | ACL Anthology D14-1162 |
| liu2019 | RoBERTa, arXiv:1907.11692 | arXiv API |
| hochreiter1997 | LSTM, Neural Computation 9(8):1735-1780 | Crossref 10.1162/neco.1997.9.8.1735 |
| schuster1997 | Bidirectional RNNs, IEEE TSP 45(11):2673-2681 | Crossref 10.1109/78.650093 |
| breiman2001 | Random forests, Machine Learning 45(1):5-32 | Crossref 10.1023/A:1010933404324 |
| pedregosa2011 | Scikit-learn, JMLR 12:2825-2830 | jmlr.org confirmed the title and start page (2825); the end page is from memory (`%% CHECK` in the .tex) |

## 8. Slide map

**No slide file exists in the repository**, so the old slides 1-18 cannot be mapped from their content. You named slides 4, 8, 14, 15 and 17 as affected; apart from those I only know what to look for. Update any slide that contains: the Task 1 figure 0.751 or "TF-IDF"/"linear text model"; "model call 0.1-0.2 s" or "1.8 s"; "Task 1 and Task 3 are deterministic"; the Ofli comparison table; the leave-one-disaster-out table; the calibration table; the pipeline diagram; the statement that text helps only where labels agree (now true for fusion in both tasks). Keep slides on the audit, the Task 2 ladder, Task 3 and the image-scale/imbalance/architecture studies unless they quote the items above.

| Old slide (your numbering) | Status | What changes |
|---|---|---|
| 4 | update | Task 1 model: BERTweet, weighted F1 and macro-F1 (3-seed means), the switch rule |
| 8 | update | design choices / pipeline: BERTweet + temperature in the Task 1 box (`fig_c6_pipeline.png`) |
| 14, 15 | update | results tables / comparison with Ofli: Task 1 rows, indicative BERTweet agreed-subset row |
| 17 | update | latency and demo: measured numbers with what each measures (table `tab:latency`) |
| others (1-3, 5-7, 9-13, 16, 18) | unknown | apply the checklist above |

Proposed new slides:

1. **Task 1 model exploration** (`fig_c4_t1_models.png`): the best classical model reaches %s dev weighted F1 and BERTweet %s; BiLSTM ties with TF-IDF (%s [%s]); the switch rule fired (%s %s).
2. **The guide's core model and the backbone swap** (`fig_c5_t2_models.png`, `fig_c8_fusion_gain.png`): BERTweet + ResNet50 reaches %s test macro-F1 against %s for L4a; swapping only the image backbone to CLIP B/32 gives %s (not distinguishable from L4a); the deficit is consistent with ResNet50 (dev weighted F1 resolved, dev macro-F1 not).
3. **Each modality helps where it describes the label** (`fig_c3_agreed_disagreed.png`): Task 1 text %.2f / %.2f (agreed / disagreed) vs image %.2f / %.2f; Task 2 text %.2f / %.2f vs image %.2f / %.2f; fusion gains only where the labels agree.
4. **Error analysis** (`contact_T1.png`, `contact_T2.png`, `fig_c1_t1_confusion.png`, `fig_c2_t2_confusion.png`): %s of %s Task 2 errors are rows where the text and image labels disagree (Task 1: %s of %s); the rule-based ambiguity criterion fires %s and %s times.
""" % (s4(max(t1[k]["dev_weighted_f1"][0] for k in ("tfidf_lr", "tfidf_nb_bal", "tfidf_cnb", "tfidf_rf", "tfidf_svm"))), s4(bw["dev_weighted_f1"][0]), sg4(tie1["point_gain"]), ci4(tie1["ci95"]), sg4(sw["seed_avg_gain"]), ci4(sw["ci95"]),
       s4(T2m["t2_early_bertweet_resnet50"]["test_macro_f1"][0]), s4(T2m["t2_L4a"]["test_macro_f1"][0]), s4(T2m["t2_early_bertweet_clip_b32"]["test_macro_f1"][0]),
       c3["T1|bertweet|agreed"][0], c3["T1|bertweet|disagreed"][0], c3["T1|img_clip_l14|agreed"][0], c3["T1|img_clip_l14|disagreed"][0], c3["T2|bertweet_text_rule8|agreed"][0], c3["T2|bertweet_text_rule8|disagreed"][0],
       c3["T2|t2_img_clip_l14_inv|agreed"][0], c3["T2|t2_img_clip_l14_inv|disagreed"][0], er["T2"]["by_category"].get("text-image mismatch", 0), er["T2"]["n_errors"], er["T1"]["by_category"].get("text-image mismatch", 0), er["T1"]["n_errors"],
       er["T1"]["by_category"].get("label ambiguity", 0), er["T2"]["by_category"].get("label ambiguity", 0)))
(ROOT / "docs" / "report_dossier.md").write_text("\n".join(OUT) + "\n", encoding="utf-8")
print("docs/report_dossier.md written:", len("\n".join(OUT).splitlines()), "lines;", len(tables(TEX)), "tables parsed from report_v2/main.tex")
