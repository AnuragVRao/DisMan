"""Day 4 final round, Part D helper: LaTeX tables and number snippets for report_v2, generated from results/ablation/D4_*.json (the same files the dossier and the verifier use).
Nothing here is typed by hand: every number comes from a JSON file; formatting is 3 decimals (report convention), the dossier gives 4."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "ablation"
J = lambda n: json.load(open(RES / n))


def esc(s):
    return (str(s).replace("\\", " ").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#").replace("±", r"$\pm$").replace("−", "$-$").replace("→", r"$\rightarrow$")
            .replace("≥", r"$\geq$").replace("≤", r"$\leq$").replace("×", r"$\times$"))


REFMAP = [("(§9.9, Table XIII)", "(Section~\\ref{sec:imbalance})"), ("§9.10 D1", "Section~\\ref{sec:arch}"), ("§9.10, D1", "Section~\\ref{sec:arch}"), ("§9.2, §9.3", "Section~\\ref{sec:component}"),
          ("§9.9", "Section~\\ref{sec:imbalance}"), ("§9.10", "Section~\\ref{sec:arch}"), ("§9.11", "Table~\\ref{tab:arch}"), ("§9.2", "Section~\\ref{sec:component}"), ("§9.3", "Section~\\ref{sec:component}"), ("§", ""),
          ("rule-8 recipes", "4-epoch recipes"), ("rule-8", "4-epoch"), ("(rule 2: fewest trainable parameters)", "(fewest trainable parameters)"), ("(simplicity, rule 2)", "(simplicity)"), ("dev rule (rule 3)", "dev rule"),
          ("(rule 5 of the imbalance selection rule)", "(imbalance selection rule)"), ("tie-break (rule 2", "tie-break (fewest trainable parameters"), ("rule 2", "the tie-break rule")]


def fixrefs(t):
    """The shared justification rows cite sections of the repository's findings document; in the report they point to the report's own sections."""
    for a, b in REFMAP:
        t = t.replace(a, b)
    return t


def n3(x):
    return f"{x:.3f}".replace("-", "$-$")


def sg(x, d=3):
    s = f"{x:+.{d}f}"
    return s.replace("-", "$-$")


def pm(v, d=3):
    """[mean, std] -> \\pmse{mean}{std}; a single run -> mean."""
    return rf"\pmse{{{v[0]:.{d}f}}}{{{v[1]:.{d}f}}}" if v[1] is not None else f"{v[0]:.{d}f}"


def ci(c, d=3):
    return "[" + sg(c[0], d) + ", " + sg(c[1], d) + "]"


# ---------------------------------------------------------------- Task 1 table
T1_LABEL = {"majority": "Majority class", "tfidf_lr": "TF-IDF + LR (v2)", "tfidf_nb": "NB, unweighted (reference)", "tfidf_nb_bal": "NB, balanced weights$^\\ast$", "tfidf_cnb": "Complement NB$^\\ast$",
            "tfidf_rf": "Random Forest", "tfidf_svm": "Linear SVM", "bilstm_glove": "BiLSTM + GloVe", "bertweet": "BERTweet, weighted CE", "bertweet_noweight": "BERTweet, unweighted CE$^\\ast$",
            "img_resnet50": "ResNet50 head (image)", "img_dinov2": "DINOv2 head (image)", "img_clip_l14": "CLIP L/14 head (image)", "late_bertweet+img_clip_l14": "Late fusion, BERTweet + CLIP L/14",
            "early_bertweet_resnet50": "Core model: BERTweet + ResNet50", "early_bertweet_clip_l14": "BERTweet + CLIP L/14 (concat)"}


def t1_table():
    rows = sorted(J("D4_T1_summary.json"), key=lambda r: -r["dev_weighted_f1"][0])
    o = ["\\begin{table*}[t]", "\\centering", "\\caption{Task 1 (informative or not) scored against the text label, sorted by development weighted F1. Mean\\,$\\pm$\\,s.d.\\ over three seeds where $n$=3; the TF-IDF models and the majority class are single deterministic runs. "
         "$^\\ast$ post-hoc arm, added after the first Day~4 results (Section~\\ref{sec:protocol}). The majority class always predicts \\emph{informative}.}", "\\label{tab:t1explore}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabular}{lcccccccc}", "\\toprule", "Model & $n$ & Dev W-F1 & Test acc. & Test W-F1 & Test macro-F1 & Bal.\\ acc. & MCC & ROC-AUC \\\\", "\\midrule"]
    for r in rows:
        o.append(f"{T1_LABEL[r['model']]} & {r['n_seeds']} & {pm(r['dev_weighted_f1'])} & {pm(r['test_acc'])} & {pm(r['test_weighted_f1'])} & {pm(r['test_macro_f1'])} & {pm(r['test_balanced_acc'])} & {pm(r['test_mcc'])} & {pm(r['test_roc_auc'])} \\\\")
    o += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(o)


# ---------------------------------------------------------------- Task 2 table
def t2_table():
    t2, tt = J("D4_T2_fusion.json")["models"], J("D4_T2_text_summary.json")["models"]
    items = [("Majority class (\\emph{not humanitarian})", tt["t2_majority"]), ("TF-IDF + LR, class-weighted", tt["t2_tfidf_lr"]), ("TF-IDF + LinearSVC, class-weighted", tt["t2_tfidf_svm"]),
             ("Complement NB, unweighted (reference)", tt["t2_tfidf_cnb"]), ("Complement NB, balanced weights$^\\ast$", tt["t2_tfidf_cnb_bal"]), ("DistilBERT, text only", t2["t2_distilbert_text"]),
             ("BERTweet, text only (DistilBERT-matched recipe)", t2["bertweet_text_matched"]), ("BERTweet, text only (4-epoch recipe)", t2["bertweet_text_rule8"]), ("Image head, CLIP B/32", t2["t2_img_clip_b32"]),
             ("Image head, CLIP L/14", t2["t2_img_clip_l14_inv"]), ("L4a: CLIP B/32 + DistilBERT cross-attention (deployed)", t2["t2_L4a"]), ("Late fusion, BERTweet + CLIP L/14 head", t2["late_bertweet_text_rule8+t2_img_clip_l14_inv"]),
             ("Core model: BERTweet + ResNet50 (frozen)", t2["t2_early_bertweet_resnet50"]), ("BERTweet + CLIP B/32 (frozen)$^\\ast$", t2["t2_early_bertweet_clip_b32"])]
    o = ["\\begin{table*}[t]", "\\centering", "\\caption{Task 2 (eight humanitarian classes) scored against the image label. Mean\\,$\\pm$\\,s.d.\\ over three seeds (the TF-IDF models and the majority class are single runs). "
         "$^\\ast$ post-hoc. The majority-class top-2 value is arbitrary (a constant predictor ties every other class) and is not meaningful.}", "\\label{tab:t2explore}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabular}{lccccccc}", "\\toprule", "Model & Dev macro-F1 & Test macro-F1 & Test W-F1 & Test acc. & Bal.\\ acc. & MCC & Top-2 \\\\", "\\midrule"]
    for lab, r in items:
        top2 = "--" if "Majority" in lab else pm(r["test_top2_acc"])
        o.append(f"{lab} & {pm(r['dev_macro_f1'])} & {pm(r['test_macro_f1'])} & {pm(r['test_weighted_f1'])} & {pm(r['test_acc'])} & {pm(r['test_balanced_acc'])} & {pm(r['test_mcc'])} & {top2} \\\\")
    o += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(o)


# ---------------------------------------------------------------- pre-registered comparisons
def prereg_table():
    sw, fu1, t2 = J("D4_switch_rule_T1.json"), J("D4_T1_fusion.json"), J("D4_T2_fusion.json")
    cmp_ = t2["comparisons"]; ph = t2["posthoc"]
    # Task 1 test-side numbers for the switch rule are reported as point differences of the 3-seed means
    t1 = {r["model"]: r for r in J("D4_T1_summary.json")}
    sw_test = t1["bertweet"]["test_weighted_f1"][0] - t1["tfidf_lr"]["test_weighted_f1"][0]
    f1d, f1t = fu1["bootstrap"]["best_fused_vs_text_only"]["dev"]["weighted_f1"], fu1["bootstrap"]["best_fused_vs_text_only"]["test"]["weighted_f1"]
    rows = [("Switch rule: BERTweet $-$ TF-IDF + LR (T1, W-F1)", f"{sg(sw['seed_avg_gain'])} {ci(sw['ci95'])}", "better (every seed)", f"{sg(sw_test)} (point)", "dev rule: switch"),
            ("Best fused $-$ BERTweet text-only (T1, W-F1)", f"{sg(f1d['point'])} {ci(f1d['ci95'])}", f1d["verdict"], f"{sg(f1t['point'])} {ci(f1t['ci95'])}", "tie $\\rightarrow$ simpler: text-only")]
    names = {"core_vs_L4a": "Core model $-$ L4a (T2, macro-F1)", "core_vs_image_only_clip_b32": "Core model $-$ image-only CLIP B/32 head (T2, macro-F1)"}
    for k, lab in names.items():
        d, t = cmp_[k]["dev"]["macro_f1"], cmp_[k]["test"]["macro_f1"]; tb = cmp_[k]["tie_break"]
        rows.append((lab, f"{sg(d['point'])} {ci(d['ci95'])}", d["verdict"], f"{sg(t['point'])} {ci(t['ci95'])}", f"tie $\\rightarrow$ fewer parameters: {'L4a' if k == 'core_vs_L4a' else 'image-only head'}"))
    for k, lab in (("posthoc_clip_b32_vs_core", "Post-hoc: BERTweet + CLIP B/32 $-$ core model (backbone swap only)"), ("posthoc_clip_b32_vs_L4a", "Post-hoc: BERTweet + CLIP B/32 $-$ L4a")):
        d, t = ph[k]["dev"]["macro_f1"], ph[k]["test"]["macro_f1"]
        rows.append((lab + " (T2, macro-F1)", f"{sg(d['point'])} {ci(d['ci95'])}", d["verdict"], f"{sg(t['point'])} {ci(t['ci95'])}", "post-hoc, descriptive"))
    d, t = ph["bertweet_matched_vs_distilbert_text"]["dev"]["macro_f1"], ph["bertweet_matched_vs_distilbert_text"]["test"]["macro_f1"]
    rows.append(("Post-hoc: BERTweet text $-$ DistilBERT text (T2, macro-F1)", f"{sg(d['point'])} {ci(d['ci95'])}", d["verdict"], f"{sg(t['point'])} {ci(t['ci95'])}", "post-hoc, descriptive"))
    o = ["\\begin{table*}[t]", "\\centering", "\\caption{Comparisons of the model exploration. The development difference decides; the test difference is reported afterwards and decides nothing. Paired bootstrap, seed-averaged, 10,000 resamples; ``not distinguishable'' means the 95\\% interval includes zero. "
         "Dev weighted F1 for the two core-model comparisons is worse for the core model (Section~\\ref{sec:explore}) and is corroboration only.}", "\\label{tab:prereg}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
         "\\begin{tabular}{p{5.6cm}p{3.3cm}p{2.3cm}p{3.2cm}p{3.0cm}}", "\\toprule", "Comparison & Dev difference [95\\% CI] & Dev verdict & Test difference (reporting only) & Basis \\\\", "\\midrule"]
    for r in rows:
        o.append(" & ".join(r) + " \\\\")
    o += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(o)


# ---------------------------------------------------------------- weighting trade-off
def weight_table():
    tm = J("D4_T1_weighting_posthoc.json")["trade_off_means"]; lab = {"ni_recall": "Not-informative recall", "inf_recall": "Informative recall", "ni_precision": "Not-informative precision"}
    o = ["\\begin{table}[t]", "\\centering", "\\caption{Task 1 loss weighting for BERTweet (post-hoc arm; mean of three seeds). Weighting trades precision for recall on the minority class; it is a different operating point, not a safety improvement.}", "\\label{tab:weight1}",
         "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{lcccc}", "\\toprule", " & \\multicolumn{2}{c}{Dev} & \\multicolumn{2}{c}{Test} \\\\", "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}", "Metric & weighted & unweighted & weighted & unweighted \\\\", "\\midrule"]
    for k, l in lab.items():
        o.append(f"{l} & {tm['dev'][k]['weighted']:.3f} & {tm['dev'][k]['unweighted']:.3f} & {tm['test'][k]['weighted']:.3f} & {tm['test'][k]['unweighted']:.3f} \\\\")
    o += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(o)


# ---------------------------------------------------------------- fusion gain (both tasks, both baselines)
def fgain_table():
    g = J("D4_fusion_gain_both.json")
    o = ["\\begin{table*}[t]", "\\centering", "\\caption{Fusion gain: the fused model minus the better (picked per seed on development data) of the single-modality models it is built from (``own components''), and minus the best unimodal model of the task "
         "(Task~1: BERTweet and the CLIP L/14 head; Task~2: BERTweet text-only and the CLIP L/14 head). Headline metric: weighted F1 (Task~1), macro-F1 (Task~2). Mean\\,$\\pm$\\,s.d.\\ over three seeds; dev / test.}", "\\label{tab:fgain}", "\\footnotesize",
         "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{llcc}", "\\toprule", "Task & Fused model (own components) & Gain vs.\\ own components & Gain vs.\\ best unimodal \\\\", "\\midrule"]
    short = {"late fusion: BERTweet + CLIP L/14 head": "late fusion (text + CLIP L/14 head)", "core model: BERTweet (fine-tuned) + ResNet50 (frozen)": "core model (text + ResNet50)", "BERTweet (fine-tuned) + CLIP L/14 (frozen)": "BERTweet + CLIP L/14 (text + CLIP L/14 head)",
             "post-hoc: BERTweet (fine-tuned) + CLIP B/32 (frozen)": "post-hoc BERTweet + CLIP B/32 (text + B/32 head)", "L4a: CLIP B/32 + DistilBERT cross-attention": "L4a (DistilBERT text + B/32 head)"}
    for t, h in (("T1", "weighted_f1"), ("T2", "macro_f1")):
        for r in g[t]["rows"]:
            f = lambda d: f"{sg(d['dev_' + h][0], 3)}\\,$\\pm$\\,{d['dev_' + h][1]:.3f} / {sg(d['test_' + h][0], 3)}\\,$\\pm$\\,{d['test_' + h][1]:.3f}"
            o.append(f"{t} & {short.get(r['label'], esc(r['label']))} & {f(r['vs_own'])} & {f(r['vs_best_unimodal'])} \\\\")
        if t == "T1":
            o.append("\\midrule")
    o += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(o)


# ---------------------------------------------------------------- latency
def latency_table():
    d = J("D4_t1_latency.json")
    rows = [("Task~1 model call, TF-IDF + LR (CPU)", d["model_call"]["tfidf_lr_cpu"]), ("Task~1 model call, BERTweet (GPU)", d["model_call"]["bertweet_cuda"]), ("Task~1 model call, BERTweet (CPU)", d["model_call"]["bertweet_cpu"]),
            ("Whole cascade in-process, TF-IDF backend (GPU)", d["end_to_end"]["tfidf_cuda"]), ("Whole cascade in-process, BERTweet backend (GPU)", d["end_to_end"]["bertweet_cuda"]),
            ("HTTP round trip, TF-IDF backend (GPU server)", d["http_round_trip"]["tfidf_gpu_server"]), ("HTTP round trip, BERTweet backend (GPU server)", d["http_round_trip"]["bertweet_gpu_server"])]
    o = ["\\begin{table}[t]", "\\centering", "\\caption{Latency on an RTX~4050 laptop GPU (median and 95th percentile in ms; 100 requests after five warm-ups, three fixed examples; the HTTP rows are 30 requests). ``Model call'' is text to probabilities only; ``whole cascade'' is the in-process handler "
         "(image features, Tasks~1--3 and retrieval) without upload or file transfer; ``HTTP round trip'' goes through the running Gradio server via \\texttt{gradio\\_client}, including the image upload and the five retrieved images.}", "\\label{tab:latency}",
         "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{lrr}", "\\toprule", "What is measured & Median & p95 \\\\", "\\midrule"]
    for lab, v in rows:
        o.append(f"{lab} & {v['median_ms']:.1f} & {v['p95_ms']:.1f} \\\\")
    o += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(o)


# ---------------------------------------------------------------- error analysis
def err_tables():
    er = J("D4_error_analysis.json"); cats = ("text-image mismatch", "label ambiguity", "model error")
    o = ["\\begin{table}[t]", "\\centering", "\\caption{Error analysis: category counts over \\emph{all} test errors (rule-based categories, Section~\\ref{sec:explore}) and in the 12-example sample per task.}", "\\label{tab:errcounts}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabular}{llcccc}", "\\toprule", "Task & Model & Errors & Mismatch & Ambiguity & Model error \\\\", "\\midrule"]
    for t, m in (("T1", "BERTweet (deployed)"), ("T2", "L4a seed 42")):
        e = er[t]; o.append(f"{t} & {m} & {e['n_errors']} / {e['n_test']:,} & {e['by_category'].get(cats[0], 0)} ({e['sample_categories'].get(cats[0], 0)}) & {e['by_category'].get(cats[1], 0)} ({e['sample_categories'].get(cats[1], 0)}) & {e['by_category'].get(cats[2], 0)} ({e['sample_categories'].get(cats[2], 0)}) \\\\")
    o += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    o += ["\\begin{table*}[t]", "\\centering", "\\caption{Examples of misclassified test tweets (the first four of the 12 per task, by the pre-set rule; all 24 are in the supplementary table and in the contact sheets). Confidence is the calibrated probability of the predicted class. Categories come from written criteria, not from reading each example.}",
          "\\label{tab:errex}", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{llp{6.2cm}p{2.9cm}p{3.4cm}}", "\\toprule", "\\# & Category & Tweet (truncated) & True & Predicted ($p$) \\\\", "\\midrule"]
    for t in ("T1", "T2"):
        for k, e in enumerate(er[t]["sample"][:4], 1):
            tw = e["tweet"].encode("ascii", "replace").decode().replace("?", "") if False else "".join(ch if ord(ch) < 128 else "?" for ch in e["tweet"])
            tw = tw if len(tw) <= 90 else tw[:87] + "..."
            o.append(f"{t}.{k} & {esc(e['cat'])} & {esc(tw)} & {esc(e['true'])} & {esc(e['pred'])} ({e['conf']:.2f}) \\\\")
        if t == "T1":
            o.append("\\midrule")
    o += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(o)


# ---------------------------------------------------------------- justification table
def justify_table():
    import d4_make_findings as MF
    t1, sw, wt, tie1 = J("D4_T1_summary.json"), J("D4_switch_rule_T1.json"), J("D4_T1_weighting_posthoc.json"), J("D4_T1_bilstm_vs_tfidf_dev.json")
    fu1, t2, t2t, pc = J("D4_T1_fusion.json"), J("D4_T2_fusion.json"), J("D4_T2_text_summary.json"), J("D4_t2_param_counts.json")
    rows = MF.jrows(t1, sw, wt, tie1, fu1, t2, t2t, pc)
    o = ["\\begin{table*}[t]", "\\centering", "\\caption{Justification of each architectural choice: the choice, the alternatives tried, the result on the development split, the verdict and the basis (a resolvable difference, or the pre-registered tie-break to the simpler model). "
         "Exploratory comparisons; no multiplicity correction. Numbers are development-split values unless stated.}", "\\label{tab:justify}", "\\tiny", "\\setlength{\\tabcolsep}{2.5pt}", "\\begin{tabular}{p{2.2cm}p{3.1cm}p{5.6cm}p{3.2cm}p{2.6cm}}",
         "\\toprule", "Choice & Alternatives tried & Result & Verdict & Basis \\\\", "\\midrule"]
    for _, c, a, r, v, b in rows:
        o.append(f"{fixrefs(esc(c))} & {fixrefs(esc(a))} & {fixrefs(esc(r))} & {fixrefs(esc(v))} & {fixrefs(esc(b))} \\\\\n\\midrule")
    o[-1] = o[-1].replace("\\midrule", "")
    o += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(o)
