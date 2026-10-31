"""Day 4 final round, Part D: build report_v2/main.tex from report/Capstone_report_Sem5.tex (the old report; never modified). Every edit is an exact-match replacement that must hit exactly once (assertion),
the numbers in the new text come from results/ablation/D4_*.json, and the new tables from d4_report_tables.py. Sentences that need your judgement carry \\todo{...} or a % CHECK comment.
Run: python ablation/d4_build_report_v2.py   (writes report_v2/main.tex and report_v2/day4_tables.tex)"""
import json
import re
from pathlib import Path

import d4_report_tables as R

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "ablation"
OUT = ROOT / "report_v2"
OUT.mkdir(exist_ok=True)
J = lambda n: json.load(open(RES / n))
OLD = (ROOT / "report" / "Capstone_report_Sem5.tex").read_text(encoding="utf8")
TEX = OLD
EDITS = []


def rep(old, new, count=1):
    global TEX
    n = TEX.count(old)
    assert n == count, f"edit #{len(EDITS) + 1}: expected {count} match(es), found {n}: {old[:90]!r}"
    TEX = TEX.replace(old, new); EDITS.append(old[:60])


f3, f4, sg, ci, pm = (lambda x: f"{x:.3f}"), (lambda x: f"{x:.4f}"), R.sg, R.ci, R.pm
t1 = {r["model"]: r for r in J("D4_T1_summary.json")}; sw = J("D4_switch_rule_T1.json"); wt = J("D4_T1_weighting_posthoc.json"); tie1 = J("D4_T1_bilstm_vs_tfidf_dev.json")
fu1 = J("D4_T1_fusion.json"); t2 = J("D4_T2_fusion.json"); t2t = J("D4_T2_text_summary.json"); ov = J("D4_cross_task_overlap.json"); dups = J("D4_T1_dups.json"); lat = J("D4_t1_latency.json")
ofl = J("D4_T1_ofli.json"); cal = J("D4_T1_calibration.json"); c3 = J("D4_fig_c3_data.json"); dep = json.load(open(RES / "D4" / "T1_bertweet_deploy_seed42.json"))["test"]; er = J("D4_error_analysis.json")
dep_dev = json.load(open(RES / "D4" / "T1_bertweet_deploy_seed42.json"))["dev"]; chk = J("D4_deploy_check.json") if (RES / "D4_deploy_check.json").exists() else J("D4_t1_deploy_check.json")
bw, lr = t1["bertweet"], t1["tfidf_lr"]
lodo = J("D4_T1_lodo.json") if (RES / "D4_T1_lodo.json").exists() else None
cm = t2["comparisons"]; ph = t2["posthoc"]; own = {r["model"]: r for r in J("D4_fusion_gain_both.json")["T2"]["rows"]}; own1 = {r["model"]: r for r in J("D4_fusion_gain_both.json")["T1"]["rows"]}
T2m = t2["models"]; ixn = ov["t3_deployed_index"]; w1 = wt["trade_off_means"]
tbv = {k: ov["by_tweet_id"][k] for k in ov["by_tweet_id"]}

# ---------------------------------------------------------------- preamble
rep("\\newcommand{\\pmse}[2]{#1\\,$\\pm$\\,#2}", "\\newcommand{\\pmse}[2]{#1\\,$\\pm$\\,#2}\n\\newcommand{\\todo}[1]{\\textcolor{red}{[TODO: #1]}}")

# ---------------------------------------------------------------- abstract + contributions
rep("that flags informativeness from the tweet text, assigns", "that flags informativeness from the tweet text with a fine-tuned BERTweet, assigns")
rep("On the official test splits the pipeline reaches macro-F1 of 0.751, 0.598 and 0.615 on the three tasks.",
    f"On the official test splits the pipeline reaches, as means over three seeds, a weighted F1 of {f3(bw['test_weighted_f1'][0])} and a macro-F1 of {f3(bw['test_macro_f1'][0])} on Task~1, and macro-F1 of 0.598 and 0.615 on Tasks~2 and~3.")
rep("Probe--retrieval agreement separates reliable damage predictions (accuracy 0.812) from those needing review (0.446).",
    f"At our guide's request we also compared the candidate models for Tasks~1 and~2 under rules fixed in advance: a fine-tuned BERTweet replaced TF-IDF with logistic regression for Task~1 (development weighted-F1 gain {sg(sw['seed_avg_gain'], 4)}, 95\\% interval {ci(sw['ci95'], 4)}), "
    "whereas the guide's early-fusion model (BERTweet with a frozen ResNet50) was not better than our Task~2 model, and in both tasks fusion helps only where the text and image labels agree. "
    "Probe--retrieval agreement separates reliable damage predictions (accuracy 0.812) from those needing review (0.446).")
rep("that text helps only where the two labels agree, and that a held-out earthquake", "that fusion helps only where the two labels agree (in both Task~1 and Task~2), and that a held-out earthquake")
rep("\\item an interactive triage tool that shows all three predictions, with temperature-calibrated confidence for the humanitarian category and the damage-severity probe.",
    "\\item a comparison of the candidate models of the guide's lists for Tasks~1 and~2 (classical, recurrent, BERTweet, image heads and three fusion designs) under rules committed before the runs, with a unified agreed/disagreed analysis, a fusion-gain analysis and a rule-based error analysis (Section~\\ref{sec:explore});\n"
    "\\item an interactive triage tool that shows all three predictions, with temperature-calibrated confidence for all three tasks.")

# ---------------------------------------------------------------- related work
rep("which helps a responder judge whether to trust it.",
    "which helps a responder judge whether to trust it. For text we additionally compare BERTweet~\\cite{nguyen2020}, a RoBERTa~\\cite{liu2019} model pre-trained on English tweets, with classical baselines (naive Bayes, random forests~\\cite{breiman2001} and linear SVMs as implemented in scikit-learn~\\cite{pedregosa2011}) and a bidirectional LSTM~\\cite{hochreiter1997,schuster1997} over GloVe vectors~\\cite{pennington2014}.")

# ---------------------------------------------------------------- III-C, III-D
rep("The informativeness and humanitarian splits are tweet-disjoint. All of these properties are asserted in the code, as is the resolution of every image path in every split.",
    f"The informativeness and humanitarian splits are tweet-disjoint within each task, but not across tasks: {tbv['T1_test']['T2_train']:,} of the 2,237 Task~1 test rows are in the Task~2 training split ({tbv['T1_test']['T2_dev']} in its development split, {tbv['T1_test']['T2_test']} in its test split), and {ixn['T1_overlap_by_tweet_id']['test']} are in the Task~3 index. "
    "A model for one task must therefore never use weights, logits or heads trained on another task's data. This is enforced by provenance fields (task, label column, hash of the training split, initialisation) stored with every run and checked by every script that consumes saved runs, and a retroactive audit found no violation. "
    "All of these properties are asserted in the code, as is the resolution of every image path in every split.")
tx, im = dups["text"], dups["image"]
dt_b = tx["rescored"]["bertweet"]["delta"]; di = im["rescored"]
mx_w = max(abs(v["delta"]["weighted_f1"]) for v in di.values()); mx_m = max(abs(v["delta"]["macro_f1"]) for v in di.values())
rep("so the duplicates do not inflate the reported results.",
    f"so the duplicates do not inflate the reported results. \\textbf{{Task~1 split.}} The same checks on the Task~1 split (test rows against its training rows) find {tx['n_exact']} exact and {tx['n_near_incl_exact'] - tx['n_exact']} further near-duplicate test tweets "
    f"(normalised text, TF-IDF cosine $\\geq 0.9$) and {im['n_exact_md5']} test images with a byte-identical file in the training split ({im['exact_same_tweet_id']} with the same tweet id), {im['n_near_incl_exact']} flagged in total with CLIP B/32 cosine $\\geq 0.98$. "
    f"Re-scoring without the flagged rows changes BERTweet's weighted F1 by {sg(dt_b['weighted_f1'], 4)} (text duplicates removed) and by {sg(di['bertweet']['delta']['weighted_f1'], 4)} (image duplicates removed), and no image head or fusion model by more than {mx_w:.4f} in weighted F1 or {mx_m:.4f} in macro-F1.")

# ---------------------------------------------------------------- Fig. 1 + IV-A, IV-D, IV-E, IV-F
rep("{TF-IDF (1--2 grams)\\\\+ logistic regression}", "{BERTweet (fine-tuned)\\\\+ temperature scaling}")
rep("\\caption{Three-stage triage pipeline over one tweet. All three tasks run on every request.", "\\caption{Three-stage triage pipeline over one tweet. Task~1 is a fine-tuned BERTweet with a development-fitted temperature. All three tasks run on every request.")
rep("Task~1 uses the cleaned tweet text only. A TF-IDF representation (50k features, unigrams and bigrams, sublinear term frequency, minimum document frequency 2) feeds a logistic-regression classifier with balanced class weights ($C=1$).",
    "Task~1 uses the tweet text only. A BERTweet~\\cite{nguyen2020} model (\\texttt{vinai/bertweet-base}, 135M parameters; its tokenizer normalises @mentions to \\texttt{@USER}, URLs to \\texttt{HTTPURL} and emoji to text) is fine-tuned on the raw tweet text (maximum length 128) with inverse-frequency weighted cross-entropy, "
    "AdamW~\\cite{loshchilov2019} (learning rate $2\\times10^{-5}$, weight decay 0.01), batch size 32, four epochs with 6\\% linear warm-up and linear decay, and mixed precision; the epoch with the best development weighted F1 is kept, and three seeds are trained. "
    f"The displayed probability is temperature-scaled ($T={chk['temperature']:.3f}$, fitted on the development logits of the deployed checkpoint). The deployed checkpoint is a retrain of seed 42 (Section~\\ref{{sec:protocol}}). "
    "The earlier representation (TF-IDF with 50k features, unigrams and bigrams, sublinear term frequency, minimum document frequency 2, and a balanced logistic regression, $C=1$) is the v2 baseline and remains available as a CPU fallback (\\texttt{CAPSTONE\\_T1=tfidf}). "
    "The change from TF-IDF to BERTweet followed a rule fixed in advance (Section~\\ref{sec:explore}).")
rep("The displayed probabilities of the Task~2 model and the Task~3 probe are temperature-scaled~\\cite{guo2017}", "The displayed probabilities of the Task~1 model, the Task~2 model and the Task~3 probe are temperature-scaled~\\cite{guo2017}")
rep("the seed-43 and seed-44 temperatures are fitted only for evaluation.", "the seed-43 and seed-44 temperatures are fitted only for evaluation; the Task~1 temperature is fitted on the development logits of the deployed BERTweet checkpoint.")
mc, ee, hr = lat["model_call"], lat["end_to_end"], lat["http_round_trip"]
rep("and a confidence for each task: temperature-calibrated for Task~2 and the Task~3 probe, and the raw logistic-regression probability for the Task~1 badge.", "and a temperature-calibrated confidence for each task.")
rep("A model call takes 0.1--0.2\\,s and an end-to-end request about 1.8\\,s on an RTX~4050 laptop GPU.",
    f"On an RTX~4050 laptop GPU (Table~\\ref{{tab:latency}}) the Task~1 model call takes {mc['tfidf_lr_cpu']['median_ms']:.2f}\\,ms with TF-IDF and {mc['bertweet_cuda']['median_ms']:.1f}\\,ms with BERTweet, the whole cascade in-process (without upload or file transfer) {ee['tfidf_cuda']['median_ms']:.1f} and {ee['bertweet_cuda']['median_ms']:.1f}\\,ms, "
    f"and a full round trip through the running Gradio server, including the image upload and the five retrieved images, {hr['tfidf_gpu_server']['median_ms']:.0f} and {hr['bertweet_gpu_server']['median_ms']:.0f}\\,ms (medians).\n\n" + R.latency_table() + "\n")
rep("\\item \\emph{A linear text model for Task~1.} TF-IDF with logistic regression lands within about 1.5 points of a published text CNN (Table~\\ref{tab:ofli}), and balanced class weights raise its macro-F1 from 0.720 to 0.751 with a paired interval excluding zero.",
    f"\\item \\emph{{BERTweet for Task~1.}} On development data a fine-tuned BERTweet beats TF-IDF with logistic regression by {sg(sw['seed_avg_gain'], 4)} weighted F1 (paired 95\\% interval {ci(sw['ci95'], 4)}, positive at every seed), the condition fixed in advance for replacing it; a bidirectional LSTM and the classical models are not better (Section~\\ref{{sec:explore}}). "
    f"TF-IDF with logistic regression, the v2 model, is within about 1.5 points of a published text CNN (Table~\\ref{{tab:ofli}}), and balanced class weights had raised its macro-F1 from 0.720 to 0.751 with a paired interval excluding zero.\n"
    f"\\item \\emph{{Task~1 stays text-only.}} Late and early fusion with image features gain at most {max(abs(v['vs_best_unimodal']['dev_weighted_f1'][0]) for v in own1.values()):.4f} weighted F1 on development data and are not distinguishable from BERTweet alone, so the simpler model is kept (Section~\\ref{{sec:explore}}).\n"
    f"\\item \\emph{{L4a is kept over the guide's core model.}} BERTweet with a frozen ResNet50 is not distinguishable on development macro-F1 from L4a or from the image-only CLIP B/32 head, so the rule fixed in advance picks the model with fewer trainable parameters (Section~\\ref{{sec:explore}}).")

# ---------------------------------------------------------------- V-A, V-B, V-C
rep("and are trained and evaluated on a single RTX~4050 laptop GPU.", "and are trained and evaluated on a single RTX~4050 laptop GPU. Task~1 additionally uses BERTweet fine-tuning, a BiLSTM over GloVe vectors and frozen image-feature heads (Section~\\ref{sec:explore}).")
rep("Macro-F1 is the headline metric because the classes are imbalanced:", "Macro-F1 is the headline metric for Tasks~2 and~3 because the classes are imbalanced (for Task~1, whose largest-to-smallest ratio is only 2.4:1, the headline is the support-weighted F1 that the guide's sheet and Ofli et al.\\ report, and macro-F1 is always given beside it):")
rep("per-class recall, and the top-label expected calibration error (ECE, 15 equal-width bins).", "per-class precision and recall, the top-label expected calibration error (ECE, 15 equal-width bins), top-2 accuracy (Task~2), confusion matrices, the fusion gain (Section~\\ref{sec:explore}) and an error analysis.")
rep("Task~2 configurations are trained with seeds 42, 43 and 44 and reported as mean\\,$\\pm$\\,sample standard deviation; Task~1 and Task~3 involve no stochastic training and are single deterministic runs.",
    f"Every trained neural model (the Task~2 configurations and, for Task~1, BERTweet, the BiLSTM and the image and fusion heads) is trained with seeds 42, 43 and 44 and reported as mean\\,$\\pm$\\,sample standard deviation; the TF-IDF models, the majority baseline and Task~3 are deterministic single runs. "
    f"GPU training is not bit-for-bit repeatable: retraining seed 42 of the Task~1 model gave a development weighted F1 of {f4(dep_dev['weighted_f1'])} against {f4(chk['dev_wf1']['original'])} and agreed with the original run on {100 * chk['prediction_agreement']['test']:.1f}\\% of the test predictions. "
    "The deployed Task~1 checkpoint is therefore a retrain of seed 42 (the original runs saved no weights), its numbers are quoted separately, and the architecture claims use the three-seed means.")
rep("For the class-imbalance and architecture studies (Sections~\\ref{sec:imbalance}--\\ref{sec:arch}),", "For the class-imbalance and architecture studies (Sections~\\ref{sec:imbalance}--\\ref{sec:arch}) and the model exploration (Section~\\ref{sec:explore}),")
rep("and breaks ties in favour of the simplest model;", "and breaks ties in favour of the simplest model (the fewest trainable parameters);")
rep("the final check covers several hundred numbers with no mismatches.", "the final check covers more than two thousand numbers with no mismatches. % CHECK: update with the final count from verify_findings.py\n\\todo{Numbers carried over unchanged from the earlier report were not re-derived by the script; see the dossier.}")

# ---------------------------------------------------------------- VI: Table I, Ofli
rep("T1 Informativeness & 1 & 0.794 & 0.751 & 0.797 \\\\",
    f"T1 BERTweet, 3-seed mean & 3 & {pm(bw['test_acc'])} & {pm(bw['test_macro_f1'])} & {pm(bw['test_weighted_f1'])} \\\\\nT1 BERTweet, deployed & 1 & {f3(dep['acc'])} & {f3(dep['macro_f1'])} & {f3(dep['weighted_f1'])} \\\\\nT1 TF-IDF + LR (v2) & 1 & 0.794 & 0.751 & 0.797 \\\\")
rep("Task~1 reaches 0.751 macro-F1; the gap to its 0.794 accuracy reflects the harder minority \\emph{not informative} class.",
    f"Task~1 reaches a weighted F1 of {f3(bw['test_weighted_f1'][0])} and a macro-F1 of {f3(bw['test_macro_f1'][0])} as three-seed means with BERTweet (deployed checkpoint {f3(dep['weighted_f1'])} and {f3(dep['macro_f1'])}), against 0.797 and 0.751 for the TF-IDF model of v2; the gap between macro-F1 and accuracy reflects the harder minority \\emph{{not informative}} class.")
oa, ob = ofl["bertweet"], ofl["tfidf_lr"]
rep("A simple TF-IDF model lands within about 1.5 points of their text CNN.",
    f"TF-IDF with logistic regression lands within about 1.5 points of their text CNN. BERTweet, scored on the {oa['n']:,} test rows of our splits whose text and image labels agree, reaches accuracy {f3(oa['acc'][0])} and weighted F1 {f3(oa['weighted_f1'][0])}, above their multimodal numbers (0.844 and 0.842); "
    "this subset is the agreed part of \\texttt{datasplit\\_all}, not their agreed-label split, so the row is indicative and we claim no improvement.")
rep(" & Ours, TF-IDF + LR & full & 2 & 0.794 & 0.797 \\\\", f" & Ours, TF-IDF + LR & full & 2 & 0.794 & 0.797 \\\\\n & Ours, BERTweet (3-seed mean) & full & 2 & {f3(bw['test_acc'][0])} & {f3(bw['test_weighted_f1'][0])} \\\\\n & Ours, BERTweet (3-seed mean) & agreed$^{{\\dagger}}$ & 2 & {f3(oa['acc'][0])} & {f3(oa['weighted_f1'][0])} \\\\")
rep("\\caption{Indicative comparison with Ofli et al.~\\cite{ofli2020} (weighted F1).}", f"\\caption{{Indicative comparison with Ofli et al.~\\cite{{ofli2020}} (weighted F1). $^{{\\dagger}}$ the {oa['n']:,} test rows of \\texttt{{datasplit\\_all}} on which our text and image labels agree, not Ofli et al.'s agreed-label split, so the comparison remains indicative.}}")

# ---------------------------------------------------------------- VII-D calibration
c = {(r["run"]): r for r in cal}
rows = "".join(f"{lab} & {c[k]['test']['acc']:.3f} & {c[k]['test']['mean_conf']:.3f} & {c[k]['test']['ece']:.3f} & {c[k]['T']:.3f} & {c[k]['test']['ece_after']:.3f} \\\\\n" for lab, k in
               (("T1, seed 42", "bertweet_seed42"), ("T1, seed 43", "bertweet_seed43"), ("T1, seed 44", "bertweet_seed44"), ("T1, deployed", "bertweet_deploy_seed42")))
rep("T3 probe & 0.684 & 0.607 & 0.077 & 0.693 & 0.038 \\\\", "T3 probe & 0.684 & 0.607 & 0.077 & 0.693 & 0.038 \\\\\n\\midrule\n" + rows.rstrip("\n"))
cd = c["bertweet_deploy_seed42"]
rep("The Task~2 model is over-confident and the Task~3 probe under-confident; temperature scaling brings both below 0.04.",
    f"The Task~2 model is over-confident and the Task~3 probe under-confident; temperature scaling brings both below 0.04. The Task~1 BERTweet model is over-confident too (deployed checkpoint: test mean confidence {cd['test']['mean_conf']:.3f} against accuracy {cd['test']['acc']:.3f}, ECE {cd['test']['ece']:.3f}); "
    f"scaling with $T={cd['T']:.3f}$ lowers its ECE to {cd['test']['ece_after']:.3f} (the three seeds reach {min(c[k]['test']['ece_after'] for k in ('bertweet_seed42', 'bertweet_seed43', 'bertweet_seed44')):.3f}--{max(c[k]['test']['ece_after'] for k in ('bertweet_seed42', 'bertweet_seed43', 'bertweet_seed44')):.3f}), "
    "which is above the 0.04 reached for Tasks~2 and~3.")

# ---------------------------------------------------------------- VII-F: Task 1 weighting
rep("Inverse-frequency weighting is therefore retained.",
    f"Inverse-frequency weighting is therefore retained. \\textbf{{Task~1 (post-hoc).}} The same question for the Task~1 BERTweet model (three seeds, development decides) shows that weighting raises the dev balanced accuracy by {sg(wt['point_gain'], 4)} (paired 95\\% interval {ci(wt['ci95'], 4)}) by raising the recall of \\emph{{not informative}} "
    f"({w1['dev']['ni_recall']['weighted']:.3f} against {w1['dev']['ni_recall']['unweighted']:.3f} on development data) at the cost of the recall of \\emph{{informative}} ({w1['dev']['inf_recall']['weighted']:.3f} against {w1['dev']['inf_recall']['unweighted']:.3f}) and of the precision on the minority class (Table~\\ref{{tab:weight1}}). "
    "Weighted F1 and macro-F1 were not tested with a bootstrap; their means differ by less than the seed spread. This is a different operating point, not a safety improvement; whether it is preferable depends on the cost of a missed informative tweet.\n\n" + R.weight_table() + "\n")

# ---------------------------------------------------------------- VII-H established + new VII-I (inserted before Discussion)
est_d = f"BERTweet over TF-IDF with logistic regression for Task~1 on development data ({sg(sw['seed_avg_gain'], 4)} weighted F1, interval {ci(sw['ci95'], 4)}, positive at every seed; the lower bound is small); the effect of Task~1 loss weighting on the recall of the minority class; and, as a secondary result, the worse development weighted F1 of the guide's core model than L4a ({sg(cm['core_vs_L4a']['dev']['weighted_f1']['point'], 4)}, interval {ci(cm['core_vs_L4a']['dev']['weighted_f1']['ci95'], 4)})"
ne_d = (f"the BiLSTM against TF-IDF with logistic regression ({sg(tie1['point_gain'], 4)} {ci(tie1['ci95'], 4)}); fusion against text-only for Task~1 (best development gain {sg(fu1['bootstrap']['best_fused_vs_text_only']['dev']['weighted_f1']['point'], 4)}); the core model against L4a and against the image-only CLIP B/32 head on development macro-F1; "
        f"BERTweet against DistilBERT as the Task~2 text encoder ({sg(ph['bertweet_matched_vs_distilbert_text']['dev']['macro_f1']['point'], 4)}, interval {ci(ph['bertweet_matched_vs_distilbert_text']['dev']['macro_f1']['ci95'], 4)}); BERTweet with CLIP B/32 against L4a; the effect of the duplicates; and the agreed/disagreed differences, which are descriptive (no interval)")
rep("and the differences among Task~3 rungs L2 to L4, are not established.",
    "and the differences among Task~3 rungs L2 to L4, are not established.\n\n"
    f"\\textbf{{Day~4, established.}} {est_d}.\n\n\\textbf{{Day~4, not established.}} {ne_d}. All of these are exploratory (about twenty further comparisons, no multiplicity correction); development data decide and test is reported afterwards.")

L = lodo
EXPLORE = rf"""
\subsection{{Model Exploration for Tasks~1 and~2}}
\label{{sec:explore}}
At our guide's request we compared the candidate models of the two task lists (Task~1: TF-IDF with logistic regression, naive Bayes or a random forest, a BiLSTM with GloVe, BERTweet, image heads on ResNet50, DINOv2 and CLIP, late and early fusion; Task~2: the same families plus DistilBERT and the Task~2 fusion designs). The candidate lists, the selection rules and the comparisons below were committed before the runs
(\texttt{{SELECTION\_RULE\_DAY4.md}}); arms added after seeing earlier results are marked $^\ast$. Development data decide, the test split is scored once, ties go to the model with the fewest trainable parameters, every neural model has three seeds, and no multiplicity correction is applied to the roughly twenty added comparisons.
Provenance fields stored with every run (task, label column, hash of the training split, initialisation) guard against using weights trained on another task (Section~\ref{{sec:data_integrity}}).

\textbf{{Task~1.}} Table~\ref{{tab:t1explore}} and Fig.~\ref{{fig:t1models}} list the models. All image heads score against the text label, so they are far below the text models ({f3(t1['img_clip_l14']['test_weighted_f1'][0])} for the best head against {f3(bw['test_weighted_f1'][0])} for BERTweet; the majority class scores {f3(t1['majority']['test_weighted_f1'][0])}). The class-weighted classical text models (logistic regression, balanced naive Bayes, Complement NB, random forest, linear SVM) plateau between {f3(min(t1[k]['dev_weighted_f1'][0] for k in ('tfidf_lr', 'tfidf_nb_bal', 'tfidf_cnb', 'tfidf_rf', 'tfidf_svm')))} and {f3(max(t1[k]['dev_weighted_f1'][0] for k in ('tfidf_lr', 'tfidf_nb_bal', 'tfidf_cnb', 'tfidf_rf', 'tfidf_svm')))} development weighted F1; naive Bayes needs class weights
(macro-F1 {f3(t1['tfidf_nb']['test_macro_f1'][0])} unweighted against {f3(t1['tfidf_nb_bal']['test_macro_f1'][0])} balanced). The BiLSTM is tied with TF-IDF and logistic regression ({sg(tie1['point_gain'], 4)} {ci(tie1['ci95'], 4)} on development weighted F1; post-hoc). BERTweet is best on development data, and the rule fixed in advance (switch if the paired development interval of the seed-averaged gain excludes zero and the gain is positive at every seed) is met
({sg(sw['seed_avg_gain'], 4)} {ci(sw['ci95'], 4)}; per seed {', '.join(sg(x, 4) for x in sw['per_seed_point_gain'])}), so BERTweet replaced TF-IDF as the deployed Task~1 model; the lower bound is small, the test gap is larger ({f3(bw['test_weighted_f1'][0])} against {f3(lr['test_weighted_f1'][0])}). Latency is not a reason to keep TF-IDF (Table~\ref{{tab:latency}}).
Table~\ref{{tab:weight1}} shows what loss weighting does (Section~\ref{{sec:imbalance}}).

{R.t1_table()}

\begin{{figure}}[t]
\centering
\includegraphics[width=\columnwidth]{{figures/fig_c4_t1_models.pdf}}
\caption{{Task~1 models by development weighted F1 (hollow: development, filled: test; whiskers: s.d.\ over seeds; dashed: majority class on the test split; $\bigstar$: the deployed model family).}}
\label{{fig:t1models}}
\end{{figure}}

\textbf{{Fusion for Task~1.}} Late fusion of BERTweet with the CLIP L/14 head, the guide's core model (BERTweet with frozen ResNet50 features) and a BERTweet and CLIP L/14 concatenation all gain {min(own1[k]['vs_best_unimodal']['dev_weighted_f1'][0] for k in own1):.4f} to {max(own1[k]['vs_best_unimodal']['dev_weighted_f1'][0] for k in own1):.4f} weighted F1 on development data over the better single model, positive at every seed but not distinguishable from BERTweet alone (best fused minus text-only
{sg(fu1['bootstrap']['best_fused_vs_text_only']['dev']['weighted_f1']['point'], 4)} {ci(fu1['bootstrap']['best_fused_vs_text_only']['dev']['weighted_f1']['ci95'], 4)}); the three fused models tie with each other on development data (a spread of 0.0003), so none is best, and the tie goes to the text-only model. On the test split the intervals exclude zero narrowly (Table~\ref{{tab:prereg}}); test does not decide.
Late fusion has a lower test ROC-AUC than BERTweet alone ({f3(J('D4_T1_summary.json')[[r['model'] for r in J('D4_T1_summary.json')].index('late_bertweet+img_clip_l14')]['test_roc_auc'][0])} against {f3(bw['test_roc_auc'][0])}); equal-weight averaging with a weaker image model plausibly degrades the ranking (not tested).
\todo{{Decide whether the ROC-AUC explanation stays or is cut.}}

\textbf{{Task~2.}} Table~\ref{{tab:t2explore}} and Fig.~\ref{{fig:t2models}} list the Task~2 models. The classical text models reach {f3(T2m['t2_distilbert_text']['test_macro_f1'][0] if False else t2t['models']['t2_tfidf_lr']['test_macro_f1'][0])} test macro-F1 at best and DistilBERT {f3(T2m['t2_distilbert_text']['test_macro_f1'][0])}; BERTweet reaches {f3(T2m['bertweet_text_matched']['test_macro_f1'][0])} with the DistilBERT recipe (dev {f3(T2m['bertweet_text_matched']['dev_macro_f1'][0])} against {f3(T2m['t2_distilbert_text']['dev_macro_f1'][0])}), consistent at every seed on development data,
but the paired development interval marginally includes zero {ci(ph['bertweet_matched_vs_distilbert_text']['dev']['macro_f1']['ci95'], 4)}: small, not robust; the input differs (BERTweet reads the raw tweet, DistilBERT the cleaned text), which is a confound. Every text-only model is far below the image heads ({f3(T2m['t2_img_clip_b32']['test_macro_f1'][0])} for CLIP B/32 and {f3(T2m['t2_img_clip_l14_inv']['test_macro_f1'][0])} for L/14).

{R.t2_table()}

\begin{{figure}}[t]
\centering
\includegraphics[width=\columnwidth]{{figures/fig_c5_t2_models.pdf}}
\caption{{Task~2 models by development macro-F1 (hollow: development, filled: test; whiskers: s.d.\ over seeds; dashed: majority class; $\bigstar$: the deployed model).}}
\label{{fig:t2models}}
\end{{figure}}

\textbf{{The guide's core model.}} BERTweet with frozen ResNet50 features reaches {f3(T2m['t2_early_bertweet_resnet50']['test_macro_f1'][0])} test macro-F1, below L4a ({f3(T2m['t2_L4a']['test_macro_f1'][0])}) and the image-only CLIP B/32 head ({f3(T2m['t2_img_clip_b32']['test_macro_f1'][0])}). On the headline development macro-F1 neither difference is distinguishable
({sg(cm['core_vs_L4a']['dev']['macro_f1']['point'], 4)} {ci(cm['core_vs_L4a']['dev']['macro_f1']['ci95'], 4)} against L4a; {sg(cm['core_vs_image_only_clip_b32']['dev']['macro_f1']['point'], 4)} {ci(cm['core_vs_image_only_clip_b32']['dev']['macro_f1']['ci95'], 4)} against the image-only head), so the tie-break picks the model with fewer trainable parameters
({R.esc(f"{J('D4_t2_param_counts.json')['t2_L4a']:,}")} for L4a, {J('D4_t2_param_counts.json')['t2_img_clip_b32_head']:,} for the head, {J('D4_t2_param_counts.json')['core_bertweet_resnet50']:,} for the core model); development weighted F1, a secondary metric, is worse for the core model in both comparisons and is corroboration only. The guide's architecture is therefore not adopted.
A post-hoc arm changes only the frozen image backbone: BERTweet with frozen CLIP B/32 features (same text encoder, fusion head structure and recipe; the input projection of the image features changes width) reaches {f3(T2m['t2_early_bertweet_clip_b32']['test_macro_f1'][0])} test macro-F1. Swapping only the image backbone raises the guide's architecture to a level not distinguishable from L4a's
({sg(ph['posthoc_clip_b32_vs_L4a']['dev']['macro_f1']['point'], 4)} {ci(ph['posthoc_clip_b32_vs_L4a']['dev']['macro_f1']['ci95'], 4)} on development macro-F1); the deficit is consistent with ResNet50, resolved on development weighted F1 ({sg(ph['posthoc_clip_b32_vs_core']['dev']['weighted_f1']['point'], 4)} {ci(ph['posthoc_clip_b32_vs_core']['dev']['weighted_f1']['ci95'], 4)}) but not on development macro-F1
({sg(ph['posthoc_clip_b32_vs_core']['dev']['macro_f1']['point'], 4)} {ci(ph['posthoc_clip_b32_vs_core']['dev']['macro_f1']['ci95'], 4)}). The arm swaps the backbone against the core model but also changes text encoder and fusion head against L4a, so it is not an encoder-only test.

{R.prereg_table()}

\textbf{{Fusion gain.}} Table~\ref{{tab:fgain}} and Fig.~\ref{{fig:fgain}} give the fusion gain over the better of the models a fused model is built from, and over the best unimodal model. In Task~2 no fused model beats the best image head on development macro-F1 (late fusion {sg(own['late_bertweet_text_rule8+t2_img_clip_l14_inv']['vs_best_unimodal']['dev_macro_f1'][0], 3)}); against its own components the core model gains {sg(own['t2_early_bertweet_resnet50']['vs_own']['dev_macro_f1'][0], 3)} (its ResNet50 head is weak)
and L4a {sg(own['t2_L4a']['vs_own']['dev_macro_f1'][0], 3)} on development data but {sg(own['t2_L4a']['vs_own']['test_macro_f1'][0], 3)} on test.

{R.fgain_table()}

\begin{{figure*}}[t]
\centering
\includegraphics[width=0.92\textwidth]{{figures/fig_c8_fusion_gain.pdf}}
\caption{{Fusion gain over own components (fused minus the better of its two single-modality models; development: hatched, test: solid; whiskers: s.d.\ over three seeds).}}
\label{{fig:fgain}}
\end{{figure*}}

\textbf{{Where each modality helps.}} Fig.~\ref{{fig:agree}} splits both test sets by whether the text and image labels agree. In Task~1 (scored against the text label) the text model keeps its score on disagreed rows (weighted F1 {c3['T1|bertweet|disagreed'][0]:.2f} against {c3['T1|bertweet|agreed'][0]:.2f} on agreed rows) while the image head falls from {c3['T1|img_clip_l14|agreed'][0]:.2f} to {c3['T1|img_clip_l14|disagreed'][0]:.2f}; in Task~2 (scored against the image label) the image head keeps its score
(macro-F1 over the classes present {c3['T2|t2_img_clip_l14_inv|disagreed'][0]:.2f} against {c3['T2|t2_img_clip_l14_inv|agreed'][0]:.2f}) while the text model falls from {c3['T2|bertweet_text_rule8|agreed'][0]:.2f} to {c3['T2|bertweet_text_rule8|disagreed'][0]:.2f}. Fusion gains only where the labels agree, and loses where they disagree (Section~\ref{{sec:discussion_modality}}). These subset numbers are descriptive (no interval).

\begin{{figure*}}[t]
\centering
\includegraphics[width=0.92\textwidth]{{figures/fig_c3_agreed_disagreed.pdf}}
\caption{{Each modality helps where it describes the label being scored (test set; bars: mean over three seeds, whiskers: s.d.; the deployed Task~1 model is a single checkpoint; Task~2 macro-F1 averages the classes present in the subset).}}
\label{{fig:agree}}
\end{{figure*}}

\begin{{figure}}[t]
\centering
\includegraphics[width=0.95\columnwidth]{{figures/fig_c1_t1_confusion.pdf}}\\[2pt]
\includegraphics[width=0.85\columnwidth]{{figures/fig_c2_t2_confusion.pdf}}
\caption{{Confusion matrices on the test set: Task~1 (deployed BERTweet; counts and row-normalised) and Task~2 (L4a seed 42; row-normalised, with the number of test rows per class).}}
\label{{fig:conf12}}
\end{{figure}}

\textbf{{Error analysis.}} For the deployed models (Task~1 BERTweet, Task~2 L4a seed 42) we took 12 misclassified test examples per task by a rule fixed before looking at any example: the (true, predicted) pairs are ranked by their error count, and the first rows by index are taken from each pair in turn, at most three per pair (six for Task~1, which has only two pairs). Each error is assigned by written criteria to
\emph{{text--image mismatch}} (the text label differs from the image label), \emph{{label ambiguity}} (the predicted class is the true class's most-confused partner and the example is a near-duplicate of a training example with a conflicting label, or has at most five words), or \emph{{model error}}. Table~\ref{{tab:errcounts}} gives the counts over \emph{{all}} errors: {er['T2']['by_category'].get('text-image mismatch', 0)} of the {er['T2']['n_errors']} Task~2 errors
({100 * er['T2']['by_category'].get('text-image mismatch', 0) / er['T2']['n_errors']:.0f}\%) are rows on which the text and image labels disagree, against {er['T1']['by_category'].get('text-image mismatch', 0)} of {er['T1']['n_errors']} for Task~1; the ambiguity criterion fires rarely ({er['T1']['by_category'].get('label ambiguity', 0)} and {er['T2']['by_category'].get('label ambiguity', 0)} errors), so the sample, which follows the rank order of the confusion pairs, contains {er['T1']['sample_categories'].get('label ambiguity', 0)} and {er['T2']['sample_categories'].get('label ambiguity', 0)} of them.
Table~\ref{{tab:errex}} shows four examples per task; the contact sheets with the images are in the supplementary material.

{R.err_tables()}

\textbf{{Justification of the design choices.}} Table~\ref{{tab:justify}} summarises, for each architectural choice, the alternatives tried, the development result, the verdict and the basis.

{R.justify_table()}
"""
rep("\n\\section{Discussion}\n", "\n" + EXPLORE + "\n\\section{Discussion}\n")

# ---------------------------------------------------------------- VIII Discussion + limitations, IX conclusion
rep("\\textbf{Retrieval is better as evidence than as a vote.}",
    "\\textbf{Each modality helps where it describes the label being scored.}\\label{sec:discussion_modality} The same pattern holds in both tasks (Fig.~\\ref{fig:agree}). Task~1 is scored against the text label: fusion with image features gains weighted F1 on rows where the text and image labels agree and loses on rows where they disagree "
    f"(late fusion: {sg(J('D4_T1_fusion.json')['subsets']['test']['models']['late_bertweet+img_clip_l14']['agree']['weighted_f1'] - J('D4_T1_fusion.json')['subsets']['test']['models']['bertweet']['agree']['weighted_f1'], 3)} against BERTweet on agreed rows, "
    f"{sg(J('D4_T1_fusion.json')['subsets']['test']['models']['late_bertweet+img_clip_l14']['disagree']['weighted_f1'] - J('D4_T1_fusion.json')['subsets']['test']['models']['bertweet']['disagree']['weighted_f1'], 3)} on disagreed rows). Task~2 is scored against the image label and shows the mirror image: the text model collapses on disagreed rows, and fusion helps only on agreed rows. "
    "The subset results suggest that the label source, not the architecture, decides which modality carries the task; we did not train or score Task~2 on text labels or Task~1 on image labels, so this is an interpretation. \\todo{Soften or cut: it rests on the descriptive agreed/disagreed split only.}\n\n\\textbf{Retrieval is better as evidence than as a vote.}")
rep("Evaluating on text labels, or on the agreed subset as Ofli et al.\\ did, would favour fusion.",
    "Evaluating on text labels, or on the agreed subset as Ofli et al.\\ did, would favour fusion. \\todo{Partly contradicted by Task~1: on the full text-labelled Task~1, fusion is not distinguishable from text-only on development data (Section~\\ref{sec:explore}); fusion gains appear on the agreed rows only. Rewrite.}")
rep("Three seeds resolve differences of about 0.01 at best, Task~1 and Task~3 results are single deterministic runs, and no multiplicity correction was applied.",
    "Three seeds resolve differences of about 0.01 at best, the Task~3 results and the classical Task~1 models are single deterministic runs, GPU training is not bit-for-bit repeatable (the deployed Task~1 checkpoint is a retrain of seed 42), the Task~1 leave-one-disaster-out column is a single seed, and about twenty further comparisons were made without multiplicity correction. "
    "The Task~1 and Task~2 splits overlap by tweet, which is safe only because no Task~1 model uses anything trained on Task~2 (asserted). The optional candidates of the guide's lists (DeBERTa-v3, ModernBERT, SigLIP, EfficientNet-B0, a zero-shot vision-language model, fine-tuning ResNet50 and a gated fusion for Task~1) were not run. "
    "Error-analysis categories are assigned by written criteria, not by human inspection.")
rep("We presented a three-task triage pipeline for crisis tweets and evaluated it rigorously against our earlier version and against alternative designs.",
    "We presented a three-task triage pipeline for crisis tweets and evaluated it rigorously against our earlier version and against alternative designs, including the candidate models of our guide's lists: a fine-tuned BERTweet now serves Task~1, chosen by a rule fixed in advance, while the guide's early-fusion model did not beat our Task~2 model.")
rep("\\item \\textbf{Agreed-label training and evaluation}, to compare directly with prior work.",
    "\\item \\textbf{Agreed-label training and evaluation}, to compare directly with prior work.\n\\item \\textbf{The optional candidates not run here}: DeBERTa-v3 or ModernBERT, SigLIP, a zero-shot vision-language model, fine-tuning ResNet50, and a gated fusion for Task~1. \\todo{Decide which of the earlier items are now done and cut them.}")

# ---------------------------------------------------------------- label for the data-integrity cross-reference and the bibliography
rep("\\subsection{Tweet-Level Integrity}", "\\subsection{Tweet-Level Integrity}\n\\label{sec:data_integrity}")
BIB = """\\bibitem{nguyen2020} D. Q. Nguyen, T. Vu, and A. Tuan Nguyen, ``BERTweet: A pre-trained language model for English Tweets,'' in \\textit{Proc. 2020 Conf. Empirical Methods in Natural Language Processing: System Demonstrations (EMNLP)}, 2020, pp. 9--14.
\\bibitem{liu2019} Y. Liu \\textit{et al.}, ``RoBERTa: A robustly optimized BERT pretraining approach,'' arXiv:1907.11692, 2019.
\\bibitem{pennington2014} J. Pennington, R. Socher, and C. D. Manning, ``GloVe: Global vectors for word representation,'' in \\textit{Proc. 2014 Conf. Empirical Methods in Natural Language Processing (EMNLP)}, 2014, pp. 1532--1543.
\\bibitem{hochreiter1997} S. Hochreiter and J. Schmidhuber, ``Long short-term memory,'' \\textit{Neural Computation}, vol. 9, no. 8, pp. 1735--1780, 1997.
\\bibitem{schuster1997} M. Schuster and K. K. Paliwal, ``Bidirectional recurrent neural networks,'' \\textit{IEEE Trans. Signal Processing}, vol. 45, no. 11, pp. 2673--2681, 1997.
\\bibitem{breiman2001} L. Breiman, ``Random forests,'' \\textit{Machine Learning}, vol. 45, no. 1, pp. 5--32, 2001.
\\bibitem{pedregosa2011} F. Pedregosa \\textit{et al.}, ``Scikit-learn: Machine learning in Python,'' \\textit{J. Machine Learning Research}, vol. 12, pp. 2825--2830, 2011. % CHECK: start page 2825 confirmed on jmlr.org; end page from memory
"""
rep("\\end{thebibliography}", BIB + "\\end{thebibliography}")

# ---------------------------------------------------------------- LODO (needs results/ablation/D4_T1_lodo.json)
if L:
    EVN = {"California wildfires": "california_wildfires", "Hurricane Harvey": "hurricane_harvey", "Hurricane Irma": "hurricane_irma", "Hurricane Maria": "hurricane_maria", "Iraq--Iran earthquake": "iraq_iran_earthquake", "Mexico earthquake": "mexico_earthquake", "Sri Lanka floods": "srilanka_floods"}
    byev = {r["event"]: r for r in L["rows"]}
    rep("Held-out event & T2 test rows & T3 test rows & T1 (text) & T2 (image, $n$=3) & T3 (5-NN) \\\\", "Held-out event & T2 test rows & T3 test rows & T1 TF-IDF & T1 BERTweet$^{\\dagger}$ & T2 (image, $n$=3) & T3 (5-NN) \\\\")
    for name, key in EVN.items():
        m = re.search(re.escape(name) + r" & (\d+) & (\d+) & (\d\.\d{3} / \d\.\d{3}) & ", TEX)
        assert m, name
        r = byev[key]; new = f"{name} & {m.group(1)} & {m.group(2)} & {m.group(3)} & {r['held_out']['macro_f1']:.3f} / {r['in_dist']['macro_f1']:.3f} & "
        TEX = TEX.replace(m.group(0), new, 1)
    avg_old = re.search(r"Average over events & 2,237 & 529 & (0\.680 / 0\.714) & ", TEX); assert avg_old
    TEX = TEX.replace(avg_old.group(0), f"Average over events & 2,237 & 529 & {avg_old.group(1)} & {L['average']['bertweet_held_out']:.3f} / {L['average']['bertweet_in_dist']:.3f} & ", 1)
    rep("\\begin{tabular}{lrrccc}", "\\begin{tabular}{lrrcccc}")
    rep("Tasks 1 and 3 are deterministic.}", "Task 3 and the Task 1 TF-IDF column are deterministic; $^{\\dagger}$ Task 1 BERTweet is a single seed (42) per fold, selected on the other events' development rows.}")
    d_t = L["average"]["tfidf_in_dist"] - L["average"]["tfidf_held_out"]; d_b = L["average"]["bertweet_in_dist"] - L["average"]["bertweet_held_out"]
    worst = min(L["rows"], key=lambda r: r["held_out"]["macro_f1"] - r["in_dist"]["macro_f1"])
    rep("Task~1 loses 0.034 on average, concentrated in the Sri Lanka floods (0.932 to 0.793), the only flood event.",
        f"Task~1 with TF-IDF loses 0.034 on average, concentrated in the Sri Lanka floods (0.932 to 0.793), the only flood event; the BERTweet model (one seed per fold) loses {d_b:.3f} on average, most on the {worst['event'].replace('_', ' ').replace('iraq iran', 'Iraq--Iran').capitalize() if not worst['event'].startswith('iraq') else 'Iraq--Iran earthquake'} "
        f"({worst['in_dist']['macro_f1']:.3f} to {worst['held_out']['macro_f1']:.3f}); with one seed per fold the per-event numbers are indicative only.")
    rep("Text and image fail on different unseen events: text on the flood, images on the earthquakes.",
        f"Text and image fail on different unseen events: text on the flood, images on the earthquakes. \\todo{{With BERTweet (single seed) the flood is no longer the text model's weak event (held out {byev['srilanka_floods']['held_out']['macro_f1']:.3f} against {byev['srilanka_floods']['in_dist']['macro_f1']:.3f} in distribution, {byev['srilanka_floods']['held_out']['n']} rows); "
        f"its largest losses are on the earthquakes. Rewrite this sentence.}}")
else:
    print("WARNING: D4_T1_lodo.json not found; the LODO table keeps the TF-IDF column only (rerun after the LODO job finishes)")

(OUT / "main.tex").write_text(TEX, encoding="utf8")
print(f"report_v2/main.tex written: {len(EDITS)} edits applied; {len(TEX.splitlines())} lines (old report {len(OLD.splitlines())})")
