# report_v2

Draft report generated from `report/Capstone_report_Sem5.tex` (never modified) by `ablation/d4_build_report_v2.py`. Prose is a draft: sentences that need your judgement carry `\todo{...}` (red in the PDF) or a `% CHECK` comment. All new numbers come from `results/ablation/D4_*.json`; the data behind every table and figure is in `docs/report_dossier.md`.

## Status: NOT COMPILED
No LaTeX engine is installed on this machine, and the Tectonic binary downloaded into `artifacts/tectonic/` was blocked from running by the sandbox. So the PDF does not exist yet. What was done instead: `ablation/d4_check_tex.py` (matched environments, brace balance, every `\ref` has a `\label`, every `\cite` has a `\bibitem`, figure files exist, no non-ASCII in new lines) reports 0 problems. Overfull boxes, undefined-reference warnings from a real run, and page count are unchecked.

## To compile (pick one)
```
cd report_v2
pdflatex main.tex; pdflatex main.tex          # needs a TeX distribution with IEEEtran, pgfplots, tikz (MiKTeX or TeX Live)
```
or, with the Tectonic binary already downloaded (run it yourself in the shell: `!` prefix in Claude Code):
```
cd report_v2
..\artifacts\tectonic\tectonic.exe -X compile main.tex
```
Afterwards: fix overfull boxes, check the log for `undefined`, and note the page count (the old report was about 9 pages; this draft has about 320 more lines, mostly tables).

## Files
- `main.tex` — the draft (single file, inline bibliography, as the old report).
- `figures/` — PDF for LaTeX and 300-dpi PNG for slides: `fig_c1_t1_confusion`, `fig_c2_t2_confusion`, `fig_c3_agreed_disagreed` (the headline), `fig_c4_t1_models`, `fig_c5_t2_models`, `fig_c6_pipeline` (matplotlib PNG for the slides; the report's Fig. 1 is the TikZ in `main.tex`), `fig_c8_fusion_gain`, and the contact sheets `contact_T1`, `contact_T2` (supplementary; not included in `main.tex`).
- Rebuild: `python ablation/d4_build_report_v2.py && python ablation/d4_check_tex.py` (Python 3.13 with the CUDA environment is not required for these two).
