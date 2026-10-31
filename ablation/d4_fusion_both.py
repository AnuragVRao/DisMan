"""Day 4 Part A2: one fusion-gain table for BOTH tasks with both baselines, per seed, mean ± std (results/ablation/D4_fusion_gain_both.{json,md}).
  vs best unimodal : fused minus the better (per seed, on DEV headline metric) of {text-only model, best image head} = the pair used in D4_T1_fusion / D4_T2_fusion.
  vs own components: fused minus the better (per seed, on dev headline) of the two single-modality models the fused model is built from.
Headline metric: Task 1 weighted F1, Task 2 macro-F1 (both are also reported). Dev picks the baseline per seed; test is reported afterwards."""
import json

import numpy as np
from sklearn.metrics import f1_score

import metrics_all as M
from common import RES

SEEDS = (42, 43, 44)
T1 = dict(task="T1", head="weighted_f1", other="macro_f1", uni=("bertweet", "img_clip_l14"),
          models=[("late_bertweet+img_clip_l14", "late fusion: BERTweet + CLIP L/14 head", ("bertweet", "img_clip_l14")),
                  ("early_bertweet_resnet50", "core model: BERTweet (fine-tuned) + ResNet50 (frozen)", ("bertweet", "img_resnet50")),
                  ("early_bertweet_clip_l14", "BERTweet (fine-tuned) + CLIP L/14 (frozen)", ("bertweet", "img_clip_l14"))])
T2 = dict(task="T2", head="macro_f1", other="weighted_f1", uni=("bertweet_text_rule8", "t2_img_clip_l14_inv"),
          models=[("late_bertweet_text_rule8+t2_img_clip_l14_inv", "late fusion: BERTweet + CLIP L/14 head", ("bertweet_text_rule8", "t2_img_clip_l14_inv")),
                  ("t2_early_bertweet_resnet50", "core model: BERTweet (fine-tuned) + ResNet50 (frozen)", ("bertweet_text_matched", "H:d1_resnet50")),
                  ("t2_early_bertweet_clip_b32", "post-hoc: BERTweet (fine-tuned) + CLIP B/32 (frozen)", ("bertweet_text_matched", "t2_img_clip_b32")),
                  ("t2_L4a", "L4a: CLIP B/32 + DistilBERT cross-attention", ("t2_distilbert_text", "t2_img_clip_b32"))])


def stored(task, name, seed):
    """-> {'dev': {metric: v}, 'test': {...}} from the run JSON, or (for 'H:<arm>') from the stored label-only Day 2 run."""
    if name.startswith("H:"):
        arm = name[2:]; r = json.load(open(RES / "H" / f"{arm}_seed{seed}.json")); y = M.load_run("T2", "t2_L4a", 42)
        o = {}
        for sp in ("dev", "test"):
            pr = np.array(r[f"pred_{sp}"]); yy = y[f"y_{sp}"]
            o[sp] = dict(macro_f1=float(f1_score(yy, pr, labels=list(range(8)), average="macro", zero_division=0)), weighted_f1=float(f1_score(yy, pr, labels=list(range(8)), average="weighted", zero_division=0)))
        return o
    r = M.load_run(task, name, seed); return {sp: dict(macro_f1=r[sp]["macro_f1"], weighted_f1=r[sp]["weighted_f1"]) for sp in ("dev", "test")}


def best(task, pair, seed, head):
    c = [(n, stored(task, n, seed)) for n in pair]
    return max(c, key=lambda t: t[1]["dev"][head])      # first of the pair wins ties (text, then image)


def table(cfg):
    out = []
    for name, label, own in cfg["models"]:
        row = dict(model=name, label=label, own_components=list(own), fused={}, vs_best_unimodal={}, vs_own={})
        per = {k: {sp: {m: [] for m in ("weighted_f1", "macro_f1")} for sp in ("dev", "test")} for k in ("fused", "uni", "own")}
        for s in SEEDS:
            f = stored(cfg["task"], name, s); u = best(cfg["task"], cfg["uni"], s, cfg["head"])[1]; o = best(cfg["task"], own, s, cfg["head"])[1]
            for sp in ("dev", "test"):
                for m in ("weighted_f1", "macro_f1"):
                    per["fused"][sp][m].append(f[sp][m]); per["uni"][sp][m].append(f[sp][m] - u[sp][m]); per["own"][sp][m].append(f[sp][m] - o[sp][m])
        ms = lambda v: [float(np.mean(v)), float(np.std(v, ddof=1))]
        for sp in ("dev", "test"):
            for m in ("weighted_f1", "macro_f1"):
                row["fused"][f"{sp}_{m}"] = ms(per["fused"][sp][m]); row["vs_best_unimodal"][f"{sp}_{m}"] = ms(per["uni"][sp][m]); row["vs_own"][f"{sp}_{m}"] = ms(per["own"][sp][m])
        out.append(row)
    return out


if __name__ == "__main__":
    res = {"T1": dict(headline="weighted_f1", rows=table(T1)), "T2": dict(headline="macro_f1", rows=table(T2))}
    (RES / "D4_fusion_gain_both.json").write_text(json.dumps(res, indent=1))
    sg = lambda v: f"{v[0]:+.4f} ± {v[1]:.4f}"; md = "# Fusion gain, both tasks, both baselines (mean ± std over 3 seeds; baseline picked per seed on dev headline; test reported afterwards)\n"
    for t, cfg in (("T1", T1), ("T2", T2)):
        h, o = cfg["head"], cfg["other"]
        md += (f"\n## {t} (headline: {h}; the other metric is in the right-hand columns)\n\n| Fused model | own components | fused {h} (dev / test) | gain vs best unimodal {h} (dev / test) | gain vs own components {h} (dev / test) | gain vs best unimodal {o} (dev / test) | gain vs own components {o} (dev / test) |\n|---|---|---|---|---|---|---|\n")
        for r in res[t]["rows"]:
            c = lambda d, m: f"{sg(d[f'dev_{m}'])} / {sg(d[f'test_{m}'])}"
            fu = f"{r['fused'][f'dev_{h}'][0]:.4f} / {r['fused'][f'test_{h}'][0]:.4f}"
            md += f"| {r['label']} | {' + '.join(r['own_components'])} | {fu} | {c(r['vs_best_unimodal'], h)} | {c(r['vs_own'], h)} | {c(r['vs_best_unimodal'], o)} | {c(r['vs_own'], o)} |\n"
    (RES / "D4_fusion_gain_both.md").write_text(md, encoding="utf-8"); print(md)
