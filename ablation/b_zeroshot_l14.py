"""B: zero-shot Task 3 with CLIP ViT-L/14, using exactly the prompts and procedure of the B/32 zero-shot run (s1_zeroshot.py; prompts in results/s1_prompts.json).

Image side: the L/14 image embeddings of the damage dev/test images (L2-normalised, from artifacts/ablation/clip_l14). Text side: the L/14 text tower. Class matrix for
each prompt set = mean of the L2-normalised prompt embeddings, re-normalised (E1, E2, E3 as in S1). Procedure as in S1: the prompt set is chosen by DEV macro-F1, TEST is
scored once for the chosen set; B/32's chosen set (E3) is also scored on L/14 so that "same prompts" is unambiguous; the dev-calibrated-bias variant (NOT pure zero-shot) is
computed as in S1 for completeness. B/32 reference numbers are read from results/s1_*.csv (not recomputed). Writes results/ablation/X/b_zeroshot_l14_seed42.json.
"""
import itertools
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import recall_score

from common import ABL_ART, DATA, RES, ROOT, SPLIT_DIR, cc, write_row

MODEL = "openai/clip-vit-large-patch14"
LAB = cc.T3_LABELS
L14 = ABL_ART / "clip_l14"


def main():
    dev = cc.load_split("damage", "dev", DATA, SPLIT_DIR, verbose=False)[0]; test = cc.load_split("damage", "test", DATA, SPLIT_DIR, verbose=False)[0]

    def img_feats(df, split):
        assert json.load(open(L14 / f"clip_damage_{split}.ids.json")) == list(df.image_id)
        x = np.load(L14 / f"clip_damage_{split}.npy"); assert x.shape == (len(df), 768)
        return cc.l2(x)
    Xd, Xt = img_feats(dev, "dev"), img_feats(test, "test")

    from transformers import CLIPModel, CLIPProcessor
    proc = CLIPProcessor.from_pretrained(MODEL)
    clip = CLIPModel.from_pretrained(MODEL, dtype=torch.float16).to("cuda").eval()

    @torch.no_grad()
    def text_emb(texts):
        tok = proc.tokenizer(texts, padding=True, return_tensors="pt").to("cuda")
        out = clip.get_text_features(**tok)
        if not isinstance(out, torch.Tensor):
            out = out.text_embeds if hasattr(out, "text_embeds") else out.pooler_output
        return cc.l2(out.float().cpu().numpy())

    P = json.load(open(ROOT / "results" / "s1_prompts.json"))

    def prompts_for(name):
        if name in ("E1", "E2"):
            return P[name]
        ph = {c: [p.removeprefix("a photo of ") for p in v] for c, v in P["E2"].items()}
        return {c: [t.format(p) for p in v for t in P["E3"]["templates"]] for c, v in ph.items()}

    def class_matrix(name):
        pr = prompts_for(name)
        return cc.l2(np.stack([cc.l2(text_emb(pr[c])).mean(0) for c in LAB]))
    W = {n: class_matrix(n) for n in ("E1", "E2", "E3")}

    def full(y, pred):
        m = cc.metrics(y, pred, labels=LAB)
        rec = recall_score(y, pred, labels=LAB, average=None, zero_division=0)
        return dict(m, **{f"recall_{l}": float(r) for l, r in zip(LAB, rec)})
    yd, yt = dev.label.to_numpy(), test.label.to_numpy()
    predict = lambda X, Wm, b=0: np.array(LAB)[(X @ Wm.T + b).argmax(1)]

    cand = {n: full(yd, predict(Xd, W[n])) for n in W}
    best = max(cand, key=lambda n: cand[n]["macro_f1"])
    s1_sel = json.load(open(RES.parent.parent / "results" / "s1_selection.json"))["chosen"]
    test_chosen = full(yt, predict(Xt, W[best]))
    test_e3 = full(yt, predict(Xt, W[s1_sel])) if best != s1_sel else test_chosen
    grid = np.arange(-0.06, 0.0601, 0.002); bestb, bestf = (0, 0), -1
    for b1, b2 in itertools.product(grid, grid):
        f = cc.metrics(yd, predict(Xd, W[best], np.array([0, b1, b2])), labels=LAB)["macro_f1"]
        if f > bestf + 1e-12:
            bestf, bestb = f, (b1, b2)
    test_cal = full(yt, predict(Xt, W[best], np.array([0, *bestb])))

    s1c = pd.read_csv(ROOT / "results" / "s1_dev_candidates.csv").set_index("ensemble")
    s1z = pd.read_csv(ROOT / "results" / "s1_zeroshot.csv")
    b32_zs = s1z[s1z.method.str.contains("zero-shot")].iloc[0]; b32_cal = s1z[s1z.method.str.contains("dev-calibrated")].iloc[0]
    print("dev macro-F1 per prompt set   L/14: " + ", ".join(f"{n} {cand[n]['macro_f1']:.4f}" for n in W) + "   | B/32: " + ", ".join(f"{n} {s1c.loc[n, 'macro_f1']:.4f}" for n in W))
    print(f"chosen on dev: L/14 {best} | B/32 {s1_sel}")
    print(f"TEST zero-shot  L/14 ({best}): acc {test_chosen['acc']:.4f} macro-F1 {test_chosen['macro_f1']:.4f} weighted-F1 {test_chosen['weighted_f1']:.4f} | "
          f"B/32 ({s1_sel}): acc {b32_zs['acc']:.4f} macro-F1 {b32_zs['macro_f1']:.4f} weighted-F1 {b32_zs['weighted_f1']:.4f}")
    if best != s1_sel:
        print(f"TEST zero-shot  L/14 with B/32's chosen set {s1_sel}: acc {test_e3['acc']:.4f} macro-F1 {test_e3['macro_f1']:.4f}")
    print(f"TEST dev-calibrated biases (not pure zero-shot)  L/14: macro-F1 {test_cal['macro_f1']:.4f} | B/32: {b32_cal['macro_f1']:.4f}")
    write_row("X", "b_zeroshot_l14", 42, dict(
        task="T3", tier="gpu-small", model=MODEL, n_test=int(len(yt)), n_dev=int(len(yd)), dev_candidates_l14={n: cand[n] for n in W},
        dev_candidates_b32={n: float(s1c.loc[n, "macro_f1"]) for n in W}, chosen_l14=best, chosen_b32=s1_sel, test_l14_chosen=test_chosen,
        test_l14_with_b32_chosen_set=test_e3, test_l14_dev_calibrated=dict(test_cal, biases=[0.0, *map(float, bestb)], dev_macro_f1=float(bestf)),
        test_b32_zero_shot=dict(acc=float(b32_zs["acc"]), macro_f1=float(b32_zs["macro_f1"]), weighted_f1=float(b32_zs["weighted_f1"])),
        test_b32_dev_calibrated=dict(acc=float(b32_cal["acc"]), macro_f1=float(b32_cal["macro_f1"]), weighted_f1=float(b32_cal["weighted_f1"]))))


if __name__ == "__main__":
    main()
