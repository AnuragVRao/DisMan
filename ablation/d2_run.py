"""Day 2 runner (ablation/SELECTION_RULE.md "Day 2 addendum"). One JSON per run in results/ablation/H/, seeds 42, 43, 44.

  python d2_run.py d1                                     image backbones, image-only head; the B/32 and L/14 arms must reproduce the stored runs EXACTLY first
  python d2_run.py d2                                     text encoders, text-only head on the frozen sentence embedding
  python d2_run.py d3 --img <backbone> --txt <encoder>    five heads with the D1 and D2 winners (image_only must equal the D1 run exactly)
  python d2_run.py d4 --head <kind> --img .. --txt .. [--shard i/n]   the fixed 27-configuration grid on the D3 winner
"""
import argparse
import itertools
import json

import numpy as np

from d2_common import HP0, KINDS, SEEDS, RES, data, img_feats, run_arm, tok_feats, txt_feats

GRID = dict(lr=(3e-5, 1e-4, 3e-4), dropout=(0.1, 0.2, 0.4), d_model=(128, 256, 512))
STORED = {"clip_b32": "b_t2_imageonly_B32_raw", "clip_l14": "b_t2_imageonly_L14_norm"}


def inputs(kind, img=None, txt=None, tok=None):
    """{'train'|'dev'|'test': tuple of the tensors the head consumes, in forward order}."""
    out = {}
    for s in ("train", "dev", "test"):
        if kind == "image_only":
            out[s] = (img[s],)
        elif kind == "text_only":
            out[s] = (txt[s],)
        elif kind in ("concat", "gated_late"):
            out[s] = (img[s], txt[s])
        else:
            out[s] = (img[s], tok[s][0], tok[s][1])
    return out


def d1():
    for arm in ("clip_b32", "clip_l14", "dinov2", "resnet50"):
        f = img_feats(arm); xs = inputs("image_only", img=f); dim = f["train"].shape[1]
        for seed in SEEDS:
            r = run_arm("D1", f"d1_{arm}", "image_only", xs, seed, HP0, d_img=dim, backbones=(arm,))
            if arm in STORED:     # validation: the Day 2 trainer must reproduce the stored image-only run exactly
                st = json.load(open(RES / "X" / f"{STORED[arm]}_seed{seed}.json"))
                same = (r["dev_macro_f1"] == st["dev_macro_f1"] and r["test_macro_f1"] == st["test_macro_f1"] and r["test"]["acc"] == st["test_acc"] and r["pred_test"] == st["pred"]
                        and r["best_epoch"] == st["best_epoch"])
                print(f"    validation vs stored {STORED[arm]} seed {seed}: identical dev F1, test F1, accuracy, best epoch and all test predictions: {same}", flush=True)
                assert same, f"{arm} seed {seed} does not reproduce the stored run"


def d2():
    for arm in ("minilm", "mpnet"):
        f = txt_feats(arm); xs = inputs("text_only", txt=f); dim = f["train"].shape[1]
        for seed in SEEDS:
            run_arm("D2", f"d2_{arm}", "text_only", xs, seed, HP0, d_txt=dim, backbones=(arm,))


def _chosen(a):
    img, txt = img_feats(a.img), txt_feats(a.txt)
    return img, txt, img["train"].shape[1], txt["train"].shape[1]


def d3(a):
    img, txt, di, dt = _chosen(a)
    tok = tok_feats(a.txt)
    for kind in KINDS:
        xs = inputs(kind, img, txt, tok)
        bb = {"image_only": (a.img,), "text_only": (a.txt,)}.get(kind, (a.img, a.txt))
        for seed in SEEDS:
            r = run_arm("D3", f"d3_{kind}", kind, xs, seed, HP0, d_img=di, d_txt=dt, backbones=bb, extra=dict(img=a.img, txt=a.txt))
            if kind == "image_only":     # the same code and seed as the D1 run of this backbone: identical result
                d = json.load(open(RES / "H" / f"d1_{a.img}_seed{seed}.json"))
                assert r["pred_test"] == d["pred_test"] and r["dev_macro_f1"] == d["dev_macro_f1"], "D3 image_only != D1 run"
                print("    D3 image_only reproduces the D1 run exactly", flush=True)


def d4(a):
    img, txt, di, dt = _chosen(a)
    tok = tok_feats(a.txt) if a.head == "cross_attention" else None
    xs = inputs(a.head, img, txt, tok)
    bb = {"image_only": (a.img,), "text_only": (a.txt,)}.get(a.head, (a.img, a.txt))
    configs = list(itertools.product(GRID["lr"], GRID["dropout"], GRID["d_model"]))
    i, n = (int(v) for v in a.shard.split("/"))
    for k, (lr, do, dm) in enumerate(configs):
        if k % n != i:
            continue
        hp = dict(lr=lr, dropout=do, d_model=dm)
        for seed in SEEDS:
            run_arm("D4", f"d4_lr{lr:g}_do{do:g}_dm{dm}", a.head, xs, seed, hp, d_img=di, d_txt=dt, backbones=bb, extra=dict(img=a.img, txt=a.txt))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("decision", choices=("d1", "d2", "d3", "d4"))
    ap.add_argument("--img"); ap.add_argument("--txt"); ap.add_argument("--head", default="image_only"); ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    data()
    {"d1": lambda: d1(), "d2": lambda: d2(), "d3": lambda: d3(a), "d4": lambda: d4(a)}[a.decision]()
