"""Day 4: provenance for every run JSON and the asserts that read it (the main guard against cross-task contamination, SELECTION_RULE_DAY4.md rules 1, 11).
Keys written on every run: task_name ("informative" | "humanitarian"; the older key `task` stays "T1"/"T2" because run filenames use it), label_col, train_split_sha256 (sha256 of the
official train TSV of that task), init_from (hub id, "none (...)" for models trained from scratch, or a local path), feature_cache (dir + sha256 of its split files, or None).
The cross-task overlap is large (D4_cross_task_overlap.json), so a model may never be initialised from a checkpoint trained on the other task."""
import hashlib
import json
import os
import re

from common import ABL_ART, SPLIT_DIR

TASKS = {"informative": ("label_text", "T1"), "humanitarian": ("label_image", "T2")}
HUB = {"openai/clip-vit-base-patch32", "microsoft/resnet-50", "facebook/dinov2-base", "openai/clip-vit-large-patch14", "vinai/bertweet-base", "distilbert-base-uncased",
       "sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"}
_C = {}


def sha256_file(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def train_sha(task_name):
    if task_name not in _C:
        _C[task_name] = sha256_file(SPLIT_DIR / f"task_{task_name}_text_img_train.tsv")
    return _C[task_name]


def cache_sha(feat_dir):
    """sha256 over the sorted (name, sha256) pairs of the split files in a feature-cache directory."""
    key = str(feat_dir)
    if key not in _C:
        h = hashlib.sha256()
        for p in sorted(feat_dir.glob("*_*.npy")) + sorted(feat_dir.glob("*_*.ids.json")):
            h.update(p.name.encode()); h.update(sha256_file(p).encode())
        _C[key] = h.hexdigest()
    return _C[key]


def is_hub_id(s):
    return bool(re.fullmatch(r"[\w.-]+/[\w.-]+|[\w.-]+", s)) and not os.path.exists(s) and not s.endswith((".pt", ".pth", ".bin", ".ckpt", ".safetensors"))


def prov(task_name, init_from, feature=None):
    """feature: the cache directory name under artifacts/ablation/d4_feats (Task 1) or d2_feats (Task 2), or None."""
    label_col, _ = TASKS[task_name]
    fc = None
    if feature:
        d = ABL_ART / ("d4_feats" if task_name == "informative" else "d2_feats") / feature
        fc = dict(path=str(d.relative_to(ABL_ART.parent.parent)), sha256=cache_sha(d), encoder=json.load(open(d / "meta.json")).get("model"))
    return dict(task_name=task_name, label_col=label_col, train_split_sha256=train_sha(task_name), init_from=init_from, feature_cache=fc)


def assert_run(run, task_name, name=""):
    """Used by every script that consumes saved component runs."""
    label_col, tid = TASKS[task_name]
    assert run.get("task_name") == task_name, f"{name}: task_name {run.get('task_name')!r} != {task_name!r}"
    assert run.get("task") == tid, f"{name}: task {run.get('task')!r} != {tid!r}"
    assert run.get("label_col") == label_col, f"{name}: label_col {run.get('label_col')!r} != {label_col!r}"
    assert run.get("train_split_sha256") == train_sha(task_name), f"{name}: trained on a different train split"
    init = run.get("init_from", "")
    assert init.startswith("none") or is_hub_id(init) or "glove" in init.lower(), f"{name}: init_from {init!r} is not a hub id / from-scratch / GloVe"
    fc = run.get("feature_cache")
    if fc:
        assert fc["encoder"] in HUB, f"{name}: feature encoder {fc['encoder']!r} is not a known pretrained hub model"


def assert_init(init_from, task_name):
    """Task 2 scripts call this (mirror image of the Task 1 rule): hub ids only, never a local checkpoint."""
    assert is_hub_id(init_from), f"{task_name}: init_from {init_from!r} must be a hub id, not a local path / checkpoint"


# provenance of the runs made before this module existed (read off the scripts that made them); metadata only
BACKFILL = {"tfidf_lr": ("none (TF-IDF + LR fit on T1 train)", None), "tfidf_nb": ("none (TF-IDF + NB fit on T1 train)", None), "tfidf_rf": ("none (TF-IDF + RF fit on T1 train)", None),
            "tfidf_svm": ("none (TF-IDF + LinearSVC fit on T1 train)", None), "tfidf_nb_bal": ("none (TF-IDF + NB fit on T1 train)", None), "tfidf_cnb": ("none (TF-IDF + ComplementNB fit on T1 train)", None),
            "majority": ("none (majority class of T1 train)", None), "bilstm_glove": ("none (BiLSTM random init; frozen GloVe Twitter 200d, see `glove`)", None),
            "bertweet": ("vinai/bertweet-base", None), "bertweet_noweight": ("vinai/bertweet-base", None),
            "img_resnet50": ("none (head trained from scratch on frozen microsoft/resnet-50 features)", "resnet50"),
            "img_dinov2": ("none (head trained from scratch on frozen facebook/dinov2-base features)", "dinov2"),
            "img_clip_l14": ("none (head trained from scratch on frozen openai/clip-vit-large-patch14 features)", "clip_l14")}


def check_t2_cache(name, ids_by_split):
    """Task 2 frozen-feature cache (artifacts/ablation/d2_feats/<name>): row order equals the T2 split image_id order, and the encoder is a pretrained, frozen hub model
    (the cache holds fixed features extracted once with the hub weights in eval mode; no training touches the encoder)."""
    d = ABL_ART / "d2_feats" / name
    meta = json.load(open(d / "meta.json"))
    assert meta["model"] in HUB, f"{name}: encoder {meta['model']!r} is not a known pretrained hub model"
    for s, ids in ids_by_split.items():
        assert json.load(open(d / f"humanitarian_{s}.ids.json")) == list(ids), f"{name}/{s}: cache row order differs from the T2 split image_id order"
    return dict(path=str(d.relative_to(ABL_ART.parent.parent)), sha256=cache_sha(d), encoder=meta["model"], frozen=True)
