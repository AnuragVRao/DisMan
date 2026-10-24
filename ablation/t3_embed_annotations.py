"""Embed all 18,082 annotation rows (the v1 T3 index contents) with the v2 encoders and cache them.

Outputs (artifacts/ablation/emb/): clip_annot_hf.npy (raw 512-d HF CLIP, QuickGELU), sbert_annot.npy (384-d, normalised).
"""
import torch

from common import ABL_EMB, ART, cc, load_annotations, timer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ann = load_annotations()
ids = ann["image_id"].tolist()
print(len(ann), "annotation rows;", ann["split"].value_counts().to_dict())

elapsed = timer()
clip, proc = cc.load_clip(DEVICE, ART / "clip")
assert clip.config.vision_config.hidden_act == "quick_gelu"

def _compute_img():
    feats, bad = cc.embed_images(ann["image_path"].tolist(), clip, proc, DEVICE)
    print("unreadable images replaced by grey:", bad)
    return feats


img = cc.cached(ABL_EMB / "clip_annot_hf.npy", ids, _compute_img)
print("CLIP", img.shape, f"{elapsed():.0f}s")

sbert = cc.load_sbert(DEVICE, ART / "sbert")
txt = cc.cached(ABL_EMB / "sbert_annot.npy", ids, lambda: cc.embed_texts(ann["text"].tolist(), sbert))
print("SBERT", txt.shape, f"{elapsed():.0f}s")
