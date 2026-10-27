"""Part B component-ablation variants of the v2 Task 2 head (ABLATION_PLAN.md).

All variants take the cached frozen-CLIP vector plus DistilBERT tokens, with the same widths/dropout as v2
(d_model 256, 4 heads, dropout 0.2). Only the removed component differs:

  text_only   DistilBERT -> txt_proj -> CLS token -> MLP (no image stream, no attention)
  image_only  cached CLIP vector -> img_proj -> MLP      (no DistilBERT, no attention)
  concat      [img_proj(CLIP), txt_proj(CLS)] -> MLP      (no attention blocks)
  no_t2i      v2 without the text->image attention block: [img_to_txt(img, txt), txt_proj(CLS)] -> MLP, image->text attention kept

MLP = Dropout, Linear(in, 256), ReLU, Dropout, Linear(256, classes), exactly v2's classifier with `in` adjusted (256 or 512).
The full model is crisis_common.DualStreamClassifier (unchanged).
"""
import torch
import torch.nn as nn

import crisis_common as cc

VARIANTS = ("text_only", "image_only", "concat", "no_t2i")


class Variant(nn.Module):
    def __init__(self, kind, num_classes, bert_config=None, img_dim=512, d_model=256, num_heads=4, dropout=0.2):
        super().__init__()
        assert kind in VARIANTS
        self.kind = kind
        if kind != "image_only":
            from transformers import DistilBertModel
            self.text_encoder = (DistilBertModel(bert_config) if bert_config is not None
                                 else DistilBertModel.from_pretrained(cc.BERT_NAME))
            self.txt_proj = nn.Sequential(nn.Linear(self.text_encoder.config.dim, d_model), nn.LayerNorm(d_model))
        if kind != "text_only":
            self.img_proj = nn.Sequential(nn.Linear(img_dim, d_model), nn.LayerNorm(d_model))
        if kind == "no_t2i":
            self.img_to_txt = cc.CrossModalAttention(d_model, num_heads, dropout)
        width = d_model * 2 if kind in ("concat", "no_t2i") else d_model
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, d_model), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, img_vec, input_ids, attn_mask):
        k = self.kind
        if k == "image_only":
            return self.classifier(self.img_proj(img_vec))
        txt = self.txt_proj(self.text_encoder(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state)
        cls = txt[:, 0, :]
        if k == "text_only":
            return self.classifier(cls)
        img = self.img_proj(img_vec)
        if k == "concat":
            return self.classifier(torch.cat([img, cls], dim=-1))
        img_att = self.img_to_txt(img.unsqueeze(1), txt, attn_mask == 0).squeeze(1)   # no_t2i
        return self.classifier(torch.cat([img_att, cls], dim=-1))


def param_groups(model, lr):
    """v2's optimiser groups: DistilBERT at 0.1x lr, every other trainable module at lr."""
    groups = []
    if hasattr(model, "text_encoder"):
        groups.append({"params": model.text_encoder.parameters(), "lr": lr * 0.1})
    for name in ("img_proj", "txt_proj", "img_to_txt", "txt_to_img", "classifier"):
        if hasattr(model, name):
            groups.append({"params": getattr(model, name).parameters(), "lr": lr})
    return groups


class FusionModel(nn.Module):
    """crisis_common.DualStreamClassifier with a swappable pretrained text encoder (fine-tuned fusion addendum in SELECTION_RULE.md).

    Same head, same module names and construction order as DualStreamClassifier (text_encoder, img_proj, txt_proj, img_to_txt, txt_to_img, classifier), so a state dict
    of DualStreamClassifier loads into it when the encoder is a DistilBertModel; only the encoder and the width of txt_proj change for another encoder."""

    def __init__(self, num_classes, text_encoder, txt_dim, img_dim=512, d_model=256, num_heads=4, dropout=0.2):
        super().__init__()
        self.text_encoder = text_encoder
        self.img_proj = nn.Sequential(nn.Linear(img_dim, d_model), nn.LayerNorm(d_model))
        self.txt_proj = nn.Sequential(nn.Linear(txt_dim, d_model), nn.LayerNorm(d_model))
        self.img_to_txt = cc.CrossModalAttention(d_model, num_heads, dropout)
        self.txt_to_img = cc.CrossModalAttention(d_model, num_heads, dropout)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, img_vec, input_ids, attn_mask):
        img = self.img_proj(img_vec).unsqueeze(1)
        txt = self.txt_proj(self.text_encoder(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state)
        img_att = self.img_to_txt(img, txt, attn_mask == 0).squeeze(1)
        txt_att = self.txt_to_img(txt[:, 0:1, :], img).squeeze(1)
        return self.classifier(torch.cat([img_att, txt_att], dim=-1))
