"""Shared code for capstone-v2.ipynb and demo/pipeline.py.

Keeping the loaders, embedders and models in one module guarantees that the demo
embeds queries exactly the way the training/evaluation notebook embedded the data.
"""
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

SEED = 42
CLIP_NAME = "openai/clip-vit-base-patch32"
SBERT_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BERT_NAME = "distilbert-base-uncased"

T1_LABELS = ["informative", "not_informative"]
T3_LABELS = ["Severe damage", "Mild damage", "Little or no damage"]


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Text cleaning (unchanged from the original notebook) ─────────────────────
ABBREV_MAP = {
    r'\bpls\b': 'please', r'\bppl\b': 'people',
    r'\bw\/': 'with', r'\bw/o\b': 'without',
    r'\bthx\b': 'thanks', r'\bplz\b': 'please',
    r'\bu\b': 'you', r'\bur\b': 'your',
    r'\bgovt\b': 'government', r'\binfo\b': 'information',
}


def clean_tweet(text) -> str:
    if not isinstance(text, str):
        return ''
    text = re.sub(r'^RT\s+', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '@user', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    for pat, rep in ABBREV_MAP.items():
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()


# ── Label normalisation (unchanged from the original notebook) ───────────────
T2_LABEL_MAP = {
    'affected_individuals': 'Affected individuals',
    'affected_individuals_people': 'Affected individuals',
    'infrastructure_and_utility_damage': 'Infrastructure and utility damage',
    'infrastructure_and_utilities_damage': 'Infrastructure and utility damage',
    'vehicle_damage': 'Vehicle damage',
    'rescue_volunteering_or_donation_effort': 'Rescue, volunteering or donation effort',
    'other_relevant_information': 'Other relevant information',
    'not_humanitarian': 'Not humanitarian',
    'missing_or_found_people': 'Missing or found people',
    'injured_or_dead_people': 'Injured or dead people',
}
T3_DAMAGE_MAP = {
    'severe_damage': 'Severe damage',
    'mild_damage': 'Mild damage',
    'little_or_no_damage': 'Little or no damage',
    'no_damage': 'Little or no damage',
    'none': 'Little or no damage',
    'dont_know_or_cant_judge': "Don't know or can't judge",
}


def _norm_key(raw):
    return str(raw).strip().lower().replace(' ', '_').replace('-', '_').replace('/', '_')


def t2_normalise(raw):
    if pd.isna(raw):
        return 'Unknown'
    return T2_LABEL_MAP.get(_norm_key(raw), str(raw).strip())


def t3_normalise(raw):
    if pd.isna(raw):
        return 'Unknown'
    return T3_DAMAGE_MAP.get(_norm_key(raw), str(raw).strip())


# ── Split loading with a drop log ────────────────────────────────────────────
# Which column of the official split files is the label for each task. These are the
# same columns the original notebook used (T1: text label, T2: image label, T3: `label`).
LABEL_COL = {'informative': 'label_text', 'humanitarian': 'label_image', 'damage': 'label'}
NORMALISE = {'informative': lambda r: 'Unknown' if pd.isna(r) else str(r).strip(),
             'humanitarian': t2_normalise, 'damage': t3_normalise}


def _per_event(df):
    return df.groupby('event_name').size().to_dict() if len(df) else {}


def load_split(task, split, data_root, split_dir, verbose=True):
    """Load one official split file and resolve image paths.

    Returns (df, droplog). droplog is a list of dicts {task, split, reason, n_dropped, per_event}.
    Every filter prints row counts total and per event; nothing is dropped silently.
    """
    data_root, split_dir = Path(data_root), Path(split_dir)
    df = pd.read_csv(split_dir / f"task_{task}_text_img_{split}.tsv", sep='\t', dtype=str)
    df.columns = df.columns.str.strip().str.lower()
    log = []
    tag = f"[{task}/{split}]"

    def stage(name, new_df, reason=None):
        nonlocal df
        dropped = df[~df.index.isin(new_df.index)]
        if reason is not None:
            log.append(dict(task=task, split=split, reason=reason, n_dropped=len(dropped),
                            per_event=_per_event(dropped)))
        df = new_df
        if verbose:
            extra = f"  (dropped {len(dropped)}: {reason}; by event {_per_event(dropped)})" if len(dropped) else ""
            print(f"{tag} {name}: {len(df)} rows | by event {_per_event(df)}{extra}")

    stage("loaded", df)
    stage("dedup image_id", df.drop_duplicates(subset=['image_id']), "duplicate image_id")

    # FIX 1: removeprefix (strip a prefix), not lstrip (strips a CHARACTER SET).
    def resolve(p):
        if pd.isna(p):
            return None
        first = str(p).split(',')[0].strip().removeprefix('data_image/')
        full = data_root / 'data_image' / first
        return str(full) if full.exists() else None

    df = df.copy()
    df['img_path'] = df['image'].map(resolve)
    stage("image resolved", df[df['img_path'].notna()], "image file missing / no image field")
    stage("non-empty text", df[df['tweet_text'].notna() & (df['tweet_text'].str.strip() != '')],
          "empty tweet text")

    df = df.copy()
    df['label'] = df[LABEL_COL[task]].map(NORMALISE[task])
    stage("label known", df[df['label'] != 'Unknown'], "missing/unknown label")

    df = df.copy()
    df['text'] = df['tweet_text'].map(clean_tweet)
    df['rel_image'] = df['image'].map(lambda p: str(p).split(',')[0].strip())
    keep = ['event_name', 'tweet_id', 'image_id', 'tweet_text', 'text', 'rel_image', 'img_path', 'label']
    return df[keep].reset_index(drop=True), log


# ── Models: CLIP (frozen image encoder) and SBERT ────────────────────────────
def load_clip(device, source=CLIP_NAME):
    """FIX 4 + FIX 6: HF CLIPModel (QuickGELU, as the OpenAI weights require)."""
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(source).to(device).eval()
    proc = CLIPProcessor.from_pretrained(source)
    return model, proc


def load_sbert(device, source=SBERT_NAME):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(str(source), device=str(device))


def _open_rgb(p):
    try:
        return Image.open(p).convert('RGB'), False
    except Exception:
        return Image.new('RGB', (224, 224), (128, 128, 128)), True


@torch.no_grad()
def clip_image_features(clip, proc, images, device):
    """Raw (un-normalised) 512-d projected image features for a list of PIL images."""
    px = proc(images=images, return_tensors='pt')['pixel_values'].to(device)
    out = clip.get_image_features(pixel_values=px)
    if not isinstance(out, torch.Tensor):  # transformers>=5 may return a ModelOutput
        out = out.image_embeds if hasattr(out, 'image_embeds') else out.pooler_output
    return out.float().cpu().numpy()


def embed_images(paths, clip, proc, device, batch_size=64, workers=8):
    """Frozen-CLIP image features for file paths. Returns (feats [N,512], n_unreadable)."""
    feats, bad = [], 0
    with ThreadPoolExecutor(workers) as ex:
        for i in range(0, len(paths), batch_size):
            loaded = list(ex.map(_open_rgb, paths[i:i + batch_size]))
            bad += sum(b for _, b in loaded)
            feats.append(clip_image_features(clip, proc, [im for im, _ in loaded], device))
            if (i // batch_size) % 20 == 0:
                print(f"  {min(i + batch_size, len(paths))}/{len(paths)} images", flush=True)
    return np.concatenate(feats), bad


def embed_texts(texts, sbert, batch_size=64):
    """L2-normalised SBERT embeddings; empty text is embedded as 'no text' (index and queries alike)."""
    texts = [t if t.strip() else 'no text' for t in texts]
    return sbert.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=False).astype('float32')


def cached(path, ids, compute):
    """Load an embedding cache if it exists and its ids match; otherwise compute and save it."""
    path = Path(path)
    idp = path.with_suffix('.ids.json')
    if path.exists() and idp.exists() and json.loads(idp.read_text()) == list(ids):
        return np.load(path)
    arr = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    idp.write_text(json.dumps(list(ids)))
    return arr


def l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-12)


# ── Task 3: image+text vector, FAISS kNN, deterministic vote ─────────────────
def t3_vectors(img_raw, txt_emb):
    """concat(normalised CLIP image, normalised SBERT text), re-normalised for inner product."""
    return l2(np.concatenate([l2(img_raw), txt_emb], axis=1)).astype('float32')


def knn_vote(labels, scores, label_order=T3_LABELS):
    """Majority vote over neighbour labels. Ties are broken by summed similarity (deterministic).

    Returns (label, vote_share {label: share}).
    """
    k = len(labels)
    share = {l: sum(1 for x in labels if x == l) / k for l in label_order}
    total = {l: sum(s for x, s in zip(labels, scores) if x == l) for l in label_order}
    best = max(label_order, key=lambda l: (share[l], total[l]))
    return best, share


# ── Task 2: DistilBERT (fine-tuned) + cached CLIP vector, cross-attention, MLP ─
class CrossModalAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, key_padding_mask=None):
        out, _ = self.attn(query, key_value, key_value, key_padding_mask=key_padding_mask)
        return self.norm(query + self.dropout(out))


class DualStreamClassifier(nn.Module):
    """Same head as the original notebook, but takes the cached frozen-CLIP vector directly."""

    def __init__(self, num_classes, bert_config=None, img_dim=512, d_model=256, num_heads=4, dropout=0.2):
        super().__init__()
        from transformers import DistilBertModel
        self.text_encoder = (DistilBertModel(bert_config) if bert_config is not None
                             else DistilBertModel.from_pretrained(BERT_NAME))
        txt_dim = self.text_encoder.config.dim
        self.img_proj = nn.Sequential(nn.Linear(img_dim, d_model), nn.LayerNorm(d_model))
        self.txt_proj = nn.Sequential(nn.Linear(txt_dim, d_model), nn.LayerNorm(d_model))
        self.img_to_txt = CrossModalAttention(d_model, num_heads, dropout)
        # Single pooled image vector => one key; this attention is effectively a projection.
        self.txt_to_img = CrossModalAttention(d_model, num_heads, dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(d_model * 2, d_model), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, img_vec, input_ids, attn_mask):
        img = self.img_proj(img_vec).unsqueeze(1)
        txt = self.txt_proj(self.text_encoder(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state)
        img_att = self.img_to_txt(img, txt, attn_mask == 0).squeeze(1)
        txt_att = self.txt_to_img(txt[:, 0:1, :], img).squeeze(1)
        return self.classifier(torch.cat([img_att, txt_att], dim=-1))


def tokenize(tokenizer, texts, max_len=128):
    enc = tokenizer(list(texts), max_length=max_len, padding='max_length', truncation=True, return_tensors='pt')
    return enc['input_ids'], enc['attention_mask']


@torch.no_grad()
def t2_predict_proba(model, img_vecs, ids, mask, device, batch_size=128):
    model.eval()
    out = []
    for i in range(0, len(ids), batch_size):
        with torch.autocast('cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
            logits = model(torch.as_tensor(img_vecs[i:i + batch_size]).float().to(device),
                           ids[i:i + batch_size].to(device), mask[i:i + batch_size].to(device))
        out.append(torch.softmax(logits.float(), -1).cpu().numpy())
    return np.concatenate(out)


# ── Metrics ──────────────────────────────────────────────────────────────────
def metrics(y_true, y_pred, labels=None):
    """Accuracy, macro-F1 (headline) and weighted-F1. `labels` fixes the class set for the averages."""
    return dict(acc=accuracy_score(y_true, y_pred),
                macro_f1=f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
                weighted_f1=f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0),
                n=len(y_true))
