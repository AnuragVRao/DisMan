#!/usr/bin/env python3
"""
CrisisMMD v2.0 - Task 3: Damage Severity Assessment Preprocessing Pipeline
==========================================================================
This script performs comprehensive multimodal preprocessing for Task 3:
Damage Severity Assessment on Infrastructure & Utility Damage images and tweets.

Task 3 Target Classes (3 Severity Levels):
  0: little_or_no_damage  (Minority: 13.5%)
  1: mild_damage          (Moderate: 23.8%)
  2: severe_damage        (Majority: 62.7%)

Total Benchmark Samples:
  Train: 2,468 rows
  Dev:     529 rows
  Test:    529 rows
  Total: 3,526 rows (matches official 3-class benchmark excluding "don't know")

Implemented Preprocessing Paradigms:
1. Concept 1 (Class-Weighted): Full 2,468 train samples with cost-sensitive weights [2.4705, 1.4015, 0.5314]
2. Concept 2 (Oversampled): 100% perfectly balanced 3-class set: 4,644 rows (1,548 each = 33.3% each)
3. Concept 3 (Undersampled): 1:1:1 balanced subset: 999 rows (333 each = 33.3% each)

All preprocessed outputs are saved to 'preprocessed_task3/'.
"""

import os
import sys
import re
import html
import json
import argparse
import unicodedata
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Reconfigure stdout/stderr for UTF-8 on Windows environments
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Ordinal Label Mapping: 0 = Little/No Damage, 1 = Mild Damage, 2 = Severe Damage
LABEL_MAP = {
    'little_or_no_damage': 0,
    'mild_damage': 1,
    'severe_damage': 2
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

# Mapping for CP1252 / ISO-8859-1 byte recovery
CP1252_TO_BYTE = {}
for b in range(256):
    try:
        ch = bytes([b]).decode('cp1252')
        CP1252_TO_BYTE[ch] = b
    except Exception:
        pass
for b in range(0x80, 0xA0):
    ch = chr(b)
    if ch not in CP1252_TO_BYTE:
        CP1252_TO_BYTE[ch] = b

COMMON_MOJIBAKE = {
    'â€™': "'",
    'â€˜': "'",
    'â€œ': '"',
    'â€\x9d': '"',
    'â€\x9c': '"',
    'â€“': '–',
    'â€”': '—',
    'â€¦': '...',
    'Â ': ' ',
    'Â': '',
    'âš ï¸\x8f': '⚠️',
    'âš\xa0ï¸\x8f': '⚠️',
    'âš\xa0': '⚠️',
    'â\x9d¤ï¸\x8f': '❤️',
    'âœ¨': '✨',
    'âœŒ': '✌️',
    'â€¼ï¸\x8f': '‼️',
    'âœ”ï¸\x8f': '✔️',
}

def clean_mojibake_and_unicode(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = html.unescape(text)
    for broken, fixed in COMMON_MOJIBAKE.items():
        if broken in text:
            text = text.replace(broken, fixed)
    if 'â' in text or 'Ã' in text or 'Â' in text or any(0x80 <= ord(c) <= 0x9F for c in text):
        byte_arr = bytearray()
        can_convert = True
        for c in text:
            if ord(c) < 128:
                byte_arr.append(ord(c))
            elif c in CP1252_TO_BYTE:
                byte_arr.append(CP1252_TO_BYTE[c])
            elif ord(c) < 256:
                byte_arr.append(ord(c))
            else:
                can_convert = False
                break
        if can_convert:
            try:
                text = byte_arr.decode('utf-8')
            except Exception:
                pass
    text = re.sub(r'[\x80-\x9f]', '', text)
    return unicodedata.normalize('NFKC', text)

def split_camel_case_hashtag(match: re.Match) -> str:
    tag = match.group(1)
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', tag)

def preprocess_bertweet(text: str) -> str:
    text = clean_mojibake_and_unicode(text)
    text = re.sub(r'^(?:RT|rt)[\s:]+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', 'HTTPURL', text)
    text = re.sub(r'@[\w_]+', '@USER', text)
    text = re.sub(r'#(\w+)', split_camel_case_hashtag, text)
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_tfidf(text: str) -> str:
    text = clean_mojibake_and_unicode(text)
    text = re.sub(r'^(?:RT|rt)[\s:]+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'@[\w_]+', ' ', text)
    text = re.sub(r'#(\w+)', split_camel_case_hashtag, text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def is_empty_or_trivial(bertweet_text: str) -> bool:
    residual = re.sub(r'\b(HTTPURL|@USER)\b', '', bertweet_text)
    residual = re.sub(r'[^a-zA-Z0-9]', '', residual)
    return len(residual) == 0

def load_and_convert_rgb(image_path: str) -> Image.Image:
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            return img.convert('RGB')
        return img.copy()

def compute_3class_weights(y: np.ndarray, num_classes: int = 3) -> dict:
    """Computes balanced weights w_c = N / (C * N_c) for the 3 severity levels."""
    n = len(y)
    weights = {}
    for c in range(num_classes):
        n_c = np.sum(y == c)
        w = float(round(n / (num_classes * n_c), 4)) if n_c > 0 else 1.0
        weights[c] = w
    return weights

class CrisisMMDTask3Dataset(Dataset):
    """PyTorch Dataset for Task 3 (Damage Severity Assessment Multimodal Training)."""
    def __init__(self, tsv_file: str, data_root: str = 'CrisisMMD_v2.0', transform=None):
        self.df = pd.read_csv(tsv_file, sep='\t')
        self.data_root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_rel = row['image']
        img_full = os.path.join(self.data_root, img_rel)

        image = load_and_convert_rgb(img_full)
        if self.transform:
            image = self.transform(image)

        label_id = int(row['label_id'])
        sample_w = float(row.get('sample_weight', 1.0))

        return {
            'image': image,
            'text_bertweet': str(row['clean_text_bertweet']),
            'text_tfidf': str(row['clean_text_tfidf']),
            'label_id': torch.tensor(label_id, dtype=torch.long),
            'sample_weight': torch.tensor(sample_w, dtype=torch.float),
            'tweet_id': str(row['tweet_id']),
            'image_id': str(row['image_id'])
        }

def main():
    parser = argparse.ArgumentParser(description="Preprocess CrisisMMD v2.0 Task 3 (Damage Severity Assessment)")
    parser.add_argument('--data_root', type=str, default='CrisisMMD_v2.0',
                        help='Root directory of CrisisMMD dataset')
    parser.add_argument('--split_dir', type=str, 
                        default='CrisisMMD_v2.0/crisismmd_datasplit_all/crisismmd_datasplit_all',
                        help='Directory containing official Task 3 TSV split files')
    parser.add_argument('--output_dir', type=str, default='preprocessed_task3',
                        help='Directory to save Task 3 preprocessed files')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible resampling')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 80)
    print(" CrisisMMD v2.0 - Task 3: Damage Severity Assessment Preprocessing Pipeline")
    print("=" * 80)
    print(f"Source Split Directory: {args.split_dir}")
    print(f"Output Directory:       {args.output_dir}")
    print(f"Random Seed:            {args.seed}\n")

    split_names = ['train', 'dev', 'test']
    processed_dfs = {}
    stats = {
        'label_map': LABEL_MAP,
        'row_counts': {},
        'unique_tweets': {},
        'class_distribution': {},
    }

    for s in split_names:
        tsv_path = os.path.join(args.split_dir, f'task_damage_text_img_{s}.tsv')
        if not os.path.exists(tsv_path):
            raise FileNotFoundError(f"Missing Task 3 split file: {tsv_path}")
            
        print(f"Processing Task 3 split: '{s}' from {tsv_path} ...")
        df = pd.read_csv(tsv_path, sep='\t', encoding='utf-8')

        # 1. Text Preprocessing
        df['clean_text_bertweet'] = df['tweet_text'].apply(preprocess_bertweet)
        df['clean_text_tfidf'] = df['tweet_text'].apply(preprocess_tfidf)
        df['is_empty'] = df['clean_text_bertweet'].apply(is_empty_or_trivial)
        df['token_count_bertweet'] = df['clean_text_bertweet'].apply(lambda t: len(t.split()))
        df['word_count_tfidf'] = df['clean_text_tfidf'].apply(lambda t: len(t.split()))

        # 2. Ordinal 3-Class Label Encoding
        df['label_id'] = df['label'].map(LABEL_MAP).astype(int)
        df['split'] = s

        # 3. Image Path Verification
        df['image_full_path'] = df['image'].apply(lambda rel: os.path.join(args.data_root, rel).replace('\\', '/'))
        df['image_exists'] = df['image_full_path'].apply(os.path.exists)
        assert df['image_exists'].all(), f"Missing images detected in Task 3 {s} split!"

        stats['row_counts'][s] = len(df)
        stats['unique_tweets'][s] = df['tweet_id'].nunique()
        stats['class_distribution'][s] = df['label'].value_counts().to_dict()

        processed_dfs[s] = df

    # 4. Compute 3-Class Weights on Training Split (Concept 1: Weighted)
    y_train = processed_dfs['train']['label_id'].values
    damage_weights = compute_3class_weights(y_train)

    stats['class_weights_training'] = {
        'weights_dict': damage_weights,
        'pytorch_weight_tensor': [damage_weights[i] for i in range(3)],
        'class_names': [ID_TO_LABEL[i] for i in range(3)]
    }

    # Attach sample weights to train dataframe
    processed_dfs['train']['sample_weight'] = processed_dfs['train']['label_id'].map(damage_weights)
    for s in ['dev', 'test']:
        processed_dfs[s]['sample_weight'] = 1.0

    # Combined dataset (3,526 rows with sample_weight)
    combined_df = pd.concat([processed_dfs['train'], processed_dfs['dev'], processed_dfs['test']], ignore_index=True)

    # 5. Save Standard Multimodal & Dev/Test Files
    for s in split_names:
        out_mm_tsv = os.path.join(args.output_dir, f'task3_multimodal_{s}.tsv')
        processed_dfs[s].to_csv(out_mm_tsv, sep='\t', index=False, encoding='utf-8')

    # Concept 1: Full Weighted Training File (2,468 rows)
    out_train_weighted_tsv = os.path.join(args.output_dir, 'task3_multimodal_train_weighted.tsv')
    processed_dfs['train'].to_csv(out_train_weighted_tsv, sep='\t', index=False, encoding='utf-8')
    print(f"\n[CONCEPT 1] Saved Class-Weighted training set to: {out_train_weighted_tsv} (2,468 rows)")

    # =========================================================================
    # CONCEPT 2: Random Oversampling (100% Balanced 3-Class Training Set)
    # =========================================================================
    max_count = processed_dfs['train']['label_id'].value_counts().max() # 1,548
    train_oversampled_parts = [
        processed_dfs['train'][processed_dfs['train']['label_id'] == c].sample(n=max_count, replace=True, random_state=args.seed)
        for c in range(3)
    ]
    train_oversampled = pd.concat(train_oversampled_parts).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    out_train_oversampled_tsv = os.path.join(args.output_dir, 'task3_multimodal_train_oversampled.tsv')
    train_oversampled.to_csv(out_train_oversampled_tsv, sep='\t', index=False, encoding='utf-8')
    print(f"[CONCEPT 2] Saved Oversampled training set to: {out_train_oversampled_tsv} ({len(train_oversampled)} rows, 1,548/class)")

    # =========================================================================
    # CONCEPT 3: Random Undersampling (1:1:1 Balanced Subset)
    # =========================================================================
    min_count = processed_dfs['train']['label_id'].value_counts().min() # 333
    train_undersampled_parts = [
        processed_dfs['train'][processed_dfs['train']['label_id'] == c].sample(n=min_count, replace=False, random_state=args.seed)
        for c in range(3)
    ]
    train_undersampled = pd.concat(train_undersampled_parts).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    out_train_undersampled_tsv = os.path.join(args.output_dir, 'task3_multimodal_train_undersampled.tsv')
    train_undersampled.to_csv(out_train_undersampled_tsv, sep='\t', index=False, encoding='utf-8')
    print(f"[CONCEPT 3] Saved Undersampled training set to: {out_train_undersampled_tsv} ({len(train_undersampled)} rows, 333/class)")

    # 6. Save Text-Only Splits
    text_cols = ['event_name', 'tweet_id', 'tweet_text', 'clean_text_bertweet', 'clean_text_tfidf',
                 'label', 'label_id', 'sample_weight', 'split', 'is_empty',
                 'token_count_bertweet', 'word_count_tfidf']
    for s in split_names:
        out_txt_tsv = os.path.join(args.output_dir, f'task3_text_{s}.tsv')
        processed_dfs[s][text_cols].to_csv(out_txt_tsv, sep='\t', index=False, encoding='utf-8')

    # Text Oversampled
    out_text_oversampled_tsv = os.path.join(args.output_dir, 'task3_text_train_oversampled.tsv')
    train_oversampled[text_cols].to_csv(out_text_oversampled_tsv, sep='\t', index=False, encoding='utf-8')

    # Unique Tweets
    unique_tweets_df = combined_df.drop_duplicates(subset=['tweet_id'])[text_cols].copy()
    out_unique_tsv = os.path.join(args.output_dir, 'task3_text_unique_tweets.tsv')
    unique_tweets_df.to_csv(out_unique_tsv, sep='\t', index=False, encoding='utf-8')

    # Combined All File
    out_all_mm_tsv = os.path.join(args.output_dir, 'task3_multimodal_all.tsv')
    combined_df.to_csv(out_all_mm_tsv, sep='\t', index=False, encoding='utf-8')

    stats['resampling_concepts'] = {
        'concept_1_weighted': {'file': 'task3_multimodal_train_weighted.tsv', 'rows': int(len(processed_dfs['train']))},
        'concept_2_oversampled': {'file': 'task3_multimodal_train_oversampled.tsv', 'rows': int(len(train_oversampled)), 'per_class': int(max_count)},
        'concept_3_undersampled': {'file': 'task3_multimodal_train_undersampled.tsv', 'rows': int(len(train_undersampled)), 'per_class': int(min_count)}
    }

    # Save summary metadata JSON
    summary_path = os.path.join(args.output_dir, 'task3_preprocessing_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)
    print(f"\nSaved Task 3 preprocessing summary to: {summary_path}")

    # Print summary tables to console
    print("\n" + "=" * 80)
    print(" TASK 3: DAMAGE SEVERITY ASSESSMENT - CLASS DISTRIBUTIONS")
    print("=" * 80)
    print(f"{'Split':<8} | {'Total Rows':<11} | {'Little/No Damage (0)':<22} | {'Mild Damage (1)':<18} | {'Severe Damage (2)':<18}")
    print("-" * 88)
    for s in split_names:
        cnts = stats['class_distribution'][s]
        c0 = cnts.get('little_or_no_damage', 0)
        c1 = cnts.get('mild_damage', 0)
        c2 = cnts.get('severe_damage', 0)
        tot = stats['row_counts'][s]
        print(f"{s:<8} | {tot:<11} | {c0:<7} ({c0/tot*100:.1f}%)       | {c1:<6} ({c1/tot*100:.1f}%)     | {c2:<6} ({c2/tot*100:.1f}%)")
    print("-" * 88)
    all_cnts = combined_df['label'].value_counts()
    a0 = all_cnts.get('little_or_no_damage', 0)
    a1 = all_cnts.get('mild_damage', 0)
    a2 = all_cnts.get('severe_damage', 0)
    print(f"{'TOTAL':<8} | {len(combined_df):<11} | {a0:<7} ({a0/len(combined_df)*100:.1f}%)       | {a1:<6} ({a1/len(combined_df)*100:.1f}%)     | {a2:<6} ({a2/len(combined_df)*100:.1f}%)")
    print("=" * 88)

    print("\nTRAINING LOSS WEIGHTS (Concept 1):")
    print(f"  Class 0 (little_or_no_damage): {damage_weights[0]:.4f} (Count: 333)")
    print(f"  Class 1 (mild_damage):         {damage_weights[1]:.4f} (Count: 587)")
    print(f"  Class 2 (severe_damage):       {damage_weights[2]:.4f} (Count: 1548)")
    print(f"  PyTorch Loss: criterion = nn.CrossEntropyLoss(weight=torch.tensor([2.4705, 1.4015, 0.5314]))")

    print("\n" + "=" * 80)
    print(" Task 3 Preprocessing Completed Successfully!")
    print("=" * 80)

if __name__ == '__main__':
    main()
