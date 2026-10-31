#!/usr/bin/env python3
"""
CrisisMMD v2.0 - Task 2: Humanitarian Categories Preprocessing Pipeline
=======================================================================
This script performs comprehensive multimodal preprocessing for Task 2 (8 Humanitarian Categories):
1. affected_individuals
2. infrastructure_and_utility_damage
3. injured_or_dead_people
4. missing_or_found_people
5. not_humanitarian
6. other_relevant_information
7. rescue_volunteering_or_donation_effort
8. vehicle_damage

Implements TWO distinct imbalance-handling preprocessing paradigms for model comparison:
-----------------------------------------------------------------------------------------
Concept 1: Class-Weighted Training (Cost-sensitive multi-task loss on 13,608 raw train samples)
Concept 2: Random Oversampling (100% perfectly balanced 8-class training sets):
  - Text-oversampled: 38,536 rows (4,817 per class = 12.5% each)
  - Image-oversampled: 52,520 rows (6,565 per class = 12.5% each)
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

LABEL_MAP = {
    'affected_individuals': 0,
    'infrastructure_and_utility_damage': 1,
    'injured_or_dead_people': 2,
    'missing_or_found_people': 3,
    'not_humanitarian': 4,
    'other_relevant_information': 5,
    'rescue_volunteering_or_donation_effort': 6,
    'vehicle_damage': 7
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

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

def compute_8class_weights(y: np.ndarray, num_classes: int = 8) -> dict:
    n = len(y)
    weights = {}
    for c in range(num_classes):
        n_c = np.sum(y == c)
        w = float(round(n / (num_classes * n_c), 4)) if n_c > 0 else 1.0
        weights[c] = w
    return weights

def random_oversample_dataframe(df: pd.DataFrame, label_col: str, random_state: int = 42) -> pd.DataFrame:
    """Oversamples all classes with replacement up to the maximum class count."""
    max_count = df[label_col].value_counts().max()
    classes = df[label_col].unique()
    sampled_parts = []
    for c in sorted(classes):
        c_df = df[df[label_col] == c]
        sampled = c_df.sample(n=max_count, replace=True, random_state=random_state)
        sampled_parts.append(sampled)
    balanced_df = pd.concat(sampled_parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return balanced_df

class CrisisMMDTask2Dataset(Dataset):
    """PyTorch Dataset for Task 2 (Humanitarian Categories Multimodal Training)."""
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

        return {
            'image': image,
            'text_bertweet': str(row['clean_text_bertweet']),
            'text_tfidf': str(row['clean_text_tfidf']),
            'label_text_id': torch.tensor(int(row['label_text_id']), dtype=torch.long),
            'label_image_id': torch.tensor(int(row['label_image_id']), dtype=torch.long),
            'label_joint_id': torch.tensor(int(row['label_id']), dtype=torch.long),
            'text_sample_weight': torch.tensor(float(row.get('text_sample_weight', 1.0)), dtype=torch.float),
            'image_sample_weight': torch.tensor(float(row.get('image_sample_weight', 1.0)), dtype=torch.float),
            'tweet_id': str(row['tweet_id']),
            'image_id': str(row['image_id'])
        }

def main():
    parser = argparse.ArgumentParser(description="Preprocess CrisisMMD v2.0 Task 2 (Weighted vs Oversampled)")
    parser.add_argument('--data_root', type=str, default='CrisisMMD_v2.0',
                        help='Root directory of CrisisMMD dataset')
    parser.add_argument('--split_dir', type=str, 
                        default='CrisisMMD_v2.0/crisismmd_datasplit_all/crisismmd_datasplit_all',
                        help='Directory containing the official Task 2 TSV split files')
    parser.add_argument('--output_dir', type=str, default='preprocessed_task2',
                        help='Directory to save Task 2 preprocessed output files')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible oversampling')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 80)
    print(" CrisisMMD v2.0 - Task 2: Humanitarian Categories Preprocessing Pipeline")
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
        'class_distribution_text': {},
        'class_distribution_image': {},
        'class_distribution_joint': {},
    }

    for s in split_names:
        tsv_path = os.path.join(args.split_dir, f'task_humanitarian_text_img_{s}.tsv')
        if not os.path.exists(tsv_path):
            raise FileNotFoundError(f"Missing Task 2 split file: {tsv_path}")
            
        print(f"Processing Task 2 split: '{s}' from {tsv_path} ...")
        df = pd.read_csv(tsv_path, sep='\t', encoding='utf-8')

        # 1. Text Preprocessing
        df['clean_text_bertweet'] = df['tweet_text'].apply(preprocess_bertweet)
        df['clean_text_tfidf'] = df['tweet_text'].apply(preprocess_tfidf)
        df['is_empty'] = df['clean_text_bertweet'].apply(is_empty_or_trivial)
        df['token_count_bertweet'] = df['clean_text_bertweet'].apply(lambda t: len(t.split()))
        df['word_count_tfidf'] = df['clean_text_tfidf'].apply(lambda t: len(t.split()))

        # 2. 8-Class Label Encoding
        df['label_text_id'] = df['label_text'].map(LABEL_MAP).astype(int)
        df['label_image_id'] = df['label_image'].map(LABEL_MAP).astype(int)
        df['label_id'] = df['label'].map(LABEL_MAP).astype(int)
        df['split'] = s

        # 3. Image Path Verification
        df['image_full_path'] = df['image'].apply(lambda rel: os.path.join(args.data_root, rel).replace('\\', '/'))
        df['image_exists'] = df['image_full_path'].apply(os.path.exists)
        assert df['image_exists'].all(), f"Missing images detected in Task 2 {s} split!"

        stats['row_counts'][s] = len(df)
        stats['unique_tweets'][s] = df['tweet_id'].nunique()
        stats['class_distribution_text'][s] = df['label_text'].value_counts().to_dict()
        stats['class_distribution_image'][s] = df['label_image'].value_counts().to_dict()
        stats['class_distribution_joint'][s] = df['label'].value_counts().to_dict()

        processed_dfs[s] = df

    # Combined dataset (18,082 rows)
    combined_df = pd.concat([processed_dfs['train'], processed_dfs['dev'], processed_dfs['test']], ignore_index=True)

    # 4. Compute 8-Class Weights on Training Split (Concept 1: Weighted)
    y_train_text = processed_dfs['train']['label_text_id'].values
    y_train_image = processed_dfs['train']['label_image_id'].values
    y_train_joint = processed_dfs['train']['label_id'].values

    text_weights = compute_8class_weights(y_train_text)
    image_weights = compute_8class_weights(y_train_image)
    joint_weights = compute_8class_weights(y_train_joint)

    stats['class_weights_training'] = {
        'text_head_weights': text_weights,
        'image_head_weights': image_weights,
        'joint_head_weights': joint_weights,
        'pytorch_text_weight_tensor': [text_weights[i] for i in range(8)],
        'pytorch_image_weight_tensor': [image_weights[i] for i in range(8)]
    }

    # Attach weights to train dataframe
    processed_dfs['train']['text_sample_weight'] = processed_dfs['train']['label_text_id'].map(text_weights)
    processed_dfs['train']['image_sample_weight'] = processed_dfs['train']['label_image_id'].map(image_weights)

    # Dev and Test have neutral weights
    for s in ['dev', 'test']:
        processed_dfs[s]['text_sample_weight'] = 1.0
        processed_dfs[s]['image_sample_weight'] = 1.0

    # 5. Save Standard Multimodal & Dev/Test Datasets
    for s in split_names:
        out_mm_tsv = os.path.join(args.output_dir, f'task2_multimodal_{s}.tsv')
        processed_dfs[s].to_csv(out_mm_tsv, sep='\t', index=False, encoding='utf-8')

    # Concept 1 File: Full Weighted Training File (13,608 rows)
    out_train_weighted_tsv = os.path.join(args.output_dir, 'task2_multimodal_train_weighted.tsv')
    processed_dfs['train'].to_csv(out_train_weighted_tsv, sep='\t', index=False, encoding='utf-8')
    print(f"\n[CONCEPT 1] Saved Class-Weighted training set to: {out_train_weighted_tsv} (13,608 rows)")

    # =========================================================================
    # CONCEPT 2: Random Oversampling (100% Balanced 8-Class Datasets)
    # =========================================================================
    # A. Text-Oversampled Multimodal Training Set (4,817 per class -> 38,536 rows)
    train_mm_oversampled = random_oversample_dataframe(processed_dfs['train'], label_col='label_text_id', random_state=args.seed)
    out_mm_oversampled_tsv = os.path.join(args.output_dir, 'task2_multimodal_train_oversampled.tsv')
    train_mm_oversampled.to_csv(out_mm_oversampled_tsv, sep='\t', index=False, encoding='utf-8')
    print(f"[CONCEPT 2] Saved Text-Oversampled multimodal training set to: {out_mm_oversampled_tsv} ({len(train_mm_oversampled)} rows, 4,817/class)")

    # B. Text-Only Oversampled Training Set
    text_cols = ['event_name', 'tweet_id', 'tweet_text', 'clean_text_bertweet', 'clean_text_tfidf',
                 'label_text', 'label_text_id', 'split', 'is_empty',
                 'token_count_bertweet', 'word_count_tfidf']
    train_text_oversampled = train_mm_oversampled[text_cols].copy()
    out_text_oversampled_tsv = os.path.join(args.output_dir, 'task2_text_train_oversampled.tsv')
    train_text_oversampled.to_csv(out_text_oversampled_tsv, sep='\t', index=False, encoding='utf-8')
    print(f"[CONCEPT 2] Saved Text-Oversampled text-only training set to: {out_text_oversampled_tsv} ({len(train_text_oversampled)} rows, 4,817/class)")

    # C. Image-Oversampled Training Set (6,565 per class -> 52,520 rows)
    train_img_oversampled = random_oversample_dataframe(processed_dfs['train'], label_col='label_image_id', random_state=args.seed)
    out_img_oversampled_tsv = os.path.join(args.output_dir, 'task2_image_train_oversampled.tsv')
    train_img_oversampled.to_csv(out_img_oversampled_tsv, sep='\t', index=False, encoding='utf-8')
    print(f"[CONCEPT 2] Saved Image-Oversampled training set to: {out_img_oversampled_tsv} ({len(train_img_oversampled)} rows, 6,565/class)")

    # 6. Save Standard Text-Only Splits
    for s in split_names:
        out_txt_tsv = os.path.join(args.output_dir, f'task2_text_{s}.tsv')
        processed_dfs[s][text_cols].to_csv(out_txt_tsv, sep='\t', index=False, encoding='utf-8')

    # Deduplicated unique tweets
    unique_tweets_df = combined_df.drop_duplicates(subset=['tweet_id'])[text_cols].copy()
    out_unique_tsv = os.path.join(args.output_dir, 'task2_text_unique_tweets.tsv')
    unique_tweets_df.to_csv(out_unique_tsv, sep='\t', index=False, encoding='utf-8')

    # Combined All File
    out_all_mm_tsv = os.path.join(args.output_dir, 'task2_multimodal_all.tsv')
    combined_df.to_csv(out_all_mm_tsv, sep='\t', index=False, encoding='utf-8')

    # Record oversampling statistics in metadata
    stats['oversampling_concept_2'] = {
        'text_oversampled_file': 'task2_text_train_oversampled.tsv',
        'text_multimodal_oversampled_file': 'task2_multimodal_train_oversampled.tsv',
        'text_oversampled_rows': len(train_mm_oversampled),
        'text_samples_per_class': 4817,
        'image_oversampled_file': 'task2_image_train_oversampled.tsv',
        'image_oversampled_rows': len(train_img_oversampled),
        'image_samples_per_class': 6565,
    }

    # Save summary metadata JSON
    summary_path = os.path.join(args.output_dir, 'task2_preprocessing_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)
    print(f"\nSaved Task 2 preprocessing summary to: {summary_path}")

    # Summary table
    print("\n" + "=" * 85)
    print(" TASK 2 PREPROCESSING COMPARISON (WEIGHTED vs OVERSAMPLED)")
    print("=" * 85)
    print(f"{'Concept':<25} | {'Training File':<38} | {'Rows':<8} | {'Distribution'}")
    print("-" * 85)
    print(f"{'Concept 1 (Weighted)':<25} | {'task2_multimodal_train_weighted.tsv':<38} | {'13,608':<8} | Natural (cost-sensitive weights)")
    print(f"{'Concept 2 (Text Oversample)':<25} | {'task2_text_train_oversampled.tsv':<38} | {'38,536':<8} | Exactly 4,817 per class (12.5%)")
    print(f"{'Concept 2 (MM Oversample)':<25} | {'task2_multimodal_train_oversampled.tsv':<38} | {'38,536':<8} | Exactly 4,817 per class (12.5%)")
    print(f"{'Concept 2 (Image Oversample)':<25} | {'task2_image_train_oversampled.tsv':<38} | {'52,520':<8} | Exactly 6,565 per class (12.5%)")
    print("=" * 85)

if __name__ == '__main__':
    main()
