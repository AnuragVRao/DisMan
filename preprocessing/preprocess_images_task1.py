#!/usr/bin/env python3
"""
CrisisMMD v2.0 - Task 1 (Informative vs Not Informative) Image Preprocessing Pipeline
=====================================================================================
This module handles comprehensive image validation, preprocessing, augmentation,
and PyTorch Dataset / DataLoader integration for Task 1.

Processing steps implemented:
- Missing and corrupt image check (confirms every image exists and opens cleanly)
- Color mode inspection & RGB conversion (handles RGBA, grayscale L, palette P -> RGB)
- Image resizing (224x224 for ResNet50, ViT, CLIP, DINOv2)
- Normalization (ImageNet mean/std and CLIP mean/std options)
- Training Data Augmentations (RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter)
- Validation/Test Deterministic Transforms (Resize 224x224, ToTensor, Normalize)
- Exact Duplicate Image Hash Check (across train, dev, and test splits)
- Label Encoding (informative -> 1, not_informative -> 0 for both text & image)
- Image Class Balance Analysis & Class Weight Calculation
- Text-Image Label Alignment Analysis (Agreement vs Disagreement)
- Multimodal Option 1: Full dataset with multi-task loss class weights (task1_multimodal_train_weighted.tsv)
- Multimodal Option 2: 50/50 text-balanced training set via undersampling (task1_multimodal_train_balanced.tsv)
- PyTorch Dataset classes (CrisisMMDImageDataset & CrisisMMDMultimodalDataset)
- Offline Image Pre-resizing & Caching utility (preprocessed_images_task1/images_224/)
"""

import os
import sys
import json
import hashlib
import argparse
from collections import Counter
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

# Normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


def get_image_transforms(target_size=(224, 224), norm_type='imagenet', is_train=True):
    """
    Returns torchvision transforms for train or evaluation.
    Train: Data augmentations (flip, rotation, random crop, color jitter) + normalize.
    Eval:  Deterministic resize (224x224) + normalize.
    """
    mean = IMAGENET_MEAN if norm_type.lower() == 'imagenet' else CLIP_MEAN
    std  = IMAGENET_STD  if norm_type.lower() == 'imagenet' else CLIP_STD

    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


def load_and_convert_rgb(image_path: str) -> Image.Image:
    """
    Safely opens an image file and converts any color mode (RGBA, L, P) to RGB.
    """
    with Image.open(image_path) as img:
        if img.mode != 'RGB':
            return img.convert('RGB')
        return img.copy()


class CrisisMMDImageDataset(Dataset):
    """
    PyTorch Dataset for Vision-only models (ResNet50, ViT, Swin, etc.).
    Returns: (image_tensor, label_image_id)
    """
    def __init__(self, tsv_file: str, data_root: str = 'CrisisMMD_v2.0', 
                 transform=None, label_col: str = 'label_image_id'):
        self.df = pd.read_csv(tsv_file, sep='\t')
        self.data_root = data_root
        self.transform = transform
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_rel_path = row['image']
        img_full_path = os.path.join(self.data_root, img_rel_path)
        
        # Load and convert to RGB
        image = load_and_convert_rgb(img_full_path)
        
        if self.transform:
            image = self.transform(image)
            
        label = int(row[self.label_col])
        return image, torch.tensor(label, dtype=torch.long)


class CrisisMMDMultimodalDataset(Dataset):
    """
    PyTorch Dataset for Multimodal Fusion models (BERTweet + ResNet/ViT).
    Returns a dictionary with:
    - image: Normalized RGB image tensor [3, 224, 224]
    - text_bertweet: Preprocessed BERTweet clean text string
    - text_tfidf: Preprocessed TF-IDF clean text string
    - label_image_id: Image informativeness label (0 or 1)
    - label_text_id: Text informativeness label (0 or 1)
    - label_agree: Whether text and image labels agree (1 = Positive, 0 = Negative)
    - text_weight: Text loss weight for this sample
    - image_weight: Image loss weight for this sample
    - tweet_id: Unique tweet identifier
    - image_id: Unique tweet image identifier
    """
    def __init__(self, tsv_file: str, data_root: str = 'CrisisMMD_v2.0', transform=None):
        self.df = pd.read_csv(tsv_file, sep='\t')
        self.data_root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_rel_path = row['image']
        img_full_path = os.path.join(self.data_root, img_rel_path)
        
        image = load_and_convert_rgb(img_full_path)
        if self.transform:
            image = self.transform(image)

        label_img = int(row['label_image_id'])
        label_txt = int(row['label_text_id'])
        label_agree = 1 if row.get('label_text_image', '') == 'Positive' else 0
        
        txt_w = float(row.get('text_sample_weight', 1.0))
        img_w = float(row.get('image_sample_weight', 1.0))

        return {
            'image': image,
            'text_bertweet': str(row['clean_text_bertweet']),
            'text_tfidf': str(row['clean_text_tfidf']),
            'label_image_id': torch.tensor(label_img, dtype=torch.long),
            'label_text_id': torch.tensor(label_txt, dtype=torch.long),
            'label_agree': torch.tensor(label_agree, dtype=torch.long),
            'text_weight': torch.tensor(txt_w, dtype=torch.float),
            'image_weight': torch.tensor(img_w, dtype=torch.float),
            'tweet_id': str(row['tweet_id']),
            'image_id': str(row['image_id'])
        }


def scan_and_verify_images(data_root: str, split_dir: str):
    """
    Scans all 18,082 images across train, dev, and test.
    Checks: missing files, corrupt files, color modes, and MD5 duplicate hashes.
    """
    splits = ['train', 'dev', 'test']
    split_dfs = {}
    split_hashes = {}
    missing_files = []
    corrupt_files = []
    color_modes = Counter()
    image_shapes = []

    print("=" * 80)
    print(" SCANNING & VERIFYING CRISISMMD V2.0 IMAGES")
    print("=" * 80)

    for s in splits:
        tsv_path = os.path.join(split_dir, f'task_informative_text_img_{s}.tsv')
        df = pd.read_csv(tsv_path, sep='\t', encoding='utf-8')
        split_dfs[s] = df
        split_hashes[s] = {}
        print(f"Scanning split '{s}' ({len(df)} rows)...")

        for _, row in df.iterrows():
            rel_img = row['image']
            full_path = os.path.join(data_root, rel_img)

            if not os.path.exists(full_path):
                missing_files.append((s, rel_img, full_path))
                continue

            try:
                with Image.open(full_path) as img:
                    img.verify()
                with Image.open(full_path) as img:
                    color_modes[img.mode] += 1
                    image_shapes.append(img.size)
            except Exception as e:
                corrupt_files.append((s, rel_img, str(e)))
                continue

            with open(full_path, 'rb') as f:
                h = hashlib.md5(f.read()).hexdigest()
            split_hashes[s][rel_img] = h

    train_h = set(split_hashes['train'].values())
    dev_h   = set(split_hashes['dev'].values())
    test_h  = set(split_hashes['test'].values())

    overlap_train_dev  = len(train_h.intersection(dev_h))
    overlap_train_test = len(train_h.intersection(test_h))
    overlap_dev_test   = len(dev_h.intersection(test_h))

    widths = [s[0] for s in image_shapes]
    heights = [s[1] for s in image_shapes]

    summary = {
        'total_images_verified': len(image_shapes),
        'missing_images_count': len(missing_files),
        'corrupt_images_count': len(corrupt_files),
        'color_modes_distribution': dict(color_modes),
        'dimensions': {
            'min_width': int(min(widths)),
            'max_width': int(max(widths)),
            'mean_width': float(round(np.mean(widths), 1)),
            'min_height': int(min(heights)),
            'max_height': int(max(heights)),
            'mean_height': float(round(np.mean(heights), 1)),
        },
        'content_hash_cross_split_duplicates': {
            'train_unique_hashes': len(train_h),
            'dev_unique_hashes': len(dev_h),
            'test_unique_hashes': len(test_h),
            'train_dev_overlap': overlap_train_dev,
            'train_test_overlap': overlap_train_test,
            'dev_test_overlap': overlap_dev_test,
            'note': 'Image file paths are 100% distinct across splits; hash overlap reflects retweeted identical images.'
        }
    }
    return summary, split_dfs, split_hashes


def compute_class_weights(y: np.ndarray) -> dict:
    """
    Computes balanced class weights: w_j = N / (C * N_j)
    """
    n = len(y)
    n0 = np.sum(y == 0)
    n1 = np.sum(y == 1)
    w0 = float(round(n / (2.0 * n0), 4)) if n0 > 0 else 1.0
    w1 = float(round(n / (2.0 * n1), 4)) if n1 > 0 else 1.0
    return {
        'class_0_weight': w0,
        'class_1_weight': w1,
        'imbalance_ratio_1_to_0': float(round(n1 / n0, 4)) if n0 > 0 else 1.0
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess CrisisMMD v2.0 Task 1 Images & Multimodal Datasets")
    parser.add_argument('--data_root', type=str, default='CrisisMMD_v2.0',
                        help='Root directory of CrisisMMD dataset')
    parser.add_argument('--split_dir', type=str, 
                        default='CrisisMMD_v2.0/crisismmd_datasplit_all/crisismmd_datasplit_all',
                        help='Directory containing the official TSV split files')
    parser.add_argument('--preprocessed_text_dir', type=str, default='preprocessed_text_task1',
                        help='Directory containing preprocessed text TSVs')
    parser.add_argument('--output_dir', type=str, default='preprocessed_images_task1',
                        help='Directory to save image metadata and multimodal datasets')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible undersampling')
    parser.add_argument('--cache_images_224', action='store_true',
                        help='If set, pre-resizes and caches all images to 224x224 RGB JPEG on disk')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Image verification and duplicate hash scan
    img_summary, split_dfs, split_hashes = scan_and_verify_images(args.data_root, args.split_dir)

    print("\n" + "=" * 80)
    print(" IMAGE INTEGRITY & DUPLICATE ANALYSIS RESULTS")
    print("=" * 80)
    print(f"Total Images Verified: {img_summary['total_images_verified']}")
    print(f"Missing Files:         {img_summary['missing_images_count']}")
    print(f"Corrupt Files:         {img_summary['corrupt_images_count']}")
    print(f"Color Modes:           {img_summary['color_modes_distribution']}")
    print(f"Resolutions:           Min=({img_summary['dimensions']['min_width']}x{img_summary['dimensions']['min_height']}), "
          f"Mean=({img_summary['dimensions']['mean_width']}x{img_summary['dimensions']['mean_height']}), "
          f"Max=({img_summary['dimensions']['max_width']}x{img_summary['dimensions']['max_height']})")
    print(f"Hash Duplicates:       Train-Dev={img_summary['content_hash_cross_split_duplicates']['train_dev_overlap']}, "
          f"Train-Test={img_summary['content_hash_cross_split_duplicates']['train_test_overlap']}, "
          f"Dev-Test={img_summary['content_hash_cross_split_duplicates']['dev_test_overlap']}")

    # 2. Alignment with Preprocessed Text Splits
    print("\n" + "=" * 80)
    print(" ALIGNING PREPROCESSED TEXT & IMAGES FOR TASK 1")
    print("=" * 80)
    
    multimodal_dfs = {}
    class_stats_image = {}
    class_stats_text = {}

    for s in ['train', 'dev', 'test']:
        txt_path = os.path.join(args.preprocessed_text_dir, f'task1_text_{s}.tsv')
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Required preprocessed text file missing: {txt_path}")
            
        df = pd.read_csv(txt_path, sep='\t', encoding='utf-8')
        
        # Verify relative image path and absolute image path
        df['image_full_path'] = df['image'].apply(lambda rel: os.path.join(args.data_root, rel).replace('\\', '/'))
        df['image_exists'] = df['image_full_path'].apply(os.path.exists)
        assert df['image_exists'].all(), f"Some images in {s} do not exist!"

        class_stats_image[s] = df['label_image'].value_counts().to_dict()
        class_stats_text[s] = df['label_text'].value_counts().to_dict()
        
        # Standard split
        out_mm_tsv = os.path.join(args.output_dir, f'task1_multimodal_{s}.tsv')
        df.to_csv(out_mm_tsv, sep='\t', index=False, encoding='utf-8')
        print(f"Saved multimodal split '{s}' to: {out_mm_tsv} ({len(df)} rows)")
        multimodal_dfs[s] = df

    # Combined multimodal dataset
    combined_mm = pd.concat([multimodal_dfs['train'], multimodal_dfs['dev'], multimodal_dfs['test']], ignore_index=True)
    out_all_mm_tsv = os.path.join(args.output_dir, 'task1_multimodal_all.tsv')
    combined_mm.to_csv(out_all_mm_tsv, sep='\t', index=False, encoding='utf-8')
    print(f"Saved combined multimodal dataset to: {out_all_mm_tsv} ({len(combined_mm)} rows)")

    # Compute Class Weights for Train Split
    train_df = multimodal_dfs['train'].copy()
    text_weights_train = compute_class_weights(train_df['label_text_id'].values)
    image_weights_train = compute_class_weights(train_df['label_image_id'].values)

    # =========================================================================
    # OPTION 1: Full Training Set with Dual-Head Class Weights
    # =========================================================================
    train_weighted_df = train_df.copy()
    w_text_0 = text_weights_train['class_0_weight']    # 1.7139
    w_text_1 = text_weights_train['class_1_weight']    # 0.7060
    w_img_0  = image_weights_train['class_0_weight']   # 1.0389
    w_img_1  = image_weights_train['class_1_weight']   # 0.9639

    train_weighted_df['text_sample_weight']  = train_weighted_df['label_text_id'].map({0: w_text_0, 1: w_text_1})
    train_weighted_df['image_sample_weight'] = train_weighted_df['label_image_id'].map({0: w_img_0, 1: w_img_1})
    
    out_train_weighted_tsv = os.path.join(args.output_dir, 'task1_multimodal_train_weighted.tsv')
    train_weighted_df.to_csv(out_train_weighted_tsv, sep='\t', index=False, encoding='utf-8')
    print(f"\n[OPTION 1] Saved full class-weighted multimodal training set to: {out_train_weighted_tsv} ({len(train_weighted_df)} rows)")

    # =========================================================================
    # OPTION 2: 50/50 Text-Balanced Multimodal Training Set via Undersampling
    # =========================================================================
    text_not_inf = train_df[train_df['label_text_id'] == 0]
    text_inf = train_df[train_df['label_text_id'] == 1].sample(n=len(text_not_inf), random_state=args.seed)
    train_balanced_df = pd.concat([text_not_inf, text_inf]).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # In this balanced text subset, compute image weights
    img_w_bal = compute_class_weights(train_balanced_df['label_image_id'].values)
    train_balanced_df['text_sample_weight'] = 1.0  # Already 50/50 balanced
    train_balanced_df['image_sample_weight'] = train_balanced_df['label_image_id'].map({0: img_w_bal['class_0_weight'], 1: img_w_bal['class_1_weight']})

    out_train_balanced_tsv = os.path.join(args.output_dir, 'task1_multimodal_train_balanced.tsv')
    train_balanced_df.to_csv(out_train_balanced_tsv, sep='\t', index=False, encoding='utf-8')
    print(f"[OPTION 2] Saved 50/50 text-balanced multimodal training set to: {out_train_balanced_tsv} ({len(train_balanced_df)} rows)")

    # Record summary statistics
    agreement_counts = combined_mm['label_text_image'].value_counts().to_dict()
    contingency = pd.crosstab(combined_mm['label_text'], combined_mm['label_image']).to_dict()

    img_summary['multimodal_option_1_weighted'] = {
        'file': 'task1_multimodal_train_weighted.tsv',
        'total_samples': len(train_weighted_df),
        'text_class_weights': text_weights_train,
        'image_class_weights': image_weights_train,
        'description': 'All 13,608 training rows intact with explicit text and image class weights for dual-head loss'
    }

    img_summary['multimodal_option_2_balanced'] = {
        'file': 'task1_multimodal_train_balanced.tsv',
        'total_samples': len(train_balanced_df),
        'text_distribution': {
            'informative': int((train_balanced_df['label_text_id'] == 1).sum()),
            'not_informative': int((train_balanced_df['label_text_id'] == 0).sum()),
            'ratio': '50.0% / 50.0%'
        },
        'image_distribution_in_subset': {
            'informative': int((train_balanced_df['label_image_id'] == 1).sum()),
            'not_informative': int((train_balanced_df['label_image_id'] == 0).sum()),
            'ratio': f"{round((train_balanced_df['label_image_id'] == 1).mean()*100, 1)}% / {round((train_balanced_df['label_image_id'] == 0).mean()*100, 1)}%"
        },
        'image_class_weights': img_w_bal,
        'description': 'Undersampled text to exact 50/50 balance (3,970 informative + 3,970 not_informative = 7,940 rows)'
    }

    img_summary['text_image_alignment'] = {
        'contingency_table': contingency,
        'agreement_counts': agreement_counts,
        'agreement_rate_percent': float(round(agreement_counts.get('Positive', 0) / len(combined_mm) * 100, 2)),
        'disagreement_rate_percent': float(round(agreement_counts.get('Negative', 0) / len(combined_mm) * 100, 2))
    }

    summary_json_path = os.path.join(args.output_dir, 'image_preprocessing_summary.json')
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(img_summary, f, indent=4)
    print(f"Saved multimodal summary & statistics to: {summary_json_path}")

    # Print summary tables to console
    print("\n" + "=" * 80)
    print(" MULTIMODAL TRAINING OPTIONS (OPTION 1 vs OPTION 2)")
    print("=" * 80)
    print(f"OPTION 1 (WEIGHTED - ALL DATA INTACT):")
    print(f"  File:                      {out_train_weighted_tsv}")
    print(f"  Total Rows:                {len(train_weighted_df)}")
    print(f"  Text Weights (w0, w1):     {w_text_0}, {w_text_1}")
    print(f"  Image Weights (w0, w1):    {w_img_0}, {w_img_1}")
    print("-" * 80)
    print(f"OPTION 2 (TEXT 50/50 BALANCED VIA UNDERSAMPLING):")
    print(f"  File:                      {out_train_balanced_tsv}")
    print(f"  Total Rows:                {len(train_balanced_df)}")
    print(f"  Text Balance:              3,970 inf (50.0%) / 3,970 not_inf (50.0%)")
    print(f"  Image Distribution:        {img_summary['multimodal_option_2_balanced']['image_distribution_in_subset']['informative']} inf / {img_summary['multimodal_option_2_balanced']['image_distribution_in_subset']['not_informative']} not_inf ({img_summary['multimodal_option_2_balanced']['image_distribution_in_subset']['ratio']})")
    print("=" * 80)

    # Pre-caching images if requested
    if args.cache_images_224:
        cache_dir = os.path.join(args.output_dir, 'images_224')
        os.makedirs(cache_dir, exist_ok=True)
        print(f"\nPre-caching resized 224x224 RGB JPEG images into: {cache_dir} ...")
        for idx, row in combined_mm.iterrows():
            rel_path = row['image']
            in_path = os.path.join(args.data_root, rel_path)
            out_img_path = os.path.join(cache_dir, f"{row['image_id']}.jpg")
            if not os.path.exists(out_img_path):
                img = load_and_convert_rgb(in_path)
                img = img.resize((224, 224), Image.Resampling.BICUBIC)
                img.save(out_img_path, "JPEG", quality=95)
        print(f"Finished caching 224x224 images into {cache_dir}")

    print("\n" + "=" * 80)
    print(" Both Option 1 and Option 2 Multimodal Files Ready!")
    print("=" * 80)


if __name__ == '__main__':
    main()
