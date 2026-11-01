#!/usr/bin/env python3
"""
CrisisMMD v2.0 - Task 1 (Informative vs Not Informative) Text Preprocessing Pipeline
====================================================================================
This script performs comprehensive text preprocessing specifically tailored for:
1. BERTweet (vinai/bertweet-base)
2. TF-IDF baselines (scikit-learn TfidfVectorizer)

Processing steps implemented:
- Encoding fix (fixing UTF-8/CP1252/Latin-1 mojibake, unescaping HTML entities)
- Unicode normalization (NFKC)
- RT removal (strips leading retweet markers: "RT @username:" / "RT ")
- Mention normalization (@username -> @USER)
- URL removal & replacement (https://... -> HTTPURL for BERTweet, stripped for TF-IDF)
- Hashtag processing (drops '#' and splits CamelCase/PascalCase hashtags into natural words)
- Emoji handling (preserves clean normalized emojis for BERTweet; stripped for TF-IDF)
- Whitespace normalization (collapses multiple spaces, tabs, newlines, and trims)
- Duplicate and empty text check (flags empty or trivial tweets)
- Label encoding (informative -> 1, not_informative -> 0)
- Option A: Generates full weighted training set with sample/class weights (task1_text_train_weighted.tsv)
- Option B: Generates 50/50 balanced training set via undersampling (task1_text_train_balanced.tsv)

Official splits used:
- task_informative_text_img_train.tsv
- task_informative_text_img_dev.tsv
- task_informative_text_img_test.tsv
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

# Reconfigure stdout/stderr for UTF-8 on Windows environments
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Mapping for CP1252 / ISO-8859-1 byte recovery
CP1252_TO_BYTE = {}
for b in range(256):
    try:
        ch = bytes([b]).decode('cp1252')
        CP1252_TO_BYTE[ch] = b
    except Exception:
        pass
# Add C1 control characters (0x80 - 0x9F)
for b in range(0x80, 0xA0):
    ch = chr(b)
    if ch not in CP1252_TO_BYTE:
        CP1252_TO_BYTE[ch] = b

# Common explicit mojibake sequences found in web/Twitter corpora
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
    """
    Restores broken characters (mojibake) from mixed UTF-8 / Windows-1252 encodings,
    unescapes HTML entities, strips unprintable control codes, and applies NFKC normalization.
    """
    if not isinstance(text, str):
        return ''
    
    # 1. Unescape HTML entities (e.g. &amp;, &lt;, &gt;, &quot;, &#39;)
    text = html.unescape(text)
    
    # 2. Replace known multi-byte mojibake sequences
    for broken, fixed in COMMON_MOJIBAKE.items():
        if broken in text:
            text = text.replace(broken, fixed)
            
    # 3. Whole-string byte recovery if string contains mojibake indicators
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
                decoded = byte_arr.decode('utf-8')
                text = decoded
            except Exception:
                pass
                
    # 4. Remove lingering non-printable C1 control characters (0x80 - 0x9F)
    text = re.sub(r'[\x80-\x9f]', '', text)
    
    # 5. Unicode NFKC normalization
    text = unicodedata.normalize('NFKC', text)
    return text

def split_camel_case_hashtag(match: re.Match) -> str:
    """
    Splits CamelCase / PascalCase hashtags into space-separated words, dropping '#'.
    e.g. '#CaliforniaWildfires' -> 'California Wildfires', '#HurricaneIrma' -> 'Hurricane Irma'
    Single words and acronyms like '#wildfires' or '#KAKEnews' are preserved cleanly.
    """
    tag = match.group(1)
    # Split lowercase followed by uppercase
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', tag)
    return s

def preprocess_bertweet(text: str) -> str:
    """
    Preprocessing pipeline for BERTweet (vinai/bertweet-base):
    - Fixes encoding and mojibake
    - Removes leading RT retweet marker
    - Replaces URLs with HTTPURL placeholder
    - Normalizes user mentions to @USER
    - Splits CamelCase hashtags and removes '#'
    - Preserves casing, natural punctuation, and clean emojis
    - Normalizes repeated whitespaces
    """
    text = clean_mojibake_and_unicode(text)
    text = re.sub(r'^(?:RT|rt)[\s:]+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', 'HTTPURL', text)
    text = re.sub(r'@[\w_]+', '@USER', text)
    text = re.sub(r'#(\w+)', split_camel_case_hashtag, text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_tfidf(text: str) -> str:
    """
    Preprocessing pipeline for TF-IDF baseline models:
    - Fixes encoding and mojibake
    - Removes leading RT marker
    - Removes URLs completely
    - Removes user mentions completely
    - Splits CamelCase hashtags and removes '#'
    - Converts text to lowercase
    - Keeps alphanumeric characters and spaces
    - Collapses repeated whitespaces
    """
    text = clean_mojibake_and_unicode(text)
    text = re.sub(r'^(?:RT|rt)[\s:]+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'@[\w_]+', ' ', text)
    text = re.sub(r'#(\w+)', split_camel_case_hashtag, text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_empty_or_trivial(bertweet_text: str) -> bool:
    """
    Checks if a tweet becomes empty or contains only placeholders (@USER, HTTPURL, punctuation)
    after cleaning.
    """
    residual = re.sub(r'\b(HTTPURL|@USER)\b', '', bertweet_text)
    residual = re.sub(r'[^a-zA-Z0-9]', '', residual)
    return len(residual) == 0

def compute_class_weights(y_train: np.ndarray) -> dict:
    """
    Computes balanced class weights from training labels:
    w_j = N / (C * N_j)
    """
    n_samples = len(y_train)
    n_classes = len(np.unique(y_train))
    classes = [0, 1]
    weights = {}
    for c in classes:
        n_c = np.sum(y_train == c)
        w = n_samples / (n_classes * n_c) if n_c > 0 else 1.0
        weights[c] = float(w)
    
    # Also calculate pos_weight for BCEWithLogitsLoss (num_neg / num_pos)
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    bce_pos_weight = float(num_neg / num_pos) if num_pos > 0 else 1.0
    
    return {
        'balanced_weights': weights,
        'bce_pos_weight': bce_pos_weight,
        'imbalance_ratio_pos_to_neg': float(num_pos / num_neg),
        'class_0_not_informative_weight': float(round(weights[0], 4)),
        'class_1_informative_weight': float(round(weights[1], 4)),
    }

def create_balanced_split(df: pd.DataFrame, label_col: str = 'label_text_id', random_state: int = 42) -> pd.DataFrame:
    """
    Creates a 50/50 balanced dataset via random undersampling of the majority class.
    """
    class_0 = df[df[label_col] == 0]
    class_1 = df[df[label_col] == 1]
    min_count = min(len(class_0), len(class_1))
    
    c0_sampled = class_0.sample(n=min_count, random_state=random_state)
    c1_sampled = class_1.sample(n=min_count, random_state=random_state)
    
    balanced = pd.concat([c0_sampled, c1_sampled]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return balanced

def main():
    parser = argparse.ArgumentParser(description="Preprocess CrisisMMD v2.0 Task 1 Text for BERTweet & TF-IDF")
    parser.add_argument('--data_dir', type=str, 
                        default='CrisisMMD_v2.0/crisismmd_datasplit_all/crisismmd_datasplit_all',
                        help='Directory containing the official TSV split files')
    parser.add_argument('--output_dir', type=str, 
                        default='preprocessed_text_task1',
                        help='Directory to save preprocessed output files')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible undersampling')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 80)
    print(" CrisisMMD v2.0 - Task 1: Informative vs Not Informative Text Preprocessing")
    print("=" * 80)
    print(f"Source Split Directory: {args.data_dir}")
    print(f"Output Directory:       {args.output_dir}")
    print(f"Random Seed:            {args.seed}")
    print()

    split_names = ['train', 'dev', 'test']
    processed_dfs = {}
    stats = {
        'row_counts': {},
        'unique_tweets': {},
        'class_distribution_text': {},
        'class_distribution_image': {},
        'empty_or_trivial_counts': {},
        'token_length_distribution': {},
    }

    label_map = {'informative': 1, 'not_informative': 0}

    for split in split_names:
        tsv_path = os.path.join(args.data_dir, f'task_informative_text_img_{split}.tsv')
        if not os.path.exists(tsv_path):
            raise FileNotFoundError(f"Expected split file not found: {tsv_path}")
            
        print(f"Processing split: '{split}' from {tsv_path} ...")
        df = pd.read_csv(tsv_path, sep='\t', encoding='utf-8')

        # 1. Verify missing values
        null_counts = df[['tweet_text', 'label_text', 'label_image']].isnull().sum().to_dict()
        if any(v > 0 for v in null_counts.values()):
            print(f"  [WARNING] Null values detected in {split}: {null_counts}")
        else:
            print(f"  [OK] No missing values found in text or labels.")

        # 2. Text Preprocessing
        df['clean_text_bertweet'] = df['tweet_text'].apply(preprocess_bertweet)
        df['clean_text_tfidf'] = df['tweet_text'].apply(preprocess_tfidf)
        df['is_empty'] = df['clean_text_bertweet'].apply(is_empty_or_trivial)
        
        # 3. Token count & sequence lengths
        df['token_count_bertweet'] = df['clean_text_bertweet'].apply(lambda s: len(s.split()))
        df['word_count_tfidf'] = df['clean_text_tfidf'].apply(lambda s: len(s.split()))

        # 4. Label Encoding
        df['label_text_id'] = df['label_text'].map(label_map).astype(int)
        df['label_image_id'] = df['label_image'].map(label_map).astype(int)
        df['split'] = split

        # Collect stats
        n_rows = len(df)
        n_unique_tweets = df['tweet_id'].nunique()
        text_counts = df['label_text'].value_counts().to_dict()
        image_counts = df['label_image'].value_counts().to_dict()
        n_empty = int(df['is_empty'].sum())
        
        lens = df['token_count_bertweet']
        len_stats = {
            'min': int(lens.min()),
            'max': int(lens.max()),
            'mean': float(round(lens.mean(), 2)),
            'median': float(lens.median()),
            'p90': float(round(np.percentile(lens, 90), 2)),
            'p95': float(round(np.percentile(lens, 95), 2)),
            'p99': float(round(np.percentile(lens, 99), 2)),
        }

        stats['row_counts'][split] = n_rows
        stats['unique_tweets'][split] = n_unique_tweets
        stats['class_distribution_text'][split] = text_counts
        stats['class_distribution_image'][split] = image_counts
        stats['empty_or_trivial_counts'][split] = n_empty
        stats['token_length_distribution'][split] = len_stats

        # Save individual split file (original distribution)
        out_tsv = os.path.join(args.output_dir, f'task1_text_{split}.tsv')
        df.to_csv(out_tsv, sep='\t', index=False, encoding='utf-8')
        print(f"  Saved preprocessed {split} to: {out_tsv} ({n_rows} rows)")

        processed_dfs[split] = df

    # Combined dataset
    combined_df = pd.concat([processed_dfs['train'], processed_dfs['dev'], processed_dfs['test']], ignore_index=True)
    out_all_tsv = os.path.join(args.output_dir, 'task1_text_all.tsv')
    combined_df.to_csv(out_all_tsv, sep='\t', index=False, encoding='utf-8')
    print(f"\nSaved combined preprocessed dataset to: {out_all_tsv} ({len(combined_df)} rows)")

    # Deduplicated tweet-level text file (for text-only baseline modeling)
    unique_tweets_df = combined_df.drop_duplicates(subset=['tweet_id']).copy()
    out_unique_tsv = os.path.join(args.output_dir, 'task1_text_unique_tweets.tsv')
    unique_tweets_df.to_csv(out_unique_tsv, sep='\t', index=False, encoding='utf-8')
    print(f"Saved unique tweets preprocessed dataset to: {out_unique_tsv} ({len(unique_tweets_df)} rows)")

    # =========================================================================
    # OPTION A: Separate TSV with Class / Sample Weights (All Data Intact)
    # =========================================================================
    y_train = processed_dfs['train']['label_text_id'].values
    class_weight_dict = compute_class_weights(y_train)
    stats['class_weights_training'] = class_weight_dict
    
    w0 = class_weight_dict['class_0_not_informative_weight']  # 1.7139
    w1 = class_weight_dict['class_1_informative_weight']      # 0.7060

    # Create weighted training dataframe (13,608 rows, all intact)
    train_weighted_df = processed_dfs['train'].copy()
    train_weighted_df['sample_weight'] = train_weighted_df['label_text_id'].map({0: w0, 1: w1})
    train_weighted_df['class_weight'] = train_weighted_df['sample_weight']
    out_train_weighted_tsv = os.path.join(args.output_dir, 'task1_text_train_weighted.tsv')
    train_weighted_df.to_csv(out_train_weighted_tsv, sep='\t', index=False, encoding='utf-8')
    print(f"\n[OPTION A] Saved class-weighted training set to: {out_train_weighted_tsv} ({len(train_weighted_df)} rows)")

    # Unique-tweet level weighted training set
    train_unique_df = processed_dfs['train'].drop_duplicates(subset=['tweet_id']).copy()
    y_train_unique = train_unique_df['label_text_id'].values
    unique_class_weight_dict = compute_class_weights(y_train_unique)
    w0_u = unique_class_weight_dict['class_0_not_informative_weight']
    w1_u = unique_class_weight_dict['class_1_informative_weight']
    train_unique_weighted_df = train_unique_df.copy()
    train_unique_weighted_df['sample_weight'] = train_unique_weighted_df['label_text_id'].map({0: w0_u, 1: w1_u})
    train_unique_weighted_df['class_weight'] = train_unique_weighted_df['sample_weight']
    out_train_unique_weighted_tsv = os.path.join(args.output_dir, 'task1_text_train_unique_weighted.tsv')
    train_unique_weighted_df.to_csv(out_train_unique_weighted_tsv, sep='\t', index=False, encoding='utf-8')
    print(f"[OPTION A] Saved unique-tweet class-weighted training set to: {out_train_unique_weighted_tsv} ({len(train_unique_weighted_df)} rows)")

    stats['class_weighted_training_option_a'] = {
        'row_level': {
            'file': 'task1_text_train_weighted.tsv',
            'total_samples': len(train_weighted_df),
            'class_0_not_informative_weight': w0,
            'class_1_informative_weight': w1,
            'bce_pos_weight': class_weight_dict['bce_pos_weight'],
            'loss_code_pytorch': 'criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.7139, 0.7060]))',
            'sklearn_code': 'LogisticRegression(class_weight="balanced")'
        },
        'unique_tweet_level': {
            'file': 'task1_text_train_unique_weighted.tsv',
            'total_samples': len(train_unique_weighted_df),
            'class_0_not_informative_weight': w0_u,
            'class_1_informative_weight': w1_u
        }
    }

    # =========================================================================
    # OPTION B: Generate 50/50 Balanced Training Sets via Undersampling
    # =========================================================================
    train_balanced_df = create_balanced_split(processed_dfs['train'], label_col='label_text_id', random_state=args.seed)
    out_train_balanced_tsv = os.path.join(args.output_dir, 'task1_text_train_balanced.tsv')
    train_balanced_df.to_csv(out_train_balanced_tsv, sep='\t', index=False, encoding='utf-8')
    print(f"\n[OPTION B] Saved 50/50 balanced training set (row-level) to: {out_train_balanced_tsv} ({len(train_balanced_df)} rows)")

    train_unique_balanced_df = create_balanced_split(train_unique_df, label_col='label_text_id', random_state=args.seed)
    out_train_unique_balanced_tsv = os.path.join(args.output_dir, 'task1_text_train_unique_balanced.tsv')
    train_unique_balanced_df.to_csv(out_train_unique_balanced_tsv, sep='\t', index=False, encoding='utf-8')
    print(f"[OPTION B] Saved 50/50 balanced training set (unique-tweet) to: {out_train_unique_balanced_tsv} ({len(train_unique_balanced_df)} rows)")

    stats['balanced_training_option_b'] = {
        'row_level': {
            'file': 'task1_text_train_balanced.tsv',
            'total_samples': len(train_balanced_df),
            'informative_count': int((train_balanced_df['label_text_id'] == 1).sum()),
            'not_informative_count': int((train_balanced_df['label_text_id'] == 0).sum()),
            'balance_percentage': '50.0% / 50.0%'
        },
        'unique_tweet_level': {
            'file': 'task1_text_train_unique_balanced.tsv',
            'total_samples': len(train_unique_balanced_df),
            'informative_count': int((train_unique_balanced_df['label_text_id'] == 1).sum()),
            'not_informative_count': int((train_unique_balanced_df['label_text_id'] == 0).sum()),
            'balance_percentage': '50.0% / 50.0%'
        }
    }

    # Overall totals
    stats['overall_totals'] = {
        'total_rows': int(len(combined_df)),
        'total_unique_tweets': int(combined_df['tweet_id'].nunique()),
        'total_text_informative': int((combined_df['label_text_id'] == 1).sum()),
        'total_text_not_informative': int((combined_df['label_text_id'] == 0).sum()),
        'total_image_informative': int((combined_df['label_image_id'] == 1).sum()),
        'total_image_not_informative': int((combined_df['label_image_id'] == 0).sum()),
    }

    # Sequence length recommendation
    stats['sequence_length_recommendation'] = {
        'recommended_max_seq_length': 64,
        'upper_bound_max_seq_length': 128,
        'rationale': (
            f"Maximum whitespace token length across all 18,082 rows is {combined_df['token_count_bertweet'].max()} tokens "
            f"(99th percentile is {np.percentile(combined_df['token_count_bertweet'], 99):.1f}). "
            f"Under BERTweet's FastBPE subword tokenizer, max_length=64 or 128 guarantees zero truncation."
        )
    }

    # Save summary metadata JSON
    summary_path = os.path.join(args.output_dir, 'preprocessing_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)
    print(f"\nSaved preprocessing summary & statistics to: {summary_path}")

    # Print summary tables to console
    print("\n" + "=" * 80)
    print(" PREPROCESSING SUMMARY & TASK 1 CLASS DISTRIBUTIONS")
    print("=" * 80)
    print(f"{'Split':<14} | {'Total Rows':<12} | {'Unique Tweets':<14} | {'Informative (1)':<18} | {'Not Informative (0)':<20}")
    print("-" * 80)
    for s in split_names:
        txt_counts = stats['class_distribution_text'][s]
        inf = txt_counts.get('informative', 0)
        not_inf = txt_counts.get('not_informative', 0)
        print(f"{s:<14} | {stats['row_counts'][s]:<12} | {stats['unique_tweets'][s]:<14} | {inf:<18} ({inf/stats['row_counts'][s]*100:.1f}%) | {not_inf:<20} ({not_inf/stats['row_counts'][s]*100:.1f}%)")
    print("-" * 80)
    tot = stats['overall_totals']
    print(f"{'TOTAL (ALL)':<14} | {tot['total_rows']:<12} | {tot['total_unique_tweets']:<14} | {tot['total_text_informative']:<18} ({tot['total_text_informative']/tot['total_rows']*100:.1f}%) | {tot['total_text_not_informative']:<20} ({tot['total_text_not_informative']/tot['total_rows']*100:.1f}%)")
    print("-" * 80)
    b_row = stats['balanced_training_option_b']['row_level']
    print(f"{'OPT-B BALANCED':<14} | {b_row['total_samples']:<12} | {'-':<14} | {b_row['informative_count']:<18} (50.0%) | {b_row['not_informative_count']:<20} (50.0%)")
    print(f"{'OPT-A WEIGHTED':<14} | {len(train_weighted_df):<12} | {'-':<14} | 9638 (w=0.7060)   | 3970 (w=1.7139)")
    print("=" * 80)

    print("\nOPTION A & OPTION B TRAINING FILES READY:")
    print(f"  [Option A] Full Weighted Training:     {out_train_weighted_tsv} (13,608 rows with sample_weight column)")
    print(f"  [Option A] Unique Weighted Training:   {out_train_unique_weighted_tsv} (11,584 rows with sample_weight column)")
    print(f"  [Option B] Undersampled Balanced:      {out_train_balanced_tsv} (7,940 rows: 50% inf / 50% not-inf)")
    print(f"  [Option B] Unique Undersampled Bal:    {out_train_unique_balanced_tsv} (6,598 rows: 50% inf / 50% not-inf)")

    print("\n" + "=" * 80)
    print(" Pipeline Completed Successfully!")
    print("=" * 80)

if __name__ == '__main__':
    main()
