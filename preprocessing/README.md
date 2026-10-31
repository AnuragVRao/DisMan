# CrisisMMD v2.0 Multimodal Disaster Response Dataset
## Comprehensive Preprocessing & Model Training Guide (Tasks 1, 2, and 3)

This repository contains the complete end-to-end preprocessing pipelines, verified datasets, and PyTorch / Scikit-Learn data loaders for all 3 tasks in the **CrisisMMD v2.0** benchmark.

---

## Table of Contents
1. [Repository Structure](#1-repository-structure)
2. [Preprocessing Architecture & Standards](#2-preprocessing-architecture--standards)
3. [Task 1: Informative vs. Not Informative (Binary)](#3-task-1-informative-vs-not-informative-binary)
   - [Overview & Class Distributions](#task-1-distributions)
   - [Case 1: Text-Only Model](#task-1-text-only)
   - [Case 2: Image-Only Model](#task-1-image-only)
   - [Case 3: Multimodal Model (Both)](#task-1-multimodal)
   - [Option A (Weighted) vs. Option B (Balanced)](#task-1-options)
4. [Task 2: Humanitarian Categories (8-Class)](#4-task-2-humanitarian-categories-8-class)
   - [8-Class Mapping & Distribution](#task-2-distributions)
   - [Concept 1 (Class-Weighted) vs. Concept 2 (Oversampled)](#task-2-concepts)
   - [Case 1: Text-Only Model](#task-2-text-only)
   - [Case 2: Image-Only Model](#task-2-image-only)
   - [Case 3: Multimodal Model (Both)](#task-2-multimodal)
5. [Task 3: Damage Severity Assessment (3-Class)](#5-task-3-damage-severity-assessment-3-class)
   - [Overview & Ordinal Severity Mapping](#task-3-distributions)
   - [Comparison of 3 Training Paradigms (Weighted, Oversampled, Undersampled)](#task-3-paradigms)
   - [Case 1: Image-Only Model (Primary)](#task-3-image-only)
   - [Case 2: Text-Only Model](#task-3-text-only)
   - [Case 3: Multimodal Model (Both)](#task-3-multimodal)
6. [Master Quick-Reference Matrix](#6-master-quick-reference-matrix)
7. [How to Re-Run Preprocessing Pipelines](#7-how-to-re-run-preprocessing-pipelines)

---

## 1. Repository Structure

```text
Capstone Dataset/
│
├── CrisisMMD_v2.0/                       # Raw dataset root (images, annotations, JSON)
│   ├── data_image/                      # 18,082 disaster images (JPG) across 7 events
│   ├── annotations/                     # Raw event annotation TSV files
│   └── crisismmd_datasplit_all/         # Official benchmark train/dev/test TSV files
│
├── preprocessed_text_task1/             # Task 1 Text-Only Datasets
│   ├── task1_text_train.tsv             # Official train split (13,608 rows)
│   ├── task1_text_train_weighted.tsv    # Option A: With sample/class weights (13,608 rows)
│   ├── task1_text_train_balanced.tsv    # Option B: 50/50 balanced via undersampling (7,940 rows)
│   ├── task1_text_train_unique_balanced.tsv # Deduplicated 50/50 unique tweets (6,598 rows)
│   ├── task1_text_dev.tsv               # Validation split (2,237 rows)
│   ├── task1_text_test.tsv              # Final test evaluation split (2,237 rows)
│   ├── task1_text_all.tsv               # Complete combined split (18,082 rows)
│   └── preprocessing_summary.json       # Metadata & distribution statistics
│
├── preprocessed_images_task1/           # Task 1 Image & Multimodal Datasets
│   ├── task1_multimodal_train.tsv       # Aligned train split (13,608 rows)
│   ├── task1_multimodal_train_weighted.tsv # Option 1: Dual-head loss weights attached
│   ├── task1_multimodal_train_balanced.tsv # Option 2: 50/50 text-balanced multimodal split
│   ├── task1_multimodal_dev.tsv         # Validation split (2,237 rows)
│   ├── task1_multimodal_test.tsv        # Final test evaluation split (2,237 rows)
│   ├── task1_multimodal_all.tsv         # Complete combined split (18,082 rows)
│   └── image_preprocessing_summary.json # Verification, image modes & duplicate audit
│
├── preprocessed_task2/                  # Task 2 Humanitarian Categories Datasets (8-Class)
│   ├── task2_multimodal_train_weighted.tsv # Concept 1: Full data with 8-class weights (13,608 rows)
│   ├── task2_multimodal_train_oversampled.tsv # Concept 2: Multimodal oversampled (38,536 rows)
│   ├── task2_text_train_oversampled.tsv # Concept 2: Text oversampled (38,536 rows, 4,817/class)
│   ├── task2_image_train_oversampled.tsv# Concept 2: Image oversampled (52,520 rows, 6,565/class)
│   ├── task2_multimodal_dev.tsv         # Validation split (2,237 rows)
│   ├── task2_multimodal_test.tsv        # Test evaluation split (2,237 rows)
│   ├── task2_text_train.tsv             # Text-only train split (13,608 rows)
│   ├── task2_text_dev.tsv               # Text-only validation split (2,237 rows)
│   ├── task2_text_test.tsv              # Text-only test split (2,237 rows)
│   └── task2_preprocessing_summary.json # 8-class weights, counts & agreement rates
│
├── preprocessed_task3/                  # Task 3 Damage Severity Assessment Datasets (3-Class)
│   ├── task3_multimodal_train_weighted.tsv # Concept 1: Full data with 3-class weights (2,468 rows)
│   ├── task3_multimodal_train_oversampled.tsv # Concept 2: Oversampled (4,644 rows, 1,548/class)
│   ├── task3_multimodal_train_undersampled.tsv # Concept 3: Undersampled (999 rows, 333/class)
│   ├── task3_multimodal_dev.tsv         # Validation split (529 rows)
│   ├── task3_multimodal_test.tsv        # Test evaluation split (529 rows)
│   ├── task3_multimodal_all.tsv         # Complete 3-class split (3,526 rows)
│   ├── task3_text_train.tsv / dev / test# Text-only splits
│   └── task3_preprocessing_summary.json # 3-class weights & distribution summary
│
├── preprocess_text_task1.py             # Script for Task 1 Text Preprocessing
├── preprocess_images_task1.py           # Script for Task 1 Image & Multimodal Alignment
├── preprocess_task2_humanitarian.py     # Script for Task 2 8-Class Preprocessing
├── preprocess_task3_damage.py           # Script for Task 3 Damage Severity Preprocessing
└── README.md                            # Master project documentation
```

---

## 2. Preprocessing Architecture & Standards

### A. Text Preprocessing Pipeline
Every text row is preprocessed to produce two dedicated representations:
1. **`clean_text_bertweet`** (Optimized for `vinai/bertweet-base` and Transformer models):
   - **Encoding & Mojibake Fix:** Decodes corrupted UTF-8/Windows-1252 byte sequences (e.g. `Weâ€™re` $\to$ `We're`, `â\x9d¤ï¸\x8f` $\to$ `❤️`), unescapes HTML entities (`&amp;` $\to$ `&`), strips non-printable C1 control codes, applies Unicode `NFKC` normalization.
   - **Retweet Marker Removal:** Leading `RT @username:` and `RT ` markers stripped.
   - **User Mentions:** Replaced with the official BERTweet `@USER` placeholder.
   - **Web Links:** Replaced with the official BERTweet `HTTPURL` placeholder.
   - **Hashtags:** `#` dropped; PascalCase / CamelCase hashtags decompounded into natural words (e.g. `#CaliforniaWildfires` $\to$ `California Wildfires`).
   - **Emojis:** Clean, normalized Unicode emojis preserved (BERTweet vocabulary includes native emojis).
   - **Whitespace:** Collapses repeated spaces, tabs, and newlines; trims ends.
   - **Token Lengths:** Max whitespace token count across all 18,082 tweets is 28 tokens (Mean: 13.9, 99th percentile: 23). **Recommended max sequence length: `64` to `128`**.

2. **`clean_text_tfidf`** (Optimized for Scikit-Learn `TfidfVectorizer` baselines):
   - Applied lowercasing.
   - URLs and user mentions stripped completely (avoids injecting synthetic tokens into vocabulary).
   - Non-alphanumeric punctuation and symbols removed.
   - Ready for standard English stopword removal (`stop_words='english'`).

### B. Image Preprocessing Pipeline
- **Integrity Verification:** 100% of images verified on disk (**0 missing, 0 corrupt**).
- **Color Mode Conversion:** Non-RGB modes (188 RGBA, 1 Grayscale L, 3 Palette P) converted to 3-channel RGB (`img.convert('RGB')`) to prevent CNN/ViT channel crashes.
- **Resizing:** Standardized to **224×224** using bicubic anti-aliasing interpolation.
- **Normalization:** ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` (CLIP values supported as well).
- **Data Augmentation (Train only):** `RandomResizedCrop(224, scale=(0.8, 1.0))`, `RandomHorizontalFlip(p=0.5)`, `RandomRotation(degrees=15)`, and mild `ColorJitter`.
- **Deterministic Transforms (Dev & Test):** `Resize((224, 224))`, `ToTensor()`, and `Normalize()`.

---

## 3. Task 1: Informative vs. Not Informative (Binary)

<a name="task-1-distributions"></a>
### Task 1 Class Distributions (18,082 Total Rows)
| Split | Total Rows | Unique Tweets | Text: Informative (`1`) | Text: Not Info (`0`) | Image: Informative (`1`) | Image: Not Info (`0`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 13,608 | 11,584 | **9,638** (70.8%) | **3,970** (29.2%) | **7,059** (51.9%) | **6,549** (48.1%) |
| **Dev** | 2,237 | 2,237 | **1,612** (72.1%) | **625** (27.9%) | **1,164** (52.0%) | **1,073** (48.0%) |
| **Test** | 2,237 | 2,237 | **1,612** (72.1%) | **625** (27.9%) | **1,151** (51.5%) | **1,086** (48.5%) |
| **Total** | **18,082** | **16,058** | **12,862** (71.1%) | **5,220** (28.9%) | **9,374** (51.8%) | **8,708** (48.2%) |

> **Key takeaway:** In Task 1, **Image is naturally balanced (~52% vs ~48%)**, while **Text is imbalanced (~71% vs ~29%)**.

---

<a name="task-1-text-only"></a>
### Case 1: Task 1 Text-Only Model (BERTweet / TF-IDF)
- **Folder:** `preprocessed_text_task1/`
- **Training File:**
  - *Option A (Weighted, full 13,608 rows):* `preprocessed_text_task1/task1_text_train_weighted.tsv`
  - *Option B (50/50 Balanced, 7,940 rows):* `preprocessed_text_task1/task1_text_train_balanced.tsv`
- **Dev / Test Files:** `preprocessed_text_task1/task1_text_dev.tsv`, `preprocessed_text_task1/task1_text_test.tsv`
- **Input Column:** `clean_text_bertweet` (or `clean_text_tfidf`)
- **Target Column:** `label_text_id` (`1` = informative, `0` = not_informative)

#### PyTorch Code (BERTweet Text-Only):
```python
import pandas as pd
import torch
import torch.nn as nn

train_df = pd.read_csv("preprocessed_text_task1/task1_text_train_weighted.tsv", sep="\t")
test_df  = pd.read_csv("preprocessed_text_task1/task1_text_test.tsv", sep="\t")

# Text Loss Weights: [Class 0 weight, Class 1 weight]
text_weights = torch.tensor([1.7139, 0.7060], dtype=torch.float)
criterion = nn.CrossEntropyLoss(weight=text_weights)
```

---

<a name="task-1-image-only"></a>
### Case 2: Task 1 Image-Only Model (ResNet50 / ViT)
- **Folder:** `preprocessed_images_task1/`
- **Training File:** `preprocessed_images_task1/task1_multimodal_train.tsv` (13,608 rows)
- **Dev / Test Files:** `preprocessed_images_task1/task1_multimodal_dev.tsv`, `preprocessed_images_task1/task1_multimodal_test.tsv`
- **Input Column:** `image_full_path` (Ignore all text columns)
- **Target Column:** `label_image_id` (`1` = informative, `0` = not_informative)
- **Loss:** Standard unweighted `nn.CrossEntropyLoss()` *(Images are already 51.9% vs 48.1% balanced)*.

#### PyTorch Code (Image-Only):
```python
from preprocess_images_task1 import CrisisMMDImageDataset, get_image_transforms
from torch.utils.data import DataLoader
import torch.nn as nn

train_tf = get_image_transforms(is_train=True)
eval_tf  = get_image_transforms(is_train=False)

train_dataset = CrisisMMDImageDataset(
    tsv_file="preprocessed_images_task1/task1_multimodal_train.tsv",
    transform=train_tf,
    label_col="label_image_id"
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
criterion = nn.CrossEntropyLoss()
```

---

<a name="task-1-multimodal"></a>
### Case 3: Task 1 Multimodal Model (Both Text + Image)
- **Folder:** `preprocessed_images_task1/`
- **Training File:**
  - *Option 1 (Weighted, 13,608 rows):* `preprocessed_images_task1/task1_multimodal_train_weighted.tsv`
  - *Option 2 (50/50 Text-Balanced, 7,940 rows):* `preprocessed_images_task1/task1_multimodal_train_balanced.tsv`
- **Dev / Test Files:** `preprocessed_images_task1/task1_multimodal_dev.tsv`, `preprocessed_images_task1/task1_multimodal_test.tsv`
- **Input Columns:** `clean_text_bertweet` AND `image_full_path`
- **Target Columns:** Dual-Head Multi-Task (`label_text_id` AND `label_image_id`)

#### PyTorch Dual-Head Loss Code:
```python
from preprocess_images_task1 import CrisisMMDMultimodalDataset, get_image_transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

train_tf = get_image_transforms(is_train=True)
dataset = CrisisMMDMultimodalDataset("preprocessed_images_task1/task1_multimodal_train_weighted.tsv", transform=train_tf)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

criterion_text  = nn.CrossEntropyLoss(weight=torch.tensor([1.7139, 0.7060]))
criterion_image = nn.CrossEntropyLoss()

for batch in loader:
    pred_text, pred_image = model(batch["text_bertweet"], batch["image"])
    loss = criterion_text(pred_text, batch["label_text_id"]) + criterion_image(pred_image, batch["label_image_id"])
    loss.backward()
```

---

## 4. Task 2: Humanitarian Categories (8-Class)

<a name="task-2-distributions"></a>
### Task 2 8-Class Mapping & Training Breakdown (13,608 Rows)
| ID | Category Name | Train Text Count | Text Weight | Train Image Count | Image Weight |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **0** | `affected_individuals` | 384 (2.8%) | **4.4297** | 397 (2.9%) | **4.2846** |
| **1** | `infrastructure_and_utility_damage` | 1,102 (8.1%) | **1.5436** | 2,761 (20.3%) | **0.6161** |
| **2** | `injured_or_dead_people` | 393 (2.9%) | **4.3282** | 92 (0.7%) | **18.4891** |
| **3** | `missing_or_found_people` | **29** (0.2%) | **58.6552** | **9** (0.07%) | **189.0000** |
| **4** | `not_humanitarian` | 3,970 (29.2%) | **0.4285** | 6,565 (48.2%) | **0.2591** |
| **5** | `other_relevant_information` | 4,817 (35.4%) | **0.3531** | 1,860 (13.7%) | **0.9145** |
| **6** | `rescue_volunteering_or_donation_effort` | 2,867 (21.1%) | **0.5933** | 1,694 (12.4%) | **1.0041** |
| **7** | `vehicle_damage` | 46 (0.34%) | **36.9783** | 230 (1.7%) | **7.3957** |
| **Total** | | **13,608** | | **13,608** | |

---

<a name="task-2-concepts"></a>
### Concept 1 (Class-Weighted) vs. Concept 2 (Oversampled)
- **Concept 1 (Class-Weighted):** Train on `task2_multimodal_train_weighted.tsv` (13,608 rows). Apply the precomputed 8-class weights to `nn.CrossEntropyLoss(weight=weights)`.
- **Concept 2 (Oversampled):** Train on `task2_text_train_oversampled.tsv` or `task2_multimodal_train_oversampled.tsv` (38,536 rows). Every class has **exactly 4,817 samples (12.5% each)**. Use standard unweighted `nn.CrossEntropyLoss()`.
- **Baseline finding:** On the test set, Oversampling achieved **0.5075 Macro F1** vs **0.4907 Macro F1** for Weighted, giving a **+1.68% boost** on minority categories.

---

<a name="task-2-text-only"></a>
### Case 1: Task 2 Text-Only Model
- **Folder:** `preprocessed_task2/`
- **Training File:**
  - *Weighted:* `preprocessed_task2/task2_text_train.tsv` (13,608 rows)
  - *Oversampled:* `preprocessed_task2/task2_text_train_oversampled.tsv` (38,536 rows)
- **Dev / Test Files:** `preprocessed_task2/task2_text_dev.tsv`, `preprocessed_task2/task2_text_test.tsv`
- **Target Column:** `label_text_id` (Integer `0` to `7`)

---

<a name="task-2-image-only"></a>
### Case 2: Task 2 Image-Only Model
- **Folder:** `preprocessed_task2/`
- **Training File:**
  - *Weighted:* `preprocessed_task2/task2_multimodal_train_weighted.tsv` (13,608 rows)
  - *Oversampled:* `preprocessed_task2/task2_image_train_oversampled.tsv` (52,520 rows, 6,565/class)
- **Dev / Test Files:** `preprocessed_task2/task2_multimodal_dev.tsv`, `preprocessed_task2/task2_multimodal_test.tsv`
- **Target Column:** `label_image_id` (Integer `0` to `7`)

---

<a name="task-2-multimodal"></a>
### Case 3: Task 2 Multimodal Model (Both)
- **Folder:** `preprocessed_task2/`
- **Training File:** `preprocessed_task2/task2_multimodal_train_weighted.tsv` (or `task2_multimodal_train_oversampled.tsv`)
- **Dev / Test Files:** `preprocessed_task2/task2_multimodal_dev.tsv`, `preprocessed_task2/task2_multimodal_test.tsv`
- **Inputs:** `clean_text_bertweet` + `image_full_path`
- **Targets:** Dual-Head (`label_text_id` [8 classes] + `label_image_id` [8 classes])

#### PyTorch Multimodal Task 2 Code:
```python
from preprocess_task2_humanitarian import CrisisMMDTask2Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

train_tf = transforms.Compose([
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = CrisisMMDTask2Dataset("preprocessed_task2/task2_multimodal_train_weighted.tsv", transform=train_tf)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 8-Class Weights
text_weights  = torch.tensor([4.4297, 1.5436, 4.3282, 58.6552, 0.4285, 0.3531, 0.5933, 36.9783], dtype=torch.float)
image_weights = torch.tensor([4.2846, 0.6161, 18.4891, 189.0, 0.2591, 0.9145, 1.0041, 7.3957], dtype=torch.float)

criterion_text  = nn.CrossEntropyLoss(weight=text_weights)
criterion_image = nn.CrossEntropyLoss(weight=image_weights)

for batch in loader:
    pred_text_8, pred_img_8 = model(batch["text_bertweet"], batch["image"])
    loss = criterion_text(pred_text_8, batch["label_text_id"]) + criterion_image(pred_img_8, batch["label_image_id"])
    loss.backward()
```

---

## 5. Task 3: Damage Severity Assessment (3-Class)

<a name="task-3-distributions"></a>
### Task 3 Overview & Ordinal Severity Mapping
Evaluates damage severity on infrastructure images & tweets:
- **`0: little_or_no_damage`** (Minority: 13.5%)
- **`1: mild_damage`** (Moderate: 23.8%)
- **`2: severe_damage`** (Majority: 62.7%)

| Split | Total Rows | Little or No (`0`) | Mild (`1`) | Severe (`2`) |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | **2,468** | **333** (13.5%) | **587** (23.8%) | **1,548** (62.7%) |
| **Dev** | **529** | **71** (13.4%) | **126** (23.8%) | **332** (62.8%) |
| **Test** | **529** | **71** (13.4%) | **126** (23.8%) | **332** (62.8%) |
| **Total** | **3,526** | **475** (13.5%) | **839** (23.8%) | **2,212** (62.7%) |

---

<a name="task-3-paradigms"></a>
### Comparison of 3 Training Paradigms (in `preprocessed_task3/`)
1. **Concept 1 (Class-Weighted):** `task3_multimodal_train_weighted.tsv` (2,468 rows, full data).
   - Weights: `[2.4705, 1.4015, 0.5314]`
2. **Concept 2 (Random Oversampling):** `task3_multimodal_train_oversampled.tsv` (4,644 rows).
   - Exactly **1,548 samples per class (33.3% each)**.
3. **Concept 3 (Random Undersampling):** `task3_multimodal_train_undersampled.tsv` (999 rows).
   - Exactly **333 samples per class (33.3% each)**.

---

<a name="task-3-image-only"></a>
### Case 1: Task 3 Image-Only Model (Primary Model for Damage Assessment)
- **Folder:** `preprocessed_task3/`
- **Training File:** `preprocessed_task3/task3_multimodal_train_weighted.tsv` (or `task3_multimodal_train_oversampled.tsv`)
- **Dev / Test Files:** `preprocessed_task3/task3_multimodal_dev.tsv`, `preprocessed_task3/task3_multimodal_test.tsv`
- **Input Column:** `image_full_path` $\to$ transformed to `[3, 224, 224]`
- **Target Column:** `label_id` (`0`, `1`, `2`)

#### PyTorch Code (Task 3 Image-Only):
```python
from preprocess_task3_damage import CrisisMMDTask3Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

train_tf = transforms.Compose([
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = CrisisMMDTask3Dataset("preprocessed_task3/task3_multimodal_train_weighted.tsv", transform=train_tf)
loader  = DataLoader(dataset, batch_size=16, shuffle=True)

# 3-Class Weights: [little_or_no, mild, severe]
weights = torch.tensor([2.4705, 1.4015, 0.5314], dtype=torch.float)
criterion = nn.CrossEntropyLoss(weight=weights)

for batch in loader:
    outputs = model(batch["image"])
    loss = criterion(outputs, batch["label_id"])
    loss.backward()
```

---

<a name="task-3-multimodal"></a>
### Case 3: Task 3 Multimodal Model (Both)
```python
for batch in loader:
    outputs_3class = multimodal_model(batch["text_bertweet"], batch["image"])
    loss = criterion(outputs_3class, batch["label_id"])
    loss.backward()
```

---

## 6. Master Quick-Reference Matrix

| Task | Target Meaning | Modality | Primary Training File | Rows | Target Label | Loss Setup |
| :---: | :--- | :--- | :--- | :--- | :---: | :--- |
| **Task 1** | Informative vs. Not Info | **Text-Only** | `preprocessed_text_task1/task1_text_train_weighted.tsv` | 13,608 | `label_text_id` | `weights=[1.7139, 0.7060]` |
| **Task 1** | Informative vs. Not Info | **Image-Only** | `preprocessed_images_task1/task1_multimodal_train.tsv` | 13,608 | `label_image_id` | Unweighted *(already ~50/50)* |
| **Task 1** | Informative vs. Not Info | **Multimodal** | `preprocessed_images_task1/task1_multimodal_train_weighted.tsv` | 13,608 | Dual Head | `L_text`(weighted) + `L_img`(unweighted) |
| **Task 2** | 8 Humanitarian Categories | **Text-Only** | `preprocessed_task2/task2_text_train_oversampled.tsv` | 38,536 | `label_text_id` | Unweighted *(12.5% each)* |
| **Task 2** | 8 Humanitarian Categories | **Image-Only** | `preprocessed_task2/task2_multimodal_train_weighted.tsv` | 13,608 | `label_image_id` | `8-class img weights` |
| **Task 2** | 8 Humanitarian Categories | **Multimodal** | `preprocessed_task2/task2_multimodal_train_weighted.tsv` | 13,608 | Dual Head | `8-class text & img weights` |
| **Task 3** | Damage Severity (0, 1, 2) | **Image-Only** | `preprocessed_task3/task3_multimodal_train_weighted.tsv` | 2,468 | `label_id` | `weights=[2.4705, 1.4015, 0.5314]` |
| **Task 3** | Damage Severity (0, 1, 2) | **Multimodal** | `preprocessed_task3/task3_multimodal_train_oversampled.tsv` | 4,644 | `label_id` | Unweighted *(33.3% each)* |

---

## 7. How to Re-Run Preprocessing Pipelines

If you ever wish to re-generate the preprocessed datasets from scratch:

```powershell
# Task 1 Text Preprocessing:
python preprocess_text_task1.py

# Task 1 Image & Multimodal Alignment:
python preprocess_images_task1.py

# Task 2 Humanitarian Categories Preprocessing:
python preprocess_task2_humanitarian.py

# Task 3 Damage Severity Assessment Preprocessing:
python preprocess_task3_damage.py
```

