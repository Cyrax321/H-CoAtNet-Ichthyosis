

---

# H-CoAtNet

## Hierarchically Enhanced Hybrid Learning for Ichthyosis Classification

**Official Research Codebase**

This repository provides the **reference implementation** of **H-CoAtNet**, a hierarchically enhanced hybrid convolution–transformer architecture for **multi-class Ichthyosis subtype classification** from dermatological images.

The repository is released to support **reproducibility, benchmarking, and further research** in rare disease medical image analysis.

---

## 📄 Associated Paper

**Hierarchical Hybrid Learning: Enhanced Classification of Ichthyosis Variants in Dermatological Images Using H-CoAtNet**
Athul Joe Joseph Palliparambil, Anandhu P. Shaji, Rajeev Rajan
*Under Review, 2025*

---

## 1. Introduction

Ichthyosis is a group of rare genetic skin disorders characterized by abnormal keratinization and severe scaling. Automated diagnosis is challenging due to:

* Extreme **class imbalance**
* **Subtle morphological differences** between subtypes
* **Limited annotated medical datasets**

**H-CoAtNet** addresses these challenges using a hybrid architecture that combines convolutional inductive biases, transformer-based global context modeling, and hierarchical squeeze-excitation with progressive token pruning.

---

## 2. Method Overview

H-CoAtNet integrates three core components:

1. **ConvNeXt Backbone**
   Captures fine-grained local textures and hierarchical spatial representations.

2. **Transformer Blocks**
   Model long-range global dependencies across dermatological regions.

3. **Hierarchical Squeeze-Excitation with Token Pruning**
   Adaptively recalibrates features and focuses computation on the most discriminative regions.

This design is optimized for **data-scarce and class-imbalanced medical image classification**.

---

## 3. Repository Structure

```text
H-CoAtNet/
├── README.md                     # Project documentation
├── requirements.txt              # Dependency specification
│
├── proposed_model/
│   └── train_h_coatnet.py        # Proposed H-CoAtNet training pipeline
│
└── baselines/
    ├── train_cnn.py              # CNN baseline
    ├── train_efficientnet.py     # EfficientNet-B0
    ├── train_vit.py              # Vision Transformer
    ├── train_swin.py             # Swin Transformer
    ├── train_coatnet.py          # CoAtNet baseline
    └── train_gft.py              # Gradient Focal Transformer
```


## 4. Dataset

* **Total images:** 1,580

* **Classes (5):**

  * Harlequin Ichthyosis (HI)
  * Ichthyosis Vulgaris (IV)
  * Lamellar Ichthyosis (LI)
  * Netherton Syndrome (NS)
  * Healthy Skin

* **Split:**

  * Train: 70%
  * Validation: 15%
  * Test: 15% (stratified)

* **Image size:** 224 × 224

---

## 5. Dataset Access (Required)

The dataset is hosted on **Roboflow Universe** and must be downloaded using **your own API key**.

### Step 1: Create a Roboflow Account

1. Visit: [https://universe.roboflow.com](https://universe.roboflow.com)
2. Sign up or log in

---

### Step 2: Obtain the Dataset API Key

1. Visit the dataset page:
   [https://universe.roboflow.com/hi-l9ueo/ich-s-7lnsj](https://universe.roboflow.com/hi-l9ueo/ich-s-7lnsj)
2. Click **Download Dataset**
3. Select **Download Dataset (Get code snippet)**
4. Choose:

   * Format: **Python**
   * Framework: **Custom / PyTorch**
5. Enable **Show download code**
6. Copy the value inside:
   api_key="YOUR_API_KEY"


---

## 6. Adding the API Key to the Code

Each training script contains a configuration section at the top.

### Example: `proposed_model/train_h_coatnet.py`

API_KEY = "PASTE_YOUR_ROBOFLOW_API_KEY_HERE"

Paste your API key as a string.

---

## 7. Installation

### Step 1: Clone the Repository

git clone [https://github.com/Cyrax321/H-CoAtNet-Ichthyosis.git](https://github.com/Cyrax321/H-CoAtNet-Ichthyosis.git)
cd H-CoAtNet

---

````markdown
## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Cyrax321/H-CoAtNet-Ichthyosis.git
cd H-CoAtNet-Ichthyosis
````

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies**

* Python ≥ 3.9
* PyTorch
* timm
* torchvision
* roboflow
* scikit-learn
* numpy, pandas, matplotlib

---

## 8. Running the Experiments

All commands must be run from the **root directory** of the repository.

### 8.1 Train the Proposed Model (H-CoAtNet)

```bash
python -m proposed_model.train_h_coatnet
```

This script will:

* Download the dataset using your Roboflow API key
* Apply preprocessing and data augmentation
* Train H-CoAtNet for 30 epochs
* Save the best model weights
* Generate confusion matrix and learning curves

---

### 8.2 Train Baseline Models

All baseline models can be trained independently using the commands below.
Each script follows the same dataset split, preprocessing pipeline, and evaluation protocol to ensure a fair comparison with the proposed H-CoAtNet model.

Run all commands from the **root directory** of the repository.

#### CNN Baseline

```bash
python -m baselines.train_cnn
```

#### EfficientNet-B0

```bash
python -m baselines.train_efficientnet
```

#### Vision Transformer (ViT)

```bash
python -m baselines.train_vit
```

#### Swin Transformer

```bash
python -m baselines.train_swin
```

#### CoAtNet Baseline

```bash
python -m baselines.train_coatnet
```

#### Gradient Focal Transformer (GFT)

```bash
python -m baselines.train_gft
```

---

## 9. Experimental Protocol (Reproducibility)

* Optimizer: Adam
* Epochs: 30
* Dropout: 0.2
* Weight decay enabled
* No external pretraining (trained from scratch)
* Fixed random seeds

### Hardware

* Apple MacBook Pro (M3 Pro, 18 GB RAM)
* Google Colab TPU v4 (verification)

---

## 10. Evaluation Metrics

* Accuracy
* Macro-averaged Precision, Recall, F1-score
* Weighted F1-score

Macro metrics are emphasized to properly evaluate **class-imbalanced medical data**.

---

## 11. Results Summary

| Model                | Accuracy   | Macro F1   | Weighted F1 |
| -------------------- | ---------- | ---------- | ----------- |
| **H-CoAtNet (Ours)** | **90.51%** | **0.8605** | **0.9024**  |
| Swin Transformer     | 82.91%     | 0.7477     | 0.8150      |
| GFT                  | 82.28%     | 0.7701     | 0.8221      |
| CoAtNet              | 74.68%     | 0.6517     | 0.7463      |
| Vision Transformer   | 72.15%     | 0.6310     | 0.7103      |
| CNN                  | 69.62%     | 0.6085     | 0.6889      |
| EfficientNet-B0      | 66.46%     | 0.5938     | 0.6675      |

---

## 12. Ethical Considerations

* No patient-identifiable data is used
* Images are anonymized and publicly sourced
* Intended as a **decision-support system**, not a standalone diagnostic tool

---

## 13. Contact

**Anandhu P Shaji**
Email: [reach.anandhu.me@gmail.com](mailto:reach.anandhu.me@gmail.com)

---
