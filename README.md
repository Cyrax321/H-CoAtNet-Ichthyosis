

---

# **H-CoAtNet**

## **Hierarchically Enhanced Hybrid Learning for Ichthyosis Classification**

---

⚠️ **Review-Only Notice**

This repository is provided **solely for peer review and reproducibility purposes** associated with the submitted manuscript.
**Reuse, redistribution, modification, or deployment of this code is not permitted** without explicit written permission from the authors.

---

## **Official Research Codebase**

This repository provides the **reference implementation** of **H-CoAtNet**, a hierarchically enhanced hybrid convolution–transformer architecture for **multi-class Ichthyosis subtype classification** from dermatological images.

The codebase is released to support **scientific reproducibility, benchmarking, and methodological verification** in rare disease medical image analysis.

---

## 📄 **Associated Paper**

**Hierarchical Hybrid Learning: Enhanced Classification of Ichthyosis Variants in Dermatological Images Using H-CoAtNet**
Athul Joe Joseph Palliparambil, Anandhu P. Shaji, Rajeev Rajan
*(Under Review, 2025)*

---

## 🔧 **Repository Structure and Execution Context (Important)**

After cloning, note that the **actual project root** is the inner `H-CoAtNet/` directory.

```bash
git clone https://github.com/Cyrax321/H-CoAtNet-Ichthyosis.git
cd H-CoAtNet-Ichthyosis
cd H-CoAtNet
```

All commands **must be executed from this directory**.
Running commands from the outer directory will result in missing file or module errors.

```
H-CoAtNet/
├── README.md
├── requirements.txt
├── proposed_method/
│   └── train_h_coatnet.py
└── baselines/
    ├── train_cnn.py
    ├── train_efficientnet.py
    ├── train_vit.py
    ├── train_swin.py
    ├── train_coatnet.py
    └── train_gft.py
```

---

## 1. **Environment Setup**

### Dependencies

```bash
pip install -r requirements.txt
```

**Requirements**

* Python ≥ 3.9
* PyTorch
* timm
* torchvision
* scikit-learn
* numpy, pandas, matplotlib
* roboflow

Tested on macOS (Apple Silicon) and Linux environments.

---

## 2. **Problem Overview**

Ichthyosis is a group of rare genetic skin disorders characterized by abnormal keratinization and severe scaling. Automated diagnosis is challenging due to:

* Severe **class imbalance**
* **Subtle morphological differences** between subtypes
* **Limited annotated medical datasets**

The proposed **H-CoAtNet** framework addresses these challenges through hybrid convolution–transformer modeling with hierarchical feature refinement.

---

## 3. **Method Overview**

H-CoAtNet integrates three core components:

* Convolutional stem for local texture and scale modeling
* Transformer blocks for global contextual dependency learning
* Hierarchical squeeze-excitation with progressive token selection

This design balances **inductive bias**, **representational flexibility**, and **computational efficiency**, making it suitable for rare disease classification.

---

## 4. **Dataset Description**

The dataset consists of **1,580 dermatological images** across five diagnostic categories:

* Harlequin Ichthyosis (HI): 158 images
* Ichthyosis Vulgaris (IV): 474 images
* Lamellar Ichthyosis (LI): 316 images
* Netherton Syndrome (NS): 182 images
* Healthy Skin: 450 images

Images are resized to **224×224**, normalized using ImageNet statistics, and split using **stratified 70/15/15 train–validation–test partitions**.

---

## 🔐 5. Dataset Access via API (Required Before Training)

For **controlled, reproducible, and ethical dataset access**, this project uses the **Roboflow Dataset API**.

### Step 1: Create a Roboflow account

👉 [https://roboflow.com](https://roboflow.com)

---

### Step 2: Generate an API key

After logging in, generate your API key here:

👉 **Roboflow API Key Dashboard**
[https://roboflow.com/account/api](https://roboflow.com/account/api)

Copy the API key. You will need it before running any training script.

---

### Step 3: Add the API key to the code (mandatory)

Open **each training script** (example shown for H-CoAtNet):

```
proposed_method/train_h_coatnet.py
```

Locate the Roboflow initialization section (usually near the top):

```python
from roboflow import Roboflow
```

Add or modify the code as follows:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="API KEY HERE")

```

Replace:

* `API KEY HERE` → your Roboflow API key

> ⚠️ **Important**
> The same dataset version must be used across **all baseline and proposed models** to reproduce reported results.

---

### Official API Documentation

* Roboflow Docs: [https://docs.roboflow.com](https://docs.roboflow.com)
* API Key Management: [https://roboflow.com/account/api](https://roboflow.com/account/api)

---

## 6. **Training and Execution**

All scripts are executed **directly** to ensure maximum compatibility and reproducibility.

### Proposed Method (H-CoAtNet)

```bash
python proposed_method/train_h_coatnet.py
```

### Baseline Models

```bash
python baselines/train_cnn.py
python baselines/train_efficientnet.py
python baselines/train_vit.py
python baselines/train_swin.py
python baselines/train_coatnet.py
python baselines/train_gft.py
```

All models use **identical dataset splits, preprocessing, and evaluation protocols**.

---

## 7. **Experimental Protocol (Reproducibility)**

* Optimizer: Adam
* Epochs: 30
* Dropout: 0.2
* Weight decay enabled
* No external pretraining (trained from scratch)
* Fixed random seeds

### Hardware

* Apple MacBook Pro (M3 Pro, 18 GB RAM)
* Google Colab (verification only)

No TPU-specific optimizations are used.

---

## 8. **Evaluation Metrics**

* Accuracy
* Macro-averaged Precision, Recall, F1-score
* Weighted F1-score

Macro metrics are emphasized due to inherent **class imbalance** in rare disease datasets.

---

## 9. **Results Summary**

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

## 10. **Ethical Considerations**

* No patient-identifiable data is used
* Images are anonymized and publicly sourced
* Intended strictly as a **clinical decision-support system**, not a standalone diagnostic tool

---

## 11. **Contact**

**Anandhu P. Shaji**
Email: [reach.anandhu.me@gmail.com](mailto:reach.anandhu.me@gmail.com)

---

