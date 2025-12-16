
---

# H-CoAtNet

## Hierarchically Enhanced Hybrid Learning for Ichthyosis Classification

---

⚠️ **Review-Only Notice**

This repository is provided **solely for peer review and reproducibility purposes** associated with the submitted manuscript.
**Reuse, redistribution, modification, or deployment of this code is not permitted** without explicit written permission from the authors.

---

## **Official Research Codebase**

This repository provides the **reference implementation** of **H-CoAtNet**, a hierarchically enhanced hybrid convolution–transformer architecture for **multi-class Ichthyosis subtype classification** from dermatological images.

The repository is released to support **reproducibility and benchmarking** in rare disease medical image analysis.

---

## 📄 Associated Paper

**Hierarchical Hybrid Learning: Enhanced Classification of Ichthyosis Variants in Dermatological Images Using H-CoAtNet**
Athul Joe Joseph Palliparambil, Anandhu P. Shaji, Rajeev Rajan
*(Under Review, 2025)*

---

## 🔧 Repository Structure and Setup (Important)

After cloning, note that the **actual project root** is the inner `H-CoAtNet/` directory.

```bash
git clone https://github.com/Cyrax321/H-CoAtNet-Ichthyosis.git
cd H-CoAtNet-Ichthyosis
cd H-CoAtNet
```

All commands below **must be executed from this directory**.
Running commands from the outer directory will result in missing file or module errors.

---

## 1. Environment Setup

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

Tested on macOS (Apple Silicon) and Linux environments.

---

## 2. Introduction

Ichthyosis is a group of rare genetic skin disorders characterized by abnormal keratinization and severe scaling. Automated diagnosis is challenging due to:

* Extreme **class imbalance**
* **Subtle morphological differences** between subtypes
* **Limited annotated medical datasets**

**H-CoAtNet** addresses these challenges using a hybrid architecture that combines convolutional inductive biases, transformer-based global context modeling, and hierarchical feature refinement.

---

## 3. Method Overview

H-CoAtNet integrates three core components:

* Convolutional stem for local texture modeling
* Transformer blocks for global contextual reasoning
* Hierarchical feature aggregation for subtype discrimination

---

## 8. Training and Execution

All scripts are executed **directly** to ensure maximum compatibility and reproducibility.

### Proposed Method (H-CoAtNet)

```bash
python proposed_method/train_h_coatnet.py
```

### Baseline Models

#### CNN Baseline

```bash
python baselines/train_cnn.py
```

#### EfficientNet-B0

```bash
python baselines/train_efficientnet.py
```

#### Vision Transformer (ViT)

```bash
python baselines/train_vit.py
```

#### Swin Transformer

```bash
python baselines/train_swin.py
```

#### CoAtNet Baseline

```bash
python baselines/train_coatnet.py
```

#### Gradient Focal Transformer (GFT)

```bash
python baselines/train_gft.py
```

> All models are trained using identical dataset splits and evaluation protocols for fair comparison.

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
* Google Colab (verification only)

No TPU-specific optimizations are used.

---

## 10. Evaluation Metrics

* Accuracy
* Macro-averaged Precision, Recall, F1-score
* Weighted F1-score

Macro-averaged metrics are emphasized to properly assess **class-imbalanced medical datasets**.

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
* Intended strictly as a **decision-support system**, not a standalone diagnostic tool

---

## 13. Contact

**Anandhu P. Shaji**
Email: [reach.anandhu.me@gmail.com](mailto:reach.anandhu.me@gmail.com)

---

