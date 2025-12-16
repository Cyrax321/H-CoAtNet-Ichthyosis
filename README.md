# H-CoAtNet

### Hierarchically Enhanced Hybrid Learning for Ichthyosis Classification

**Official Research Codebase**

This repository provides the **reference implementation** of **H-CoAtNet**, a
hierarchically enhanced hybrid convolution–transformer architecture for
multi-class Ichthyosis classification.

The repository is released to support **reproducibility, benchmarking, and
further research** in rare disease medical image analysis.


## 📄 Paper

> **Hierarchical Hybrid Learning: Enhanced Classification of Ichthyosis Variants in Dermatological Images Using H-CoAtNet**
> Rajeev Rajan, Athul Joe Joseph Palliparambil, Anandhu P. Shaji
> *Under Review, 2025* 

---

## 1. Introduction

Ichthyosis comprises a heterogeneous group of rare genetic skin disorders characterized by abnormal keratinization and severe scaling. Automated diagnosis remains challenging due to:

* Extreme **class imbalance**
* **Subtle inter-class morphological differences**
* **Limited availability of labeled medical data**

H-CoAtNet addresses these challenges through a **hybrid architectural design** that integrates convolutional inductive biases, transformer-based global context modeling, and hierarchical feature recalibration with adaptive token pruning.

---

## 2. Method Overview

**H-CoAtNet** is a sequential hybrid architecture consisting of:

* **ConvNeXt backbone** for hierarchical local feature extraction
* **Transformer blocks** for global contextual dependency modeling
* **Hierarchical Squeeze-Excitation (H-SE)** with gradient-based token importance scoring
* **Progressive token pruning** to improve computational efficiency

This design enables robust learning under **data-scarce and imbalanced conditions**, which are typical in rare disease classification.

Architectural details, mathematical formulations, and complexity analysis are provided in **Section 2** of the paper .

---

## 3. Repository Structure

```text
H-CoAtNet/
│
├── README.md                     # This document
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

---

## 4. Dataset

* **Total images:** 1,580

* **Classes (5):**

  * Harlequin Ichthyosis (HI)
  * Ichthyosis Vulgaris (IV)
  * Lamellar Ichthyosis (LI)
  * Netherton Syndrome (NS)
  * Healthy Skin

* **Split:** 70% train / 15% validation / 15% test (stratified)

* **Resolution:** 224 × 224

* **Normalization:** ImageNet statistics

The dataset was curated from multiple public dermatological sources and validated for diagnostic correctness. Full dataset construction details are provided in **Section 3.1** of the paper .

---

## 5. Experimental Protocol (Reproducibility)

All experiments strictly follow a **controlled and reproducible protocol**, consistent with NeurIPS and IEEE guidelines.

### Training Configuration

* Optimizer: Adam
* Epochs: 30
* Batch normalization enabled
* Dropout: 0.2
* Weight decay and learning rate scheduling applied
* No external pretraining (trained from scratch)

### Hardware

* Apple MacBook Pro (M3 Pro, 18 GB RAM)
* Google Colab TPU v4 (verification)

Random seeds were fixed to ensure reproducibility across runs.

---

## 6. Running the Code

### Train Proposed Model

```bash
python -m proposed_model.train_h_coatnet
```

### Train Baselines

```bash
python -m baselines.train_cnn
python -m baselines.train_efficientnet
python -m baselines.train_vit
python -m baselines.train_swin
python -m baselines.train_coatnet
python -m baselines.train_gft
```

Each script:

* Implements the full architecture
* Trains using the same experimental protocol
* Saves best-performing model checkpoints
* Outputs confusion matrices and learning curves

---

## 7. Evaluation Metrics

Performance is evaluated using:

* **Accuracy**
* **Macro-averaged Precision, Recall, and F1-score**
* **Weighted F1-score**

Macro metrics are emphasized to account for **class imbalance**, aligning with best practices in medical AI evaluation.

---

## 8. Results

| Model                |   Accuracy |   Macro F1 | Weighted F1 |
| -------------------- | ---------: | ---------: | ----------: |
| **H-CoAtNet (Ours)** | **90.51%** | **0.8605** |  **0.9024** |
| Swin Transformer     |     82.91% |     0.7477 |      0.8150 |
| GFT                  |     82.28% |     0.7701 |      0.8221 |
| CoAtNet              |     74.68% |     0.6517 |      0.7463 |
| Vision Transformer   |     72.15% |     0.6310 |      0.7103 |
| CNN                  |     69.62% |     0.6085 |      0.6889 |
| EfficientNet-B0      |     66.46% |     0.5938 |      0.6675 |

H-CoAtNet demonstrates superior performance across **all classes**, with particularly strong improvements for **minority and clinically critical subtypes** .

---

## 9. Ethical Considerations

* No patient-identifiable information is used
* All images are anonymized and sourced from publicly available or educational resources
* The study complies with ethical standards for secondary medical data usage
* The system is intended as a **decision-support tool**, not a standalone diagnostic system

---

## 10. Limitations

* Dataset size remains limited due to disease rarity
* External clinical validation is required prior to deployment
* Genetic and clinical metadata are not incorporated in the current model

---

## 11. Citation

If you use this repository, please cite:

```bibtex
@article{HCoAtNet2025,
  title   = {Hierarchical Hybrid Learning: Enhanced Classification of Ichthyosis Variants in Dermatological Images Using H-CoAtNet},
  author  = {Rajan, Rajeev and Palliparambil, Athul Joe Joseph and Shaji, Anandhu P},
  journal = {Under Review},
  year    = {2025}
}
```

---

## 12. Contact

**Anandhu P Shaji**
📧 [reach.anandhu.me@gmail.com](mailto:reach.anandhu.me@gmail.com)

---
