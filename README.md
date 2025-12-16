

---

# **H-CoAtNet**

## **Hierarchical Hybrid Learning for Ichthyosis Variant Classification**

---

### **Review-Only Research Code**

> ⚠️ **Notice to Reviewers**
> This repository is released **exclusively for peer review and reproducibility** of the associated manuscript.
> Any reuse, redistribution, modification, or deployment is **not permitted** without explicit written authorization from the authors.

---

## **Abstract**

Ichthyosis represents a heterogeneous group of rare genetic dermatological disorders characterized by abnormal keratinization and severe scaling. Automated classification of ichthyosis variants remains challenging due to limited annotated datasets, extreme class imbalance, and subtle inter-class morphological variations.
This repository provides the **official reference implementation** of **H-CoAtNet**, a *Hierarchically Enhanced Hybrid Learning Framework* that integrates convolutional feature extraction, transformer-based global context modeling, and hierarchical squeeze-excitation mechanisms for multi-class ichthyosis classification from dermatological images.
The proposed architecture incorporates progressive token pruning and adaptive feature recalibration to enhance focus on discriminative regions while maintaining computational efficiency. Extensive experimental evaluation across five diagnostic categories demonstrates that H-CoAtNet outperforms strong convolutional and transformer baselines, achieving **90.51% test accuracy** and a **macro-averaged F1-score of 0.8605**, highlighting its effectiveness in class-imbalanced rare disease settings.

---

## **Associated Manuscript**

**Hierarchical Hybrid Learning: Enhanced Classification of Ichthyosis Variants in Dermatological Images Using H-CoAtNet**
Rajeev Rajan, Athul Joe Joseph Palliparambil, Anandhu P. Shaji
*Under Review, 2025* 

---

## **Repository Structure and Execution Context (Critical)**

After cloning, the **actual project root** is the inner `H-CoAtNet/` directory.

```bash
git clone https://github.com/Cyrax321/H-CoAtNet-Ichthyosis.git
cd H-CoAtNet-Ichthyosis
cd H-CoAtNet
```

All commands **must be executed from this directory**.
Running from the outer directory will result in missing file or module errors.

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

## **1. Problem Definition**

Given a dermatological image ( x ), the objective is to predict its corresponding ichthyosis subtype
( y \in {\text{HI, IV, LI, NS, Healthy}} ).
The task is complicated by severe class imbalance, limited sample size, and overlapping morphological characteristics between subtypes, particularly Lamellar Ichthyosis and Netherton Syndrome.

---

## **2. Proposed Method: H-CoAtNet**

H-CoAtNet is a hybrid architecture that sequentially integrates:

1. **ConvNeXt-based convolutional stages** for hierarchical local texture modeling
2. **Transformer blocks** for global contextual dependency learning
3. **Hierarchical squeeze-excitation mechanisms** with progressive token selection for adaptive feature recalibration

The architecture is designed to balance inductive bias and representational flexibility while maintaining computational efficiency, as detailed in Section 2 of the manuscript .

---

## **3. Dataset**

The dataset consists of **1,580 dermatological images** collected from multiple publicly available sources and curated under strict quality and ethical guidelines.
Five diagnostic categories are included:

* Harlequin Ichthyosis (HI): 158 images
* Ichthyosis Vulgaris (IV): 474 images
* Lamellar Ichthyosis (LI): 316 images
* Netherton Syndrome (NS): 182 images
* Healthy Skin: 450 images

Images are resized to **224×224**, normalized using ImageNet statistics, and split using **stratified 70/15/15 train–validation–test partitions**.

---

## **4. Training and Execution**

### **Proposed Model**

```bash
python proposed_method/train_h_coatnet.py
```

### **Baseline Models**

```bash
python baselines/train_cnn.py
python baselines/train_efficientnet.py
python baselines/train_vit.py
python baselines/train_swin.py
python baselines/train_coatnet.py
python baselines/train_gft.py
```

All models use **identical preprocessing, dataset splits, and evaluation protocols**.

---

## **5. Experimental Protocol**

* Optimizer: Adam
* Epochs: 30
* Dropout: 0.2
* Weight decay: enabled
* Training: from scratch (no external pretraining)
* Random seeds: fixed

### **Hardware**

* Apple MacBook Pro (M3 Pro, 18 GB RAM)
* Google Colab (verification only)

No TPU-specific optimizations are employed.

---

## **6. Evaluation Metrics**

* Accuracy
* Macro-averaged Precision, Recall, F1-score
* Weighted F1-score

Macro-averaged metrics are emphasized due to pronounced class imbalance in rare disease data.

---

## **7. Results**

| Model                    | Accuracy   | Macro F1   | Weighted F1 |
| ------------------------ | ---------- | ---------- | ----------- |
| **H-CoAtNet (Proposed)** | **0.9051** | **0.8605** | **0.9024**  |
| Swin Transformer         | 0.8291     | 0.7477     | 0.8150      |
| GFT                      | 0.8228     | 0.7701     | 0.8221      |
| CoAtNet                  | 0.7468     | 0.6517     | 0.7463      |
| Vision Transformer       | 0.7215     | 0.6310     | 0.7103      |
| CNN                      | 0.6962     | 0.6085     | 0.6889      |
| EfficientNet-B0          | 0.6646     | 0.5938     | 0.6675      |

---

## **8. Ethical Considerations**

* No patient-identifiable information is used
* All images are anonymized and publicly sourced
* Intended strictly as a **clinical decision-support system**, not a standalone diagnostic tool

---

## **9. Contact**

**Anandhu P. Shaji**
Email: [reach.anandhu.me@gmail.com](mailto:reach.anandhu.me@gmail.com)

---

