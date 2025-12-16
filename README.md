

````markdown
# H-CoAtNet: Hierarchically Enhanced Hybrid Learning for Ichthyosis Classification

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Roboflow-purple)](https://universe.roboflow.com/hi-l9ueo/ich-s-7lnsj)

This repository contains the official PyTorch implementation of **H-CoAtNet**, a novel hybrid architecture proposed for the fine-grained classification of Ichthyosis variants. This work establishes a new state-of-the-art benchmark for discriminating between *Harlequin Ichthyosis*, *Lamellar Ichthyosis*, *Ichthyosis Vulgaris*, *Netherton Syndrome*, and *Healthy Skin*.

## 📄 Abstract

Ichthyosis comprises a heterogeneous group of genetic skin disorders characterized by dry, scaling skin. Automated diagnosis is challenging due to the extreme rarity of specific subtypes and subtle inter-class morphological similarities.

We propose **H-CoAtNet**, which synergistically integrates:
* **Convolutional Inductive Biases** (ConvNeXt) for local texture extraction.
* **Transformer Self-Attention** for global context modeling.
* **Hierarchical Squeeze-Excitation** for adaptive feature recalibration.

**Performance:** The proposed model achieves **90.51% accuracy**, significantly outperforming standard CNNs, pure Transformers (ViT, Swin), and hybrid baselines like Gradient Focal Transformer (GFT) and standard CoAtNet.

## 📂 Repository Structure

The codebase is organized as follows:

```text
H-CoAtNet-Ichthyosis/
│
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
│
├── proposed_model/
│   └── train_h_coatnet.py    # Implementation of H-CoAtNet (Proposed Method)
│
└── baselines/                # Comparative Baselines
    ├── train_coatnet.py      # Standard CoAtNet Baseline
    ├── train_gft.py          # Gradient Focal Transformer (GFT)
    ├── train_swin.py         # Swin Transformer
    ├── train_vit.py          # Vision Transformer (ViT)
    ├── train_efficientnet.py # EfficientNet-B0 (Trained from scratch)
    └── train_cnn.py          # Standard CNN Baseline
````

## 🛠️ Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/username/H-CoAtNet-Ichthyosis.git](https://github.com/username/H-CoAtNet-Ichthyosis.git)
    cd H-CoAtNet-Ichthyosis
    ```

2.  **Install Dependencies:**
    The code requires PyTorch, `timm`, and `roboflow`. Install them using:

    ```bash
    pip install -r requirements.txt
    ```

## 📊 Dataset & Configuration

The dataset used in this study is publicly available on **Roboflow**. To reproduce the experiments, you must download the data using your own Roboflow API key.

### **How to Access the Data**

To obtain your API key, follow these steps:

1.  Click the **[Project Link](https://universe.roboflow.com/hi-l9ueo/ich-s-7lnsj)** to visit the dataset page.

2.  Navigate to **Dataset** in the sidebar.

3.  Click **Download Dataset**.

4.  Select **Download Dataset (Get a code snippet or ZIP file)**.

5.  In the format selection, ensure **Show download code** is selected.

6.  Choose the option: **Custom train this dataset using the provided code snippet in a notebook**.

7.  Copy **only** the API key string from the provided snippet.

      * *Example format:* `api_key="xxxxxxxxxxxxxxxx"`

8.  Open the training script you wish to run (e.g., `proposed_model/train_h_coatnet.py`) and paste your key into the configuration section:

    ```python
    # === Configuration ===
    API_KEY = "PASTE_YOUR_KEY_HERE"
    ```

*Note: The dataset contains 1,580 images balanced across 5 classes.*

### **Dataset Citation**

If you use this dataset, please cite it as follows:

```bibtex
@misc{ ich-s-7lnsj_dataset,
    title = { ich-s Dataset },
    type = { Open Source Dataset },
    author = { HI },
    howpublished = { \url{ [https://universe.roboflow.com/hi-l9ueo/ich-s-7lnsj](https://universe.roboflow.com/hi-l9ueo/ich-s-7lnsj) } },
    url = { [https://universe.roboflow.com/hi-l9ueo/ich-s-7lnsj](https://universe.roboflow.com/hi-l9ueo/ich-s-7lnsj) },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2025 },
    month = { oct },
    note = { visited on 2025-12-16 },
}
```

## Usage

To ensure reproducibility, all scripts contain the complete model architecture and training loop. Run the scripts as modules from the root directory.

### **1. Train Proposed Model (H-CoAtNet)**

To train the H-CoAtNet model which achieves the state-of-the-art results:

```bash
python -m proposed_model.train_h_coatnet
```

### **2. Train Baselines**

We provide 6 comparative baselines as detailed in the paper:

  * **CoAtNet Baseline (ConvNeXt-Tiny):**
    ```bash
    python -m baselines.train_coatnet
    ```
  * **Gradient Focal Transformer (GFT):**
    ```bash
    python -m baselines.train_gft
    ```
  * **Swin Transformer:**
    ```bash
    python -m baselines.train_swin
    ```
  * **Vision Transformer (ViT):**
    ```bash
    python -m baselines.train_vit
    ```
  * **EfficientNet-B0:**
    ```bash
    python -m baselines.train_efficientnet
    ```
  * **Standard CNN:**
    ```bash
    python -m baselines.train_cnn
    ```

## 📈 Results

Summary of performance metrics reported in the paper:

| Model | Accuracy | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: |
| **H-CoAtNet (Ours)** | **90.51%** | **0.8605** | **0.9024** |
| Swin Transformer | 82.91% | 0.7477 | 0.8150 |
| GFT | 82.28% | 0.7701 | 0.8221 |
| CoAtNet Baseline | 74.68% | 0.6517 | 0.7463 |
| Vision Transformer | 72.15% | 0.6310 | 0.7103 |
| CNN | 69.62% | 0.6085 | 0.6889 |
| EfficientNet-B0 | 66.46% | 0.5938 | 0.6675 |

Each training script will automatically generate:

  * `best_model.pth`: The saved model weights.
  * `confusion_matrix.png`: A visualization of class-wise performance.
  * `loss_curves.png`: Training and validation loss plots.

## 📜 Paper Citation

If you use this code or methodology in your research, please cite our work:

```bibtex
@article{HCoAtNet2025,
  title={Hierarchical Hybrid Learning: Enhanced Classification of Ichthyosis Variants in Dermatological Images Using H-CoAtNet},
  author={Rajan, Rajeev and Palliparambil, Athul Joe Joseph and Shaji, Anandhu P},
  journal={Submission under Review},
  year={2025}
}
```

## 📧 Contact

For questions regarding the code or dataset, please contact:

  * **Anandhu P Shaji:** reach.anandhu.me@gmail.com

<!-- end list -->

```
```
