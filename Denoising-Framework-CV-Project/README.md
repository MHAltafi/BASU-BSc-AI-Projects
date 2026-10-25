# Casting Defect Detection and Adaptive Image Denoising Using PyTorch

A computer vision course project developed for *Introduction to Computer Vision* at BuAli Sina University in Fall 2023. It investigates whether adaptive image denoising can improve manufacturing defect detection in submersible pump impellers.

The pipeline combines binary defect classification, noise classification, and noise-specific image restoration using PyTorch and classical image processing methods.

## Table of Contents

- [Overview](#overview)
- [Visual Overview](#visual-overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [HOG/LBP Experiment](#hoglbp-experiment)
- [Results](#results)
- [Limitations and Future Work](#limitations-and-future-work)

## Overview

The project consists of three phases:

1. **Baseline classification:** Train defect classifiers on noisy casting images and establish baseline performance.
2. **Noise classification and denoising:** Train a noise classifier, identify corruption types, and denoise the images in `DataSet2`.
3. **Post-restoration classification:** Train a Vision Transformer (ViT) on restored images and compare its performance with the Phase 1 ViT baseline.

ResNet and ViT models are trained and evaluated in Phases 1 and 2. Phase 3 uses ViT only. ViT is the selected model for the final pipeline.

Supported noise types and restoration methods:

| Noise type | Restoration method |
|---|---|
| Gaussian | BM3D, with noise parameter `σ_psd = 30/255` |
| Salt-and-pepper | Two sequential `3 × 3` median-filter passes |
| Periodic | Custom U-Net convolutional autoencoder |

A supplementary experiment compares handcrafted HOG and LBP features using Random Forest and SVM classifiers.

## Visual Overview

Examples of the three investigated corruption types:

**Periodic noise**

![Periodic Noise](figures/Periodic%20Noise.png)

**Gaussian noise**

![Gaussian Noise](figures/Gaussian%20Noise.png)

**Salt-and-pepper noise**

![Salt-and-Pepper Noise](figures/Salt%20%26%20Pepper%20Noise.png)

## Project Structure

```text
Denoising-Framework-CV-Project/
├── .gitignore
├── README.md
├── Phase 1.ipynb
├── Phase 2.ipynb
├── Autoencoder (Periodic Noise).ipynb
├── DenoisingFramework.ipynb
├── DenoisingFramework.py
├── Phase 3.ipynb
└── HOG & LBP.ipynb
```

| File | Purpose |
|---|---|
| `README.md` | Project documentation and usage instructions. |
| `Phase 1.ipynb` | Train and evaluate ResNet and ViT defect classifiers using `DataSet1`. |
| `Phase 2.ipynb` | Train and evaluate ResNet and ViT noise classifiers using `DataSet2` and `Labels.xlsx`; denoise `DataSet2` images. |
| `Autoencoder (Periodic Noise).ipynb` | Train and evaluate the periodic-noise autoencoder. |
| `DenoisingFramework.ipynb` | Run adaptive denoising on test images and log restoration metrics. |
| `DenoisingFramework.py` | Interactive batch restoration on test images with CSV metric export. |
| `Phase 3.ipynb` | Train and evaluate a ViT defect classifier on restored images in `DataSet3`. |
| `HOG & LBP.ipynb` | Supplementary handcrafted-feature experiment. |

The Phase 2 notebook performs denoising on `DataSet2`. The separate `DenoisingFramework.ipynb` notebook and `DenoisingFramework.py` script are intended for applying the trained restoration pipeline to later test sets.

## Dataset

Datasets and trained checkpoints are not included in the repository. The data was manually curated for this project by TAs. Contact the repository owner to request access, subject to applicable sharing restrictions.

Expected organization:

```text
DataSet1/
├── Defective/
└── Ok/

DataSet2/
├── With-Noise/
└── Without-Noise/

Labels.xlsx

DataSet3/
├── Defective/
└── Ok/
```

- **`DataSet1/`:** Noisy casting images for baseline defect classification.
- **`DataSet2/`:** Noisy images, corresponding clean references, and labels for Gaussian, salt-and-pepper, and periodic noise. This dataset is used for noise classification and denoising in Phase 2.
- **`DataSet3/`:** Restored casting images organized by defect class for Phase 3.
- **`Labels.xlsx`:** Maps noisy image filenames to their noise classes.
- **`Periodic Autoencoder.pt`:** Trained periodic-noise autoencoder weights.
- **`models/`:** Classifier checkpoints.
- **`DenoisingResults.csv`:** Exported restoration metrics.

A subset of 450 paired periodic-noise and clean images is used to train the autoencoder. Preserve noisy/clean correspondence and avoid placing related source images across different dataset splits.

### Preprocessing

- Input sizes: `256 × 256` for CNNs and the autoencoder; `224 × 224` for ViT.
- Pixel scaling: `[0, 1]`.
- Normalization: mean `0.55`, standard deviation `0.22`, where used by the corresponding notebook.
- Augmentation: horizontal and vertical flips, and rotations within `±60°`.
- Dataset split: 70% training, 15% validation, and 15% testing for experiments using this split.

Preprocessing and splitting can vary between notebooks. Follow each notebook's implementation and keep normalization consistent with the selected model, especially for pretrained backbones.

## Model Architecture

### Defect and Noise Classifiers

ResNet and Vision Transformer models from `torchvision.models` are trained in Phases 1 and 2 using modified classification heads.

- **Defect classification:** `Defective` and `Ok` (two logits).
- **Noise classification:** `Gaussian`, `Salt`, and `Periodic` (three logits).
- **Loss:** `nn.CrossEntropyLoss`.

Phase 1 evaluates ResNet and ViT for defect classification. Phase 2 evaluates ResNet and ViT for noise classification. ViT is selected for the final pipeline. Phase 3 trains a ViT defect classifier on restored images and compares it with the Phase 1 ViT baseline.

### Periodic-Noise U-Net

The custom `PeriodicAutoencoder` reconstructs clean grayscale images from periodic-noise inputs.

- Input/output: `1 × 256 × 256`.
- Four encoder stages with channels `64 → 128 → 256 → 512`.
- Two `3 × 3` convolutions per block, followed by batch normalization and ReLU.
- `2 × 2` max pooling and a 1,024-channel bottleneck at `16 × 16`.
- Four decoder stages using bilinear interpolation and skip connections.
- Final `1 × 1` convolution and sigmoid activation.

**Training configuration:** 450 paired images, `nn.MSELoss`, Adam optimizer, learning rate `0.001`, batch size `8`, and 15 epochs.

## Installation

Requires Python, PyTorch, Torchvision, and the image-processing and machine learning libraries listed below.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

Install PyTorch and Torchvision using the [official installation selector](https://pytorch.org/get-started/locally/) to match your operating system and hardware.

For example, for a compatible CUDA 11.8 environment:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Install the remaining dependencies:

```bash
pip install torchmetrics numpy pandas matplotlib Pillow \
    opencv-python scikit-learn scikit-image bm3d lpips \
    tqdm openpyxl jupyter
```

The standalone script also uses `tkinter`, which may require separate installation on Linux (for example, `python3-tk`).

Launch Jupyter:

```bash
jupyter notebook
```

A CUDA-compatible GPU is recommended for training and autoencoder inference. CPU execution may be possible but substantially slower. For reproducibility, use a tested environment and record dependency versions.

## Usage

Run the experiments in the following order:

1. **Phase 1 — Baseline classification:** Open `Phase 1.ipynb`, select `DataSet1/`, train the ResNet and ViT classifiers, and record the baseline test metrics. Use the ViT results as the baseline for the Phase 3 comparison.

2. **Phase 2 — Noise classification and denoising:** Open `Phase 2.ipynb`, select `DataSet2/With-Noise/` and `Labels.xlsx`, train the ResNet and ViT noise classifiers, and denoise the `DataSet2` images using the corresponding restoration methods. Save the required classifier checkpoint.

3. **Periodic-noise autoencoder:** Open `Autoencoder (Periodic Noise).ipynb`, select the periodic-noise images and corresponding clean references, train the autoencoder, and save its weights.

4. **Adaptive denoising on test sets:** Run `DenoisingFramework.ipynb` or execute:

   ```bash
   python DenoisingFramework.py
   ```

   The standalone script prompts for the noisy-image directory, clean-reference directory, output directory, Phase 2 noise-classifier checkpoint (ViT for the selected pipeline), autoencoder checkpoint, and CSV output path. It predicts the noise class, applies the corresponding restoration method, and records PSNR, SSIM, and LPIPS when clean references are supplied.

5. **Phase 3 — Post-restoration classification:** Open `Phase 3.ipynb`, select the restored `DataSet3/` directory, train the ViT defect classifier, and compare its results with the Phase 1 ViT baseline.

6. **Optional — HOG/LBP:** Run `HOG & LBP.ipynb` for the supplementary classical machine learning experiment.

## HOG/LBP Experiment

The supplementary notebook compares HOG and LBP descriptors using Random Forest and RBF-kernel SVM classifiers at `512 × 512`, `256 × 256`, and `128 × 128` resolutions.

- **HOG:** 9 orientations, `8 × 8` pixels per cell, `2 × 2` cells per block, and L2-Hys normalization.
- **LBP:** Radius 3, 24 neighbors, uniform patterns, and a 26-bin normalized histogram.
- **Split:** 80/20 stratified train/test.
- **SVM:** `StandardScaler` with an RBF kernel.

The notebook generates result tables, comparisons, bar charts, and a confusion matrix. This experiment is independent of the main three-phase pipeline.

## Results

The following values are recorded experimental results, not guarantees of performance in unseen industrial environments.

### Periodic Autoencoder

| Stage | MSE loss |
|---|---:|
| Epoch 1 — Training | 0.025852 |
| Epoch 1 — Validation | 0.015524 |
| Epoch 15 — Training | 0.001781 |
| Epoch 15 — Validation | 0.002639 |
| Final test evaluation | 0.002451 |

### Image Restoration

Higher PSNR and SSIM generally indicate better reconstruction; lower LPIPS indicates greater perceptual similarity.

| Noise type | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|
| Gaussian | BM3D | 34.24 dB | 0.9120 | 0.1659 |
| Salt-and-pepper | Two-pass median filter | 35.17 dB | 0.9511 | 0.0797 |
| Periodic | U-Net autoencoder | 26.89 dB | 0.8902 | 0.1745 |
| **Overall** | **Adaptive pipeline** | **30.80 dB** | **0.9109** | **0.1487** |

Values are from `DenoisingFramework.ipynb`. Overall metrics depend on the notebook's aggregation method and may differ from unweighted averages of class-level means.

### Defect Classification: ViT Baseline vs. Restored Images

Phase 1 and Phase 3 results compare ViT defect classification on noisy and restored images, respectively.

| Metric | Phase 1: Noisy | Phase 3: Restored |
|---|---:|---:|
| Accuracy | 93% | 95% |
| Weighted precision | 93% | 95% |
| Weighted recall | 93% | 95% |
| Specificity (class 1) | 90% | 94% |
| Weighted F1-score | 93% | 95% |
| ROC-AUC | 0.93 | 0.95 |

These results suggest an improvement in the recorded experiment. For a fair comparison, use equivalent test sets, consistent evaluation procedures, and verified image correspondence wherever possible.

### HOG/LBP Results

**Best individual configurations**

| Feature | Resolution | Classifier | Accuracy | F1-score |
|---|---|---|---:|---:|
| LBP | Original (`512 × 512`) | Random Forest | 0.784 | 0.8188 |
| HOG | `128 × 128` | SVM | 0.764 | 0.8053 |
| LBP | `256 × 256` | Random Forest | 0.756 | 0.7932 |
| HOG | `256 × 256` | SVM | 0.716 | 0.7760 |
| LBP | `128 × 128` | Random Forest | 0.712 | 0.7662 |

**Average results**

| Comparison | Category | Mean accuracy | Mean F1-score |
|---|---|---:|---:|
| Feature | LBP | 0.6947 | 0.7699 |
| Feature | HOG | 0.6860 | 0.7609 |
| Resolution | `128 × 128` | 0.704 | 0.7696 |
| Resolution | `256 × 256` | 0.694 | 0.7661 |
| Resolution | Original | 0.673 | 0.7605 |
| Classifier | Random Forest | 0.7107 | 0.7704 |
| Classifier | SVM | 0.6700 | 0.7605 |

LBP slightly outperformed HOG on average, and Random Forest achieved higher average scores than SVM in the recorded experiment. Reduced resolutions also achieved higher average scores than the original resolution.

## Limitations and Future Work

Potential improvements include:

- Detecting compound and non-stationary noise.
- Jointly optimizing restoration and defect classification.
- Adding `argparse` support for command-line and headless execution.
- Extending binary classification to defect localization using object detection or segmentation.
- Combining HOG/LBP features with CNN or ViT embeddings.
- Evaluating whether denoising preserves small defects and improves classification sensitivity.
- Recording random seeds, dependency versions, dataset splits, and checkpoint configurations.

## Acknowledgments

Developed for *Introduction to Computer Vision* at BuAli Sina University in Fall 2023. The project provided practical experience in PyTorch-based classification, image restoration, U-Net autoencoders, classical filtering, handcrafted feature extraction, and perceptual image quality evaluation.