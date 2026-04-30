# Image Super-Resolution using ESPCN

> **Course:** CSE445 – Machine Learning &nbsp;|&nbsp; **Section:** 06 &nbsp;|&nbsp; **Group:** 04 &nbsp;<br>
 **Instructor:** Professor M. Shifat-E-Rabbi (MSRb) <br>
 **Institution:** North South University, Dhaka, Bangladesh

[![Python](https://img.shields.io/badge/Python-3.12.3-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)](https://pytorch.org/)
[![Upscale](https://img.shields.io/badge/Upscale-×4-brightgreen)]()
[![Best Val PSNR](https://img.shields.io/badge/Best%20Val%20PSNR-37.74%20dB-yellow)]()

This project implements a complete supervised machine learning pipeline for **single-image super-resolution (SISR)**. Low-resolution (LR) images are generated from high-resolution (HR) originals using bicubic downsampling and Gaussian blur. An **ESPCN** (Efficient Sub-Pixel Convolutional Neural Network) model is trained to restore them to their original quality at a **×4 upscale factor**, and results are compared against bicubic interpolation using PSNR and visual analysis.

---

## Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Training](#training)
- [Results](#results)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [Limitations & Future Work](#limitations--future-work)
- [Group Members](#group-members)
- [References](#references)

---

## Overview

Traditional upscaling methods like bicubic interpolation are fast but cannot recover lost high-frequency details — they apply a fixed mathematical rule with no understanding of image content. ESPCN addresses this by learning filters directly from paired HR-LR image data.

The key design principle of ESPCN is that all convolution operations happen in **LR space**, making the model computationally efficient. Upscaling occurs only at the very end through a **sub-pixel convolution (PixelShuffle)** layer that rearranges feature channels into a larger spatial output:

```
LR Input (64×64)
    → Conv(5×5) + ReLU
    → Conv(3×3) + ReLU
    → Conv(3×3) + ReLU
    → Conv(3×3)  [48 channels for RGB ×4]
    → PixelShuffle(×4)
    → SR Output (256×256)
```

---

## Architecture

The ESPCN model has **74,128 trainable parameters** across 4 convolutional layers. Weights are initialized using **Kaiming Normal** initialization.

| Layer | In Channels | Out Channels | Kernel | Activation |
|-------|-------------|--------------|--------|------------|
| Conv1 | 3 | 64 | 5×5 | ReLU |
| Conv2 | 64 | 64 | 3×3 | ReLU |
| Conv3 | 64 | 32 | 3×3 | ReLU |
| Conv4 | 32 | 3 × (4²) = **48** | 3×3 | — |
| PixelShuffle | 48 ch → | 3 ch, ×4 spatial | — | — |

For RGB ×4 upscaling, the final layer must produce exactly 48 channels. PixelShuffle then rearranges them into the final 3-channel HR image. Output pixels are clamped to [0, 1].

---

## Dataset

The dataset was built from a collection of internet images containing diverse objects, textures, colors, and structures. Through standardization, augmentation, and patch-based extraction, the final training pipeline used **149,490 paired HR-LR samples**.

| Split | Samples | Percentage |
|-------|---------|------------|
| Train | 119,592 | 80% |
| Validation | 14,949 | 10% |
| Test | 14,949 | 10% |
| **Total** | **149,490** | **100%** |

The split uses a fixed random seed (`seed=42`) for full reproducibility.

### HR-LR Pair Generation

Each HR image (256×256) is degraded into an LR version (64×64) by:
1. **Bicubic downsampling** by ×4 (LANCZOS for HR resize, BICUBIC for LR)
2. **Gaussian blur** (radius = 0.5) to simulate realistic sensor degradation

### Training Augmentations

- Random horizontal flip (p = 0.5)
- Random vertical flip (p = 0.5)
- Random 90° rotations (k ∈ {0, 1, 2, 3})
- Random patch cropping — 32×32 LR patch → 128×128 HR patch

Patch extraction was critical because the original image collection was small compared to professional SR datasets. It allows the CNN to learn local structures such as grass blades, fabric lines, petal edges, and building corners.

---

## Training

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Batch size | 16 |
| Loss function | L1 Loss (MAE) |
| Optimizer | Adam |
| Initial learning rate | 1e-3 |
| LR scheduler | CosineAnnealingLR (T_max=100, η_min=1e-5) |
| Mixed precision | `torch.amp.autocast` + `GradScaler` |
| GPU | NVIDIA GeForce RTX 5060 Ti (17.1 GB VRAM) |
| ~Time per epoch | ~92 seconds |
| Total training time | ~2.6 hours |

L1 Loss was chosen over MSE because it typically preserves sharper image details. CosineAnnealingLR gradually reduces the learning rate, enabling large updates early and fine refinements near the end. Mixed precision (AMP) improves VRAM efficiency without sacrificing training stability.

The best model checkpoint is saved automatically whenever validation PSNR improves → `best_espcn_x4.pth`.

### Training Algorithm

```
1.  Load paired LR and HR images
2.  Apply augmentation and patch extraction
3.  Split data 80% / 10% / 10% (seed=42)
4.  Initialize ESPCN with Kaiming Normal weights
5.  For each epoch:
    a. Pass LR batches through the model
    b. Compute L1 loss against HR targets
    c. Backpropagate with AMP + Adam update
    d. Apply CosineAnnealingLR step
    e. Compute validation PSNR
    f. Save checkpoint if val PSNR improves
6.  Generate final comparison images on test samples
```

---

## Results

### Training History

The plot below shows L1 training loss (left) and validation PSNR (right) across all 100 epochs. Loss drops sharply in the first 20 epochs then continues to decrease steadily. Validation PSNR climbs from ~33 dB to a peak of **37.74 dB** at epoch 99.

![Training History](https://github.com/Rokib-Hasan-Oli/CSE445_Sec6_Machine_Learning_Project/blob/main/support/output_Image/training_history.png)

| Epoch | Val PSNR | |
|-------|----------|-|
| 1 | 33.07 dB | |
| 10 | 36.04 dB | |
| 25 | 36.95 dB | |
| 50 | 37.21 dB | |
| 75 | 37.57 dB | |
| 77 | 37.62 dB | ✅ |
| 96 | 37.70 dB | ✅ |
| **99** | **37.74 dB** | ✅ **Best** |
| 100 | 37.71 dB | |

---

### Visual Comparisons — Training Data Samples

Each panel shows: **LR Input (upscaled for view) | Bicubic | ESPCN | HR Ground Truth**

#### Tower (Image ID: 117675) — Bicubic: 24.20 dB → ESPCN: 29.95 dB (+5.74 dB)
ESPCN recovers the circular band structures of the tower and sharpens the cherry blossom branches that are completely blurred in both the LR input and bicubic output.

![Tower Compare](https://github.com/Rokib-Hasan-Oli/CSE445_Sec6_Machine_Learning_Project/blob/main/support/output_Image/117675_COMPARE.png)

#### Fruit Texture (Image ID: 134722) — Bicubic: 21.74 dB → ESPCN: 26.66 dB (+4.92 dB)
ESPCN reconstructs individual fruit boundaries and leaf edges from the heavily blurred LR input, closely matching the HR ground truth.

![Fruit Compare](https://github.com/Rokib-Hasan-Oli/CSE445_Sec6_Machine_Learning_Project/blob/main/support/output_Image/134722_COMPARE.png)

#### Grass Texture (Image ID: 137207) — Bicubic: 26.96 dB → ESPCN: 32.67 dB (+5.71 dB)
The strongest PSNR gains appear on this grass sample. ESPCN recovers individual blade structures and the fine dark gaps between them, demonstrating that CNN filters excel at learning repeated local textures.

![Grass Compare](https://github.com/Rokib-Hasan-Oli/CSE445_Sec6_Machine_Learning_Project/blob/main/support/output_Image/137207_COMPARE.png)

---

### Visual Comparisons — Test Data (Custom Images)

The model was also tested on custom images that were **not part of training or validation**.

#### Daisy Flower — Bicubic: 30.55 dB → ESPCN: 25.91 dB (−4.64 dB)
ESPCN produces visually sharper petal edges and more defined petal separation than bicubic. However, PSNR is lower because the model generates high-frequency edge detail that doesn't align exactly at pixel level with the HR reference. The over-sharpened petal outlines are a known artifact of PSNR-trained CNNs.

![Flower Compare](https://github.com/Rokib-Hasan-Oli/CSE445_Sec6_Machine_Learning_Project/blob/main/support/output_Image/flower-729510_1280_COMPARE.png)

#### NSU Building — Bicubic: 30.23 dB → ESPCN: 22.26 dB (−7.97 dB)
Text regions and precise architectural lines are the hardest case for ESPCN. While structural textures of the brick facade are somewhat improved, exact character recovery in the signage is limited because those fine pixel patterns are largely lost in the 64×64 LR input. This highlights the known limitation of super-resolution on text and high-precision geometric content.

![NSU Building Compare](https://github.com/Rokib-Hasan-Oli/CSE445_Sec6_Machine_Learning_Project/blob/main/support/output_Image/6529fb046efd17541e77b547bb95f38638f039b3babdf61d_COMPARE.png)

---

### Test PSNR Summary

| Image | Type | Bicubic PSNR | ESPCN PSNR | Gain |
|-------|------|-------------|------------|------|
| 137207 | Grass texture | 26.96 dB | 32.67 dB | **+5.71 dB** |
| 117675 | Tower + blossoms | 24.20 dB | 29.95 dB | **+5.74 dB** |
| 134722 | Fruit texture | 21.74 dB | 26.66 dB | **+4.92 dB** |
| flower-729510_1280 | Daisy flower | 30.55 dB | 25.91 dB | −4.64 dB |
| NSU Building | Text + architecture | 30.23 dB | 22.26 dB | −7.97 dB |

ESPCN outperforms bicubic on **texture-rich images** with repeated local patterns (grass, fabric, petals, tower rings). For images with precise text or thin architectural lines, PSNR drops below bicubic — but this does not mean total failure; the model still sharpens structural textures, and the PSNR gap reflects pixel-level misalignment rather than visual degradation. Both PSNR and side-by-side visual inspection are needed for a fair evaluation.

---

## Project Structure

```
CSE445_Sec6_Machine_Learning_Project/
├── data/                                        # Subfolder containing datasets and images
│   ├── HR_256/                                  # High-resolution images (256×256)
│   ├── LR_x4/                                   # Low-resolution images (64×64)
│   └── Selected Image for project/              # Custom test images and results
├── others/                                      # Subfolder for reports, presentations, and video
│   ├── CSE445_Final_Report_Group04.pdf          # Final report PDF
│   ├── presentation.pdf                         # Final presentation PPTX/PDF
│   ├── update_report.pdf                        # Update report PDF
│   ├── update_presentation.pptx                 # Update presentation PPTX
│   └── project_demo.mp4                         # One-minute video file showing demo run
├── support/                                     # Subfolder containing other code/support files
│   ├── output_Image/                            # All output figures used in README
│   │   ├── training_history.png                 # L1 loss & validation PSNR curves
│   │   ├── 117675_COMPARE.png                   # Tower comparison (train sample)
│   │   ├── 134722_COMPARE.png                   # Fruit texture comparison (train sample)
│   │   ├── 137207_COMPARE.png                   # Grass texture comparison (train sample)
│   │   ├── flower-729510_1280_COMPARE.png       # Daisy flower comparison (test)
│   │   └── 6529fb046efd17541e77b547_COMPARE.png # NSU Building comparison (test)
│   ├── Other test Model/                        # Other test Model(SRCNN etc)
│   ├── Standardize_images.ipynb                 # Image preprocessing & extreme augmentation pipeline
│   └── best_espcn_x4.pth                        # Best model checkpoint (saved by val PSNR)
├── main.ipynb                                   # Main training & inference notebook
├── README.md                                    # Project explanation
└── requirements.txt                             # List of required tools and libraries
```

---

## Requirements

```
Python     3.12.3
torch
torchvision
Pillow
numpy
matplotlib
jupyter
```

Install dependencies:

```bash
pip install torch torchvision pillow numpy matplotlib jupyter
```

---

## Usage

### Step 1 — Prepare the Dataset

Run `Standardize_images.ipynb` to standardize your HR images to 256×256 and generate the paired LR dataset in `LR_x4/`.

### Step 2 — Train the Model

Open `main.ipynb` and run all cells sequentially:

```
Cell 1  —  Imports & device setup (detects CUDA automatically)
Cell 2  —  Auto-detect dataset size & build 80/10/10 split
Cell 3  —  Dataset classes (SRDataset + SRImageDataset)
Cell 4  —  DataLoaders (batch=16 train, batch=1 val/test)
Cell 5  —  ESPCN model definition (74,128 params)
Cell 6  —  Helper functions (PSNR, bicubic baseline)
Cell 7  —  Training loop (100 epochs, L1, Adam, AMP)
```

The best checkpoint is saved automatically to `best_espcn_x4.pth`.

### Step 3 — Test on Custom Images

Place any `.png` / `.jpg` images into `~/sr_project/Test/`, then run the final inference cell in `main.ipynb`. For each image the pipeline will:

1. Resize to a multiple of 4 (LANCZOS)
2. Downsample ×4 with bicubic + Gaussian blur (radius 0.5) → LR
3. Run ESPCN on the LR image
4. Compute PSNR for both ESPCN and bicubic vs HR reference
5. Save a 4-panel comparison: **LR Input | Bicubic | ESPCN | HR Reference**

```
Example output:
  Original size : 640×480
  LR size       : 160×120   → saved as image_LR.png
  ESPCN output  : 640×480   → saved as image_ESPCN.png
  Bicubic PSNR  : 28.41 dB
  ESPCN   PSNR  : 33.12 dB  (gain: +4.71 dB)
  Comparison    : saved as image_COMPARE.png
```

---

## Limitations & Future Work

**Current limitations:**
- Original image diversity is limited compared to large professional SR datasets (e.g., DIV2K, VOC2012)
- PSNR does not always reflect perceived visual quality — sharper ESPCN outputs can score lower than bicubic if reconstructed edges don't align exactly at pixel level
- The model may introduce over-sharpened artifacts in regions where high-frequency detail is not recoverable from the LR input
- Designed specifically for ×4 upscaling; other scale factors require retraining

**Future improvements:**
- Train on a larger and more diverse dataset
- Add SSIM as an additional perceptual evaluation metric
- Compare with SRCNN, FSRCNN, EDSR, and ESRGAN
- Experiment with perceptual / adversarial loss for more natural textures
- Build a simple web UI for uploading LR images and receiving SR output automatically

---

## Group Members

| Student ID | Name | Email |
|------------|------|-------|
| 2211950642 | MD. Rokib Hasan Oli | rokib.oli@northsouth.edu |
| 1831906642 | Kazi Eraj Al Minahi Turjo | kazi.turjo@northsouth.edu |
| 1620018042 | Md. Sifur Rahman | sifur.rahman@northsouth.edu |


**Department of ECE, North South University, Dhaka, Bangladesh**

---

