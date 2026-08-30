# Raman Microscopy Cancer Staging

MSc Advanced Computer Science Dissertation
University of Leeds, 2025--2026
Umar Amjad Khan
Supervisors: Dr Sharib Ali, Dr Julia Gala de Pablo

---

## Project Overview

This repository contains the code for the dissertation:
**Analysis of Raman Microscopy Images for Cancer Staging and Therapy Response**

The project develops and evaluates a deep learning pipeline for automated
classification of colorectal cancer stages (Dukes staging) from hyperspectral
Stimulated Raman Scattering (SRS) microscopy images of human colon cell lines.

---

## Key Results

| Model | Dataset | Split | Accuracy | Macro F1 |
|---|---|---|---|---|
| ResNet50 pretrained 12ch protein_max | Primary 2026 | Session-based | 77.32% | 0.793 |
| ResNet50 pretrained 12ch protein_max | Primary 2026 | Random upper bound | 98.56% | 0.984 |
| ResNet50 pretrained 75px border | Secondary 2025 | Acquisition-based | 92.86% | 0.943 |
| ResNet50 + GroupNorm | Primary 2026 | Session-based | 66.01% | 0.666 |
| ResNet50 + SCL | Primary 2026 | Session-based | 73.17% | 0.714 |
| ResNet50 + DANN | Primary 2026 | Session-based | 31.88% | 0.307 |
| ResNet50 + LoRA | Primary 2026 | Session-based | 40.02% | 0.444 |

---

## Repository Structure

- train_primary_best.py: Best model, 12ch protein_max pretrained ResNet50 (77.32%)
- train_primary_random_split.py: Random split upper bound evaluation (98.56%)
- train_primary_groupnorm.py: GroupNorm architecture experiment (66.01%)
- train_primary_dann.py: Domain Adversarial Neural Network experiment (31.88%)
- train_primary_lora.py: Low-Rank Adaptation experiment (40.02%)
- train_primary_scl.py: Supervised Contrastive Learning experiment (73.17%)
- train_secondary_best.py: Secondary 2025 dataset, 75px border (92.86%)
- notebooks/01_data_exploration.ipynb: Dataset exploration and visualisation
- notebooks/02_cell_extraction.ipynb: Cell extraction pipeline
- notebooks/03_normalisation_investigation.ipynb: Normalisation strategy comparison
- experiments/: Exploratory scripts used during development

---

## Datasets

- Primary dataset 2026: 4,946 cells from 6 colon cell lines (FHC, HCT116, CaCo2, SW480, HT29, SW620) across 3 imaging sessions. 53-channel hyperspectral SRS arrays.
- Secondary dataset 2025: 1,010 cells from the same 6 cell lines in a single session. Standard 4-channel TIF format.

Data is not included in this repository. Contact Dr Julia Gala de Pablo (University of Leeds) for data access.

---

## Requirements

pip install torch torchvision numpy matplotlib scikit-learn tifffile roifile cellpose

Tested with Python 3.9, PyTorch 2.5.1, CUDA 12.x on NVIDIA A100.

---

## Reproducing Results

All scripts use a fixed random seed of 42 and report mean +/- std across 3 runs.

Primary dataset best model (77.32%):
python train_primary_best.py

Secondary dataset (92.86%):
python train_secondary_best.py

Update the SCRATCH and DATA_ROOT variables at the top of each script for your environment.

---

## Citation

Khan, U.A. (2026). Analysis of Raman Microscopy Images for Cancer Staging and Therapy Response. MSc Dissertation, University of Leeds.

---

## Key Finding

A central finding of this project is the large gap between random split (98.56%) and
session-based split (77.32%) performance on the primary dataset -- a difference of
21.24 percentage points. This demonstrates that random cell-level splitting produces
optimistic performance estimates that do not reflect real-world cross-session
generalisation. Session-based splitting, where all cells from one imaging session are
held out for testing, is the appropriate evaluation protocol for multi-session SRS
microscopy classification tasks.

The protein_max normalisation strategy (dividing by the maximum intensity of the
protein channel across training sessions) substantially outperforms the physically
motivated ratio normalisation for cross-session generalisation (77.32% vs 51.29%),
suggesting that preserving absolute intensity structure is more important than
channel-wise normalisation for this task.
