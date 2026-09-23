# Team 4 — Sentiment Analysis (NLP)

Machine Learning (Semester 2026-III) — Systems Engineering
**Universidad Distrital Francisco José de Caldas**
**Instructor:** Eng. Carlos Andrés Sierra, M.Sc.

**Team Members:**
* Xiomara Salome Arias Arias
* Julian David Celis Giraldo
* Alejandro Gomez Moyano

---

## Project Overview

This repository contains the progressive machine learning workshops developed by Team 4, focused on **Natural Language Processing (NLP)** and text classification.

The project is built around a dataset of **42,656 Disneyland park reviews**, featuring ratings (1–5 stars), visit dates, reviewer locations, and park branches. Each workshop applies a core machine learning paradigm to this text domain, building on a single, reproducible data pipeline that evolves across the semester.

---

## Repository Structure

```
ML_NPL/
├── checkpoints/          # Saved model weights (populated from Workshop 2 onward)
├── data/                 # Split files and dataset metadata (raw data downloaded separately)
├── notebooks/             # Jupyter notebooks for exploration and analysis (EDA, experiments)
├── runs/                  # Training logs (wandb / TensorBoard)
├── src/                    # Entry-point / orchestration scripts that use the ml_npl package
├── download_dataset.py     # Script to download and verify dataset integrity
├── poetry.lock
├── pyproject.toml
└── README.md               # This file
```

Reports for each workshop are kept separately, outside the codebase:

```
reports/
├── Workshop #1/   # Data Preparation & Supervised Learning Baselines
├── Workshop #2/   # Deep Learning Basics
├── Workshop #3/   # Reinforcement Learning
├── Workshop #4/   # Unsupervised Learning
├── Workshop #5/   # Generative Models (GANs)
└── Workshop #6/   # Domain Adaptation
```

---

## Workshops & Focus Areas

| Report | Focus Area | Key Implementation Goals |
| :--- | :--- | :--- |
| `reports/Workshop #1/` | Supervised Learning | EDA, feature engineering, and baseline classification models (linear, trees, ensembles). |
| `reports/Workshop #2/` | Deep Learning Basics | Sequence modeling with RNNs/LSTMs and transformers using TensorFlow/PyTorch. |
| `reports/Workshop #3/` | Reinforcement Learning | Reward design, environment setup, and Q-learning or policy gradient methods. |
| `reports/Workshop #4/` | Unsupervised Learning | Autoencoders & VAEs for dimensionality reduction and latent space analysis. |
| `reports/Workshop #5/` | Generative Models (GANs) | Generator/discriminator architecture, training stability, and synthetic data generation. |
| `reports/Workshop #6/` | Domain Adaptation | Advanced transfer learning, model interpretability (attention/saliency), and evaluation. |

---

## Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
# Install dependencies
poetry install

# Activate the environment
poetry shell
```

## Reproducing Results

```bash
# 1. Download and verify the dataset
python download_dataset.py

# 2. Run the data pipeline / baselines (see reports/Workshop #1/ for details)
python -m ml_npl.baselines
```

Train/validation/test splits are fixed and stored in `data/splits.json` — they are generated once and reused across all workshops to prevent data leakage and ensure reproducibility.

---

## Guidelines Followed

* **Reproducibility:** random seeds fixed for numpy, Python, and sklearn/torch at the start of every script.
* **No data leakage:** all statistics and transformations are fit on the training set only.
* **Consistent splits:** `data/splits.json` is generated once and never re-randomized.
* **Metric selection:** given class imbalance in review ratings, accuracy is never used as the sole metric — precision, recall, F1-score, and AUC-ROC are reported where applicable.