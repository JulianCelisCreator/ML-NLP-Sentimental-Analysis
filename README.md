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
.
├── README.md                 # This file
├── docs/                     # Team working documents
│   ├── PROJECT_REVIEW.md     #   how to run the project, current results, known issues
│   ├── WORKSHOP1_TODO.md     #   Workshop #1 gap analysis against the brief
│   └── TeamsDistribution.pdf #   course team assignment
├── reports/                  # One folder per workshop: the brief PDF and, later, our report
│   ├── workshop_1/           #   Data Preparation & Supervised Learning Baselines
│   ├── workshop_2/           #   Deep Learning Basics
│   ├── workshop_3/           #   Reinforcement Learning
│   ├── workshop_4/           #   Unsupervised Learning
│   ├── workshop_5/           #   Generative Models (GANs)
│   └── workshop_6/           #   Domain Adaptation
└── ML_NPL/                   # The code: one Poetry project shared by every workshop
    ├── src/                  #   Python package (dataset, preprocessing, features, baselines, …)
    ├── notebooks/            #   eda.ipynb, baselines.ipynb
    ├── data/                 #   Medallion layers: bronze / silver / gold (gitignored)
    ├── checkpoints/          #   Saved model weights (from Workshop #2 onward)
    ├── runs/                 #   Training logs (from Workshop #2 onward)
    ├── download_dataset.py   #   Downloads the corpus and verifies its SHA256
    ├── pyproject.toml
    └── poetry.lock
```

---

## Workshops & Focus Areas

| Report | Focus Area | Key Implementation Goals |
| :--- | :--- | :--- |
| `reports/workshop_1/` | Supervised Learning | EDA, feature engineering, and baseline classification models (linear, trees, ensembles). |
| `reports/workshop_2/` | Deep Learning Basics | Sequence modeling with RNNs/LSTMs and transformers using TensorFlow/PyTorch. |
| `reports/workshop_3/` | Reinforcement Learning | Reward design, environment setup, and Q-learning or policy gradient methods. |
| `reports/workshop_4/` | Unsupervised Learning | Autoencoders & VAEs for dimensionality reduction and latent space analysis. |
| `reports/workshop_5/` | Generative Models (GANs) | Generator/discriminator architecture, training stability, and synthetic data generation. |
| `reports/workshop_6/` | Domain Adaptation | Advanced transfer learning, model interpretability (attention/saliency), and evaluation. |

---

## Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management.
Every command below runs from `ML_NPL/`.

```bash
cd ML_NPL
poetry install                 # creates the environment and installs the lock file
source .venv/bin/activate      # optional; otherwise prefix commands with `poetry run`
```

## Reproducing Results

```bash
# 1. Download the corpus into data/bronze/ and verify its SHA256 (needs Kaggle credentials)
poetry run python download_dataset.py

# 2. Clean and split: data/silver/ and data/gold/splits.json
poetry run python -m src.preprocessing

# 3. Train and evaluate the four Workshop #1 baselines on the validation split
poetry run python -m src.baselines
```

No Kaggle account? Place Kaggle's `DisneylandReviews.csv` in `ML_NPL/data/bronze/` by hand;
step 1 then only verifies it. The file must hash to
`3c5eb28455a5156bbe337b3768fdd422a3311c1a91cb33214682bf80419e885c`.

Train/validation/test splits are fixed and stored in `data/gold/splits.json`. They are
generated once and reused across all workshops, to prevent data leakage and keep results
comparable.

---

## Guidelines Followed

* **Reproducibility:** random seeds fixed for Python and numpy, and `random_state=42` passed to every sklearn estimator.
* **No data leakage:** all statistics and transformations are fit on the training set only.
* **Consistent splits:** `data/gold/splits.json` is generated once and never re-randomized.
* **Medallion layers:** `data/` separates bronze (as published), silver (cleaned, materialised) and gold (the frozen split). See [`ML_NPL/data/README.md`](ML_NPL/data/README.md).
* **Metric selection:** given class imbalance in review ratings, accuracy is never used as the sole metric — precision, recall, F1-score, and AUC-ROC are reported where applicable.