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
├── src/                    # Python package: data pipeline, EDA, features, baselines
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

All commands below run from the **`ML_NPL/` directory** (the folder that contains `src/`),
so `from src...` imports resolve:

```bash
cd ML_NPL
```

Pick one of the two environments.

### Option A — Poetry (requires Poetry 2.x)

```bash
poetry install
poetry shell
```

> If you get `The Poetry configuration is invalid: fields ['authors', 'description',
> 'name', 'version'] are required in package mode`, your Poetry is 1.x. The project uses
> the modern `[project]` layout, so upgrade with `pip install -U poetry` (do **not** edit
> `pyproject.toml` to downgrade it). Check with `poetry --version`.

### Option B — Plain venv (no Poetry needed)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas scikit-learn matplotlib seaborn nltk langdetect kagglehub ipykernel
```

## Get the dataset

Either download it through the API, or drop the CSV in by hand — the pipeline prefers a
local file and only hits Kaggle as a fallback.

```bash
# Option 1: via the Kaggle API (needs a token in ~/.kaggle/kaggle.json)
python download_dataset.py          # -> data/raw/disneyland_reviews.csv

# Option 2: by hand — download from
#   https://www.kaggle.com/datasets/arushchillar/disneyland-reviews
# and place the CSV anywhere inside data/raw/ (any filename works)
```

## Reproducing Results

With Poetry, prefix each command with `poetry run`; with the venv, run them directly.

```bash
# 1. Build the reproducible train/val/test split -> data/splits.json
python -m src.preprocessing

# 2. Print a structured dataset summary (counts per class, feature ranges, dtypes)
python -m src.dataset_info

# 3. Open the EDA notebook
jupyter notebook notebooks/EDA.ipynb
#   if it reports "ModuleNotFoundError: src", launch it from ML_NPL/ with:
#   PYTHONPATH=. jupyter notebook notebooks/EDA.ipynb

# 4. Run the baselines (see reports/Workshop #1/ for details) — pending
# python -m src.baselines
```

Train/validation/test splits are fixed and stored in `data/splits.json` — they are generated once and reused across all workshops to prevent data leakage and ensure reproducibility.

---

## Guidelines Followed

* **Reproducibility:** random seeds fixed for numpy, Python, and sklearn/torch at the start of every script.
* **No data leakage:** all statistics and transformations are fit on the training set only.
* **Consistent splits:** `data/splits.json` is generated once and never re-randomized.
* **Metric selection:** given class imbalance in review ratings, accuracy is never used as the sole metric — precision, recall, F1-score, and AUC-ROC are reported where applicable.