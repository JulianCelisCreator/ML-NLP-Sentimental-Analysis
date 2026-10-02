# src

Python package for the Team 4 sentiment-analysis project. See the
[repository README](../README.md) for setup, layout and project status.

```python
from src.dataset import load_reviews
from src import eda, plots
from src.preprocessing import build_splits, load_splits

reviews = load_reviews()                                   # 42,656 validated rows
clean = build_splits()                                     # clean + split -> data/gold/splits.json
```

## Module responsibilities

| Module | Responsibility |
| --- | --- |
| `dataset.py` | Acquisition: Kaggle download, schema validation, `load_reviews()` (UTF-8) |
| `eda.py` | Analysis only: inventory, quality report, text metrics, n-grams. Owns the pure `sentiment_from_rating` / `parse_period` helpers |
| `plots.py` | EDA figures |
| `preprocessing.py` | Pipeline: clean → target → period → stratified 70/15/15 split → `splits.json` |
| `features.py` | Learned features for the baselines (TF-IDF, scaler, class weights) — fitted on train only |
| `dataset_info.py` | Required script: prints a structured dataset summary |
| `baselines.py` | The three baseline classifiers (Workshop #1, pending) |
| `data_loader.py` | Deprecated shim → import from `preprocessing` instead |
# Data Processing Pipeline

This section describes the data processing pipeline used for the Disneyland Reviews sentiment analysis project.

The pipeline is divided into different stages. Each stage has a specific responsibility to avoid duplicating data processing operations and to ensure reproducibility.

## Pipeline structure

```text
Raw Dataset (Kaggle)
    │
    ▼
dataset.py  — download, validate schema, load_reviews() (UTF-8)
    │
    ▼
preprocessing.py
    │
    ├── Step 1: Remove exact duplicates          (whole dataset)
    ├── Step 2: Remove duplicated review texts   (whole dataset)
    ├── Step 3: Parse period ('missing' -> NaT)  (whole dataset)
    ├── Step 4: Create sentiment target          (whole dataset)
    │
    └── Split dataset (stratified) -> data/gold/splits.json
            ├── 70% Training
            ├── 15% Validation
            └── 15% Testing
                    │
                    ▼
                features.py  (fitted on TRAIN only)
                    │
                    ├── Step 5: Generate text features
                    ├── Step 6: Normalize text
                    ├── Step 7: Fit TF-IDF vocabulary
                    ├── Step 8: Scale numeric features
                    └── Step 9: Compute class weights
```

## `dataset.py`

Acquisition layer: downloads the corpus from Kaggle, validates the schema, and
exposes it as a UTF-8 DataFrame via `load_reviews()`. Every later step reads
through this loader instead of touching the CSV directly.

## `preprocessing.py`

Owns cleaning, the target, and the split — the single place those decisions
live.

### Main steps

- **Step 1 — Remove exact duplicates:** rows identical in every column.
- **Step 2 — Remove duplicated review texts:** the leakage vector if left in.
- **Step 3 — Parse period:** converts the `"missing"` sentinel to `NaT`.
- **Step 4 — Create sentiment:** three-class target derived from `Rating`.
- **Split + save:** stratified 70/15/15 on the target, persisted as row
  indices in `data/gold/splits.json` (with the frozen `random_state`), so later
  work loads the split instead of re-splitting randomly.

Steps 1–4 are properties of the data and run before the split; the duplicate
removal happens first so identical texts cannot straddle the train/test
boundary.

## `features.py`

This module is responsible for transforming the cleaned datasets into features that can be used by the machine learning model.

It receives the splits produced by `preprocessing.py` and learns every
parameter from the training split only.

### Main steps

- **Step 5 — Text features:** reuses the descriptive metrics from `eda.py`.
- **Step 6 — Text normalization:** lowercase and strip URLs, preserving negations.
- **Step 7 — TF-IDF:** fitted only on the training set, then applied to
  validation and test.
- **Step 8 — Scaling:** `StandardScaler` on the numeric features
  (`n_chars` dropped as redundant with `n_words`), fitted on train only.
- **Step 9 — Class weights:** balanced weights computed from `y_train` only.

## Data leakage prevention

The pipeline separates the dataset before learning parameters from the data.

The TF-IDF vectorizer is fitted only with the training data:

```python
vectorizer.fit_transform(X_train["text_normalised"])
```

The validation and test sets are transformed using the vocabulary learned from the training set:

```python
vectorizer.transform(X_val["text_normalised"])
vectorizer.transform(X_test["text_normalised"])
```

Similarly, class weights are calculated only from `y_train`.

This prevents information from the validation and test sets from influencing the training process.