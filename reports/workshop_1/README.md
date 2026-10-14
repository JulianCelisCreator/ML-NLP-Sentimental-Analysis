# Workshop #1 — Data Preparation & Supervised Learning Baselines

| File | What it is |
| --- | --- |
| `Workshop_1.pdf` | The brief, as handed out by the course |
| `groupN_workshop1.pdf` | Our report — written outside the repository; add it here once final |

## Where each deliverable lives

| The brief asks for | Location |
| --- | --- |
| Dataset download + integrity check | `ML_NPL/download_dataset.py` (SHA256 against Kaggle) |
| `dataset_info.py` | `ML_NPL/src/dataset_info.py` |
| Preprocessing pipeline | `ML_NPL/src/preprocessing.py`, `ML_NPL/src/features.py` |
| `data/splits.json` | `ML_NPL/data/splits.json` (generated, gitignored) |
| Three classifiers + majority baseline | `ML_NPL/src/baselines.py` |
| Results notebook | `ML_NPL/notebooks/baselines.ipynb` |
| EDA | `ML_NPL/notebooks/eda.ipynb` |

How to run everything: [`docs/PROJECT_REVIEW.md`](../../docs/PROJECT_REVIEW.md).
What is still missing against the brief: [`docs/WORKSHOP1_TODO.md`](../../docs/WORKSHOP1_TODO.md).
