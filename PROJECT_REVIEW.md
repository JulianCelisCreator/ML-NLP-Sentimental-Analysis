# Project Review — Execution Guide and State

**Team 4 — Sentiment Analysis (NLP).** Machine Learning 2026-III, Universidad Distrital
Francisco José de Caldas.

Last verified 1 October 2026. Every command below was executed on this checkout before
being written down, and every number comes from that run.

**Pending work lives in [`WORKSHOP1_TODO.md`](WORKSHOP1_TODO.md)**, checked against
`ML_NPL/docs/Workshop_1.pdf`. This document covers how to run the project and what state
it is in; that one covers what is left.

---

# Part A — How to run the project

## 0. Prerequisites

Python 3.12, Poetry 2.2, and a Kaggle account for the initial download.

```bash
cd ML_NPL
poetry install
```

## 1. Kaggle credentials

kaggle.com → Settings → API → **Create New Token**, then:

```bash
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

`~/.kaggle/access_token` holding the raw token works too. Without credentials Kaggle
answers **404, not 401**, so the failure reads as a wrong dataset name; `src/dataset.py`
intercepts it and prints the real cause.

Credentials are needed **only for the first download**. `load_reviews()` prefers a local
CSV in `data/bronze/` and falls back to Kaggle only when nothing is there.

## 2. The pipeline

Run from `ML_NPL/`. Each step writes what the next one reads.

| # | Command | Produces | Verified result |
| --- | --- | --- | --- |
| 1 | `poetry run python download_dataset.py` | `data/bronze/disneyland_reviews.csv` | 30.54 MB, 42,656 rows |
| 2 | `poetry run python -m src.dataset_info` | checksums + structured summary | sha256 `ee675f3b…05af7a5` |
| 3 | `poetry run python -m src.preprocessing` | `data/silver/reviews.parquet`, `data/gold/splits.json` (+ the spec alias `data/splits.json`) | 42,632 cleaned → 29,842 / 6,395 / 6,395 |
| 4 | `poetry run python -m src.baselines` | the four baselines, evaluated on validation | see Part C |

Step 3 removes 24 rows — 12 fully duplicated rows and 12 further duplicate review texts —
**before** the split, which is what stops identical text from landing on both sides.

## 3. The notebooks

```bash
poetry run jupyter nbconvert --to notebook --execute --inplace notebooks/eda.ipynb
poetry run jupyter nbconvert --to notebook --execute --inplace notebooks/baselines.ipynb
```

`eda.ipynb` is the Workshop #0 deliverable (53 cells, 15 sections). `baselines.ipynb`
presents the Workshop #1 results and trains nothing itself — it calls `src.baselines.run()`
so the stored numbers are reproducible from the command line.

---

# Part B — How the project is organised

## Data: medallion layers

```
data/
├── bronze/disneyland_reviews.csv   the Kaggle corpus as published, never edited
├── silver/reviews.parquet          cleaned, deduplicated, target attached
└── gold/splits.json                the frozen 70/15/15 stratified split
```

Silver is **materialised, not recomputed**, and that is the point. Gold stores positional
indices into silver, and those indices only mean anything against a frozen frame: if the
cleaning ran again on every call, a change in pandas or in the dedup order would silently
repoint every split row at a different review, with no error raised.

Parquet rather than CSV because silver carries typed columns a CSV cannot round-trip —
`sentiment` is categorical and `period` is a monthly Period. The round-trip is verified
lossless with `DataFrame.equals`.

`data/splits.json` is also written, byte-identical to the gold copy, because the Workshop
#1 brief names that exact path as a required artifact.

Full detail in [`ML_NPL/data/README.md`](ML_NPL/data/README.md).

## Code

```
ML_NPL/src/
├── dataset.py         bronze: Kaggle download, schema validation, load_reviews()
├── preprocessing.py   silver + gold: cleaning, target, the 70/15/15 split
├── features.py        TF-IDF + scaled numeric features, class weights
├── baselines.py       the four Workshop #1 models and their evaluation
├── eda.py             exploratory statistics, no matplotlib dependency
├── plots.py           figures on a colour-blind-validated palette
├── dataset_info.py    checksums and a structured corpus summary
└── data_loader.py     deprecated shim -> preprocessing
```

`RANDOM_STATE = 42` is defined once in `preprocessing.py` and imported everywhere.
`baselines.py` also seeds Python and numpy at import, as the brief requires.

## The leakage boundary

Everything **learned** is fitted on the training split alone, inside `features.py`: the
TF-IDF vocabulary, the numeric scaler, the class weights. Validation and test are only
ever transformed. Everything above the split — deduplication, parsing, deriving the target
— is a property of the data, not of a sample, and so runs on the whole corpus.

---

# Part C — Current results

Validation split, 6,395 reviews. Reproduced from a clean run; the notebook stores the same
numbers.

| Model | accuracy | AUC-ROC | macro-F1 |
| --- | --- | --- | --- |
| Logistic Regression | 0.7844 | **0.8866** | **0.6258** |
| Multinomial Naive Bayes | **0.8027** | 0.8187 | 0.4318 |
| Random Forest | 0.7961 | 0.8618 | 0.3016 |
| Majority class | 0.7953 | 0.5000 | 0.2953 |

Per-class recall, which is what the headline numbers hide:

| Model | negative | neutral | positive |
| --- | --- | --- | --- |
| Logistic Regression | **0.713** | **0.495** | 0.836 |
| Multinomial Naive Bayes | 0.173 | 0.086 | 0.978 |
| Random Forest | 0.009 | **0.000** | 1.000 |
| Majority class | 0.000 | 0.000 | 1.000 |

## What the comparison shows

**Three of four models beat Logistic Regression on accuracy. All four lose to it on
macro-F1.** Ranking by accuracy would have selected one of the weaker classifiers. The
metric was fixed during the EDA, before any result existed — which is what makes the
conclusion defensible rather than lucky.

**The Random Forest is the majority baseline with 400 trees attached.** It assigns
"positive" to all but **7 of 6,395** validation reviews; neutral recall is 0.000, not
merely low. `class_weight="balanced_subsample"` is set and does not prevent it —
reweighting inside bootstrap samples is not enough against 35,595 sparse features.

**AUC-ROC and macro-F1 disagree about that model, and the disagreement is useful.** AUC
0.8618 says its probabilities rank reviews well; macro-F1 0.3016 says its argmax decisions
do not. Different things are being measured — ranking versus the decision cut — so the gap
points at the threshold, not at the model. The majority baseline's AUC of exactly 0.5000
is the sanity check that the metric is wired correctly.

**The confusion concentrates on neutral/positive, not negative/positive.** Logistic
Regression misplaces 690 positives as neutral and 172 neutrals as positive, but confuses
negatives with positives only 31 times. A 3-star review reads much like a 4-star one; the
models reproduce an ambiguity that is in the data.

---

# Part D — Known issues

Resolved items have moved to [`WORKSHOP1_TODO.md`](WORKSHOP1_TODO.md). What remains open
in the repository itself:

### Files duplicated byte for byte

| | |
| --- | --- |
| `reports/Workshop #1/Workshop_1.pdf` | `ML_NPL/docs/Workshop_1.pdf` |
| `TeamsDistribution.pdf` | `ML_NPL/docs/TeamsDistribution (1).pdf` |

`ML_NPL/docs/` is untracked while `reports/` is committed, so the copies will drift. Pick
one location.

### Two directories named `reports/`

`reports/` at the repository root holds the workshop PDFs and is tracked;
`ML_NPL/reports/` holds generated figures and is gitignored. Same name, opposite purpose.

### Lint

`pylint src/ download_dataset.py` scores **9.60/10**. What remains is stylistic and worth
a decision rather than drift: `src/features.py` uses `X_train` / `X_val` / `X_test`, which
is standard ML notation pylint does not know about. Either silence the rule in that file
or rename — leaving ten warnings is the worst of the three options.

### `tests/` is empty

`parse_period`, the label schemes and `clean_dataset` all carry logic worth pinning. The
row counts (42,656 → 42,632) are asserted nowhere.
