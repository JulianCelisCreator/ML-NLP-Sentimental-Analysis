# Workshop #1 — gap analysis

Checked against `ML_NPL/docs/Workshop_1.pdf` on 1 October 2026. Each item says what the
brief asks for, whether it exists, and what is left. Work through it top-down: the
sections are ordered by how much they put the grade at risk.

Current baseline results (validation split, after this session's changes):

| Model | accuracy | AUC-ROC | macro-F1 |
| --- | --- | --- | --- |
| Logistic Regression | 0.7844 | **0.8866** | **0.6258** |
| Multinomial Naive Bayes | **0.8027** | 0.8187 | 0.4318 |
| Random Forest | 0.7961 | 0.8618 | 0.3016 |
| Majority class | 0.7953 | 0.5000 | 0.2953 |

---

## Done in this session

- [x] **`data/splits.json`** — the brief names this exact path. The medallion refactor had
      moved it to `data/gold/`; it is now written to both, byte-identical.
- [x] **AUC-ROC** — explicitly required and previously missing. One-vs-rest, macro. The
      majority baseline scores exactly 0.5000, which is the sanity check that it is
      wired correctly.
- [x] **A third real classifier** — the brief wants three *traditional* classifiers plus
      the floor. Multinomial Naive Bayes added; the majority `DummyClassifier` is the
      reference point, not one of the three.
- [x] **Seeds for Python, numpy and sklearn** at module import, as the brief requires.
- [x] **`n_jobs=-1` removed** from `LogisticRegression` (no effect since sklearn 1.8,
      removed in 1.10).
- [x] **`checkpoints/` and `runs/`** created with READMEs explaining why they are empty.
- [x] **Random Forest comment corrected** — it claimed "few features" against 35,605
      columns, and `max_features` was commented out.

---

## Blocking — do these first

### 1. ~~Re-execute `baselines.ipynb`~~ — done

The notebook now stores exactly what a fresh `python -m src.baselines` produces, verified
line by line. It also gained the AUC column, a fourth confusion matrix in a 2x2 grid, and
a per-class recall table.

That table corrected a claim that had been in the prose: the Random Forest's neutral
recall is **0.000**, not the ~0.02 previously written. It assigns "positive" to all but
**7 of 6,395** validation reviews — functionally the majority baseline with 400 trees
attached.

### 1b. ~~The old reproducibility problem~~ — resolved

The committed notebook still shows the old numbers, and its Random Forest result
(**macro-F1 0.4630**) does not reproduce: a clean run gives **0.3016**. Seed, library
versions and the split were all checked and ruled out — the stored outputs predate the
final state of `src/baselines.py` (the cell shows a Windows path, so it was run elsewhere
and not re-run before committing).

The brief is explicit: *"All reported metric values must be reproducible by running the
submitted code."* This is the single biggest risk in the deliverable.

```bash
cd ML_NPL
poetry run jupyter nbconvert --to notebook --execute --inplace notebooks/baselines.ipynb
```

The notebook also needs a fourth confusion matrix and the AUC column in its table.

### 2. ~~Dataset integrity verification~~ — done

`src/dataset_info.py` now opens with a bronze-layer section: file count, size in bytes and
MB, MD5 and SHA256, hashed in 1 MB chunks. For the current corpus:

```
disneyland_reviews.csv   30.54 MB (32,022,233 bytes)
md5     84f706ba94964ad8fcaa967d442d4482
sha256  ee675f3bfd33b80bd7dfa787e616549633bdd8139b2804170da45273b05af7a5
```

**Still to do:** copy that SHA256 into the README so a teammate can verify their download
against it without running the script.

### 3. The report PDF  ← now the top priority

Not started. The brief requires `groupN_workshop1.pdf` with: executive summary, dataset
description, EDA summary, preprocessing pipeline, baseline results table, and a discussion
of *"what the baseline results reveal about problem difficulty, class balance, and
features"*.

Most of the content already exists in `PROJECT_REVIEW.md`, the EDA notebook and the
baselines notebook — this is assembly, not new analysis.

---

## Required by the brief, still missing

### 4. Augmentation strategy (training set only)

The brief asks for one to be *defined and justified*, not necessarily applied. For text
the honest answer may well be "none, and here is why" — synonym replacement and
back-translation both risk flipping sentiment, which is the label. **Write the
justification down**; silence reads as an omission rather than a decision.

### 5. Cross-validation

Not required by name, but *"hyperparameter tuning"* is, and tuning without CV means
selecting against the validation set until it stops being a held-out set.

There is a measured reason too: bootstrapping the current validation split puts the
macro-F1 95% interval at **[0.6104, 0.6401]** — nearly 3 points wide, because the
negative class has only 543 examples there. Any tuning gain smaller than that is noise.

**The trap:** `extract_features()` fits TF-IDF once over the whole training split. Running
CV on that already-vectorised matrix lets every fold share a vocabulary built from the
other folds. The fix is an sklearn `Pipeline` passed to `cross_val_score` with
`StratifiedKFold`, so the vectoriser is refitted inside each fold.

### 6. ~~Naming: `notebooks/eda.ipynb`~~ — done

Renamed with `git mv`, so the history survives. Every reference in the docs was updated.

---

## Worth doing, not required

### 7. Dimensionality reduction — as an experiment, for the Random Forest only

The brief mentions *"reduce dimensionality if necessary (e.g., with PCA)"*. Two points:

**PCA does not fit here.** It centres the data, which destroys sparsity: the TF-IDF matrix
is 99.75% zeros and would go from 20 MB to **7.9 GB** dense. `TruncatedSVD` (LSA) is the
sparse-safe equivalent.

**And it should not be applied to the linear model.** High-dimensional sparse input is
exactly where regularised linear models work best, and projecting would destroy the
interpretability that is Logistic Regression's main asset here.

**But it is a well-founded hypothesis for the Random Forest.** Its AUC-ROC is 0.8618 — the
probabilities rank reviews well — while its macro-F1 is 0.3016, meaning the argmax decision
collapses onto the majority class. Trees struggle with 35,595 near-empty columns in a way
linear models do not. `TruncatedSVD(300) + RandomForest` is worth one row in an ablation
table.

### 8. Threshold tuning for the Random Forest

Same observation, cheaper fix. An AUC of 0.8618 with a macro-F1 of 0.3016 says the model
*ranks* well and *decides* badly. Moving the decision threshold off argmax could recover a
large part of the gap without changing the model at all. One experiment, potentially the
biggest single gain available.

### 9. An ablation table

Where the "with or without X" comparisons belong — **not in the EDA notebook**, which
analyses data, not models. A separate `notebooks/experiments.ipynb`, one row per
configuration, reporting CV mean ± σ:

| Configuration | macro-F1 (CV) |
| --- | --- |
| TF-IDF only + LogReg | |
| TF-IDF + numeric features + LogReg | |
| … + TruncatedSVD(300) + RandomForest | |
| … + threshold tuning | |

### 10. Tests

`tests/` is empty. `parse_period`, the label schemes and `clean_dataset` all have logic
worth pinning. Not graded, but the row counts (42,656 → 42,632) are asserted nowhere.

---

## Findings worth putting in the report

Two results from this session are the most interesting things the baselines produced, and
both answer the brief's *"what do the results reveal"* question directly.

**Three of the four models beat Logistic Regression on accuracy. All four lose to it on
macro-F1.** Naive Bayes reaches the highest accuracy of any model (0.8027) with a macro-F1
of 0.4318. Ranking by accuracy would have selected the worst classifier available. The
metric was fixed during the EDA, before any result was visible — that is what makes the
conclusion defensible rather than lucky.

**AUC-ROC and macro-F1 disagree about the Random Forest**, and the disagreement is
informative. AUC 0.8618 says its probability ranking is sound; macro-F1 0.3016 says its
decisions are not. Those measure different things — ranking versus the argmax cut — and
the gap points at the threshold, not the model. This is the kind of observation the
discussion section is asking for.
