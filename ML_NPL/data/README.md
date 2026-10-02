# Data layers

This directory follows the **medallion architecture**: each layer is derived from the one
above it, and no layer is edited by hand. Everything here is reproducible, so nothing but
this file is tracked in git.

| Layer | Path | What it holds | Produced by |
| --- | --- | --- | --- |
| **Bronze** | `bronze/disneyland_reviews.csv` | The Kaggle corpus exactly as published — 42,656 rows, Latin-1, duplicates and all | `poetry run python download_dataset.py` |
| **Silver** | `silver/reviews.parquet` | Cleaned and conformed: 42,632 rows, deduplicated, `"missing"` turned into `NaT`, sentiment target attached | `poetry run python -m src.preprocessing` |
| **Gold** | `gold/splits.json` | The frozen 70/15/15 stratified split, as row indices into silver | same command |

## Why each boundary is where it is

**Bronze is never touched.** It is the landing zone: if a cleaning rule turns out to be
wrong, the pipeline can be replayed from a known origin instead of re-downloading and
hoping Kaggle still serves the same version.

**Silver is the cleaning boundary — and it is materialised on purpose.** Gold stores
*positional indices* into silver, and those indices only mean anything against a frozen
frame. If the cleaning were recomputed on every run, a change in pandas or in the
deduplication order would silently repoint every split row at a different review, and no
error would be raised. Writing silver once removes that whole class of failure.

Silver is Parquet, not CSV, because it carries typed columns a CSV cannot round-trip:
`sentiment` is categorical and `period` is a monthly Period. `load_silver()` restores both,
so the round-trip is lossless (verified with `DataFrame.equals`).

**Everything above gold is pre-split, and that is deliberate.** Only operations that are
properties of the data live in silver — deduplication, parsing, deriving the target.
Anything *learned* (a TF-IDF vocabulary, class weights, a scaler) is fitted on the training
split alone, inside `src/features.py`. Fitting a vectoriser on all 42,632 reviews would
mean the vocabulary has already read the test set.

## Regenerating from scratch

```bash
cd ML_NPL
rm -rf data/silver data/gold          # bronze can stay; it costs a 31 MB download
poetry run python -m src.preprocessing
```

Deleting bronze too is fine, but then Kaggle credentials are required again — see the
repository README.
