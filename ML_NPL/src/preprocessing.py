"""Reproducible preprocessing pipeline for the Disneyland reviews corpus.

This module owns everything that turns the *raw* corpus into model-ready
splits: cleaning, the sentiment target, period parsing, and the
train/validation/test split. It is the single place where those decisions
live, so the EDA notebook and the baselines consume the same definitions
instead of re-deriving them.

Pipeline order (see the Workshop #1 report, section 14):

    1-2. drop exact duplicate rows and duplicate review texts  (whole dataset)
    3.   convert the 'missing' period sentinel to NaT           (whole dataset)
    4.   build the sentiment target from Rating                 (whole dataset)
    5.   split 70/15/15, stratified on sentiment               (-> splits.json)

Steps that *learn* parameters (TF-IDF vocabulary, scaling, class weights) do
NOT live here: they belong in features.py and must be fitted on the training
split only, to avoid leaking the test set into the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .dataset import load_reviews
from .eda import (
    RATING_COL,
    TEXT_COL,
    PERIOD_COL,
    parse_period,
    sentiment_from_rating,
)

# Anchored to the package, not the working directory: notebooks run with their
# own folder as cwd, and a relative path would scatter copies around.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Medallion layers. Bronze is owned by src/dataset.py; this module writes the
# two downstream ones.
#
#   bronze/  the Kaggle CSV exactly as published — never edited
#   silver/  cleaned, deduplicated, target attached, period parsed
#   gold/    the frozen train/validation/test split, ready for modelling
#
# Silver is MATERIALISED rather than recomputed per call, and that is the point
# of the layer: gold stores positional indices into silver, so those indices
# are only meaningful against a frozen frame. Recomputing the cleaning on the
# fly would let a change in pandas, or in the dedup order, silently repoint
# every split row at a different review.
DATA_DIR = PROJECT_ROOT / "data"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
DEFAULT_SILVER_PATH = SILVER_DIR / "reviews.parquet"
DEFAULT_SPLITS_PATH = GOLD_DIR / "splits.json"

# The Workshop #1 brief names `data/splits.json` as a required artifact, so the
# split is also written there. Gold stays canonical; this is the spec-compliant
# alias, and both files are byte-identical.
SPEC_SPLITS_PATH = DATA_DIR / "splits.json"

RANDOM_STATE = 42  # frozen across every script, so results stay comparable
TARGET_COL = "sentiment"

# The target and period helpers are defined in eda.py (pure, dependency-free)
# and re-exported here so callers can treat them as part of the preprocessing
# surface without creating a circular import.
__all__ = [
    "RANDOM_STATE",
    "TARGET_COL",
    "DEFAULT_SILVER_PATH",
    "DEFAULT_SPLITS_PATH",
    "SPEC_SPLITS_PATH",
    "build_silver",
    "load_silver",
    "clean_dataset",
    "split_dataset",
    "save_splits",
    "load_splits",
    "build_splits",
    "sentiment_from_rating",
    "parse_period",
]


def clean_dataset(frame: pd.DataFrame, *, scheme: str = "three_class") -> pd.DataFrame:
    """Clean the raw corpus and attach the target, without splitting.

    Everything here is a property of the data, not something learned from it,
    so it is safe to run before the split:

    * drop exact duplicate rows (12 rows, pure redundancy)
    * drop duplicate review texts (24 rows; the leakage vector if left in)
    * turn the 'missing' period sentinel into NaT so it is visible to isna()
    * derive the sentiment target from Rating
    * drop rows with no usable target or no review text
    """
    out = frame.copy()

    # Steps 1-2: remove exact duplicates, then duplicate review bodies.
    out = out.drop_duplicates()
    out = out.drop_duplicates(subset=[TEXT_COL], keep="first")

    # Step 3: the sentinel is a literal string, so isna() misses it.
    out["period"] = parse_period(out[PERIOD_COL])

    # Step 4: derive the target. Rating is the SOURCE of the label and must
    # never be fed back as a feature (see the leakage section of the report).
    out[TARGET_COL] = sentiment_from_rating(out[RATING_COL], scheme=scheme)

    # A review with no target or no text carries no signal for a text model.
    out = out.dropna(subset=[TARGET_COL, TEXT_COL])

    return out.reset_index(drop=True)


def build_silver(
    *,
    scheme: str = "three_class",
    destination: Path | str = DEFAULT_SILVER_PATH,
) -> Path:
    """Clean the bronze corpus and persist it as the silver layer.

    Parquet rather than CSV because silver carries typed columns a CSV cannot
    round-trip: ``sentiment`` is categorical and ``period`` is a monthly
    Period. Writing them as text would force every reader to re-derive them,
    and a reader that forgot would get strings instead.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    clean = clean_dataset(load_reviews(), scheme=scheme)
    # Period has no parquet representation; store the month as a timestamp and
    # let load_silver() restore the type, so the round-trip is lossless.
    out = clean.copy()
    out["period"] = out["period"].dt.to_timestamp()
    out.to_parquet(destination, index=False)
    return destination


def load_silver(
    source: Path | str = DEFAULT_SILVER_PATH,
    *,
    scheme: str = "three_class",
) -> pd.DataFrame:
    """Return the silver layer, building it first if it is not on disk."""
    source = Path(source)
    if not source.exists():
        build_silver(scheme=scheme, destination=source)
    frame = pd.read_parquet(source)
    frame["period"] = frame["period"].dt.to_period("M")
    frame[TARGET_COL] = frame[TARGET_COL].astype("category")
    return frame


def split_dataset(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_STATE,
):
    """Stratified 70/15/15 split on the sentiment target.

    Stratifying on sentiment keeps the 9.4:1 class imbalance identical across
    all three sets; a plain random split would leave the 8.5% negative class
    unevenly represented by chance.

    Returns ``(X_train, X_val, X_test, y_train, y_val, y_test)``.
    """
    features = frame.drop(columns=[TARGET_COL])
    target = frame[TARGET_COL]

    # First cut: hold out (val + test); keep the rest for training.
    holdout = test_size + val_size
    x_train, x_hold, y_train, y_hold = train_test_split(
        features,
        target,
        test_size=holdout,
        random_state=random_state,
        stratify=target,
    )

    # Second cut: divide the hold-out into validation and test.
    test_fraction = test_size / holdout
    x_val, x_test, y_val, y_test = train_test_split(
        x_hold,
        y_hold,
        test_size=test_fraction,
        random_state=random_state,
        stratify=y_hold,
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


def save_splits(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    destination: Path | str = DEFAULT_SPLITS_PATH,
) -> Path:
    """Persist the split as row indices, so every later step reuses them.

    Saving indices rather than copies of the data keeps splits.json small and
    makes the split the single source of truth: baselines load it instead of
    re-splitting randomly.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    splits = {
        "random_state": RANDOM_STATE,
        "train": x_train.index.tolist(),
        "validation": x_val.index.tolist(),
        "test": x_test.index.tolist(),
    }
    payload = json.dumps(splits, indent=2)
    destination.write_text(payload, encoding="utf-8")
    # Mirror to the path the brief asks for, unless that is already the target.
    if destination.resolve() != SPEC_SPLITS_PATH.resolve():
        SPEC_SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SPEC_SPLITS_PATH.write_text(payload, encoding="utf-8")
    return destination


def load_splits(source: Path | str = DEFAULT_SPLITS_PATH) -> dict[str, list[int]]:
    """Load the saved split indices.

    Later work must load the split from here and never re-split randomly, or
    the metrics stop being comparable across workshops.
    """
    source = Path(source)
    with source.open(encoding="utf-8") as file:
        return json.load(file)


def build_splits(
    *,
    scheme: str = "three_class",
    destination: Path | str = DEFAULT_SPLITS_PATH,
) -> pd.DataFrame:
    """Bronze -> silver -> gold, in one call.

    Returns the silver frame the saved split indexes into. The split is read
    back from silver on disk rather than from a frame held in memory, so the
    indices in gold are guaranteed to address the same rows a later run will
    load.
    """
    silver = load_silver(scheme=scheme)
    x_train, x_val, x_test, *_ = split_dataset(silver)
    save_splits(x_train, x_val, x_test, destination=destination)
    return silver


if __name__ == "__main__":
    cleaned = build_splits()
    saved = load_splits()
    total = len(cleaned)
    print(f"cleaned rows: {total:,}")
    for name in ("train", "validation", "test"):
        count = len(saved[name])
        print(f"  {name:<11}: {count:>6,} ({count / total:.1%})")
    print(f"splits saved to {DEFAULT_SPLITS_PATH}")
