"""Scan the dataset and print a structured summary.

Required artifact for Workshop #1: a ``dataset_info.py`` script that scans the
dataset directory and prints sample counts per class, feature ranges, and data
types, so a new team member can characterise the corpus from the command line.

Run it with:

    poetry run python -m src.dataset_info
"""

from __future__ import annotations

import pandas as pd

from .dataset import load_reviews, summary
from .eda import RATING_COL, add_text_features, class_balance, column_inventory
from .preprocessing import TARGET_COL, clean_dataset

NUMERIC_RANGE_COLUMNS = ["n_words", "n_chars", "n_sentences", "unique_ratio"]


def _section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main() -> int:
    """Print a structured description of the raw and cleaned corpus."""
    raw = load_reviews()

    _section("RAW CORPUS")
    print(summary(raw))

    _section("COLUMN INVENTORY (dtype, cardinality, nulls, role)")
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(column_inventory(raw).to_string(index=False))

    _section("RATING DISTRIBUTION (source of the target)")
    ratings = raw[RATING_COL].value_counts().sort_index()
    for level, count in ratings.items():
        print(f"  {level} stars: {count:>6,} ({count / len(raw):.1%})")

    # Clean + attach target so class counts reflect the modelling corpus.
    clean = clean_dataset(raw)

    _section("SENTIMENT TARGET (three_class, after cleaning)")
    print(class_balance(clean[TARGET_COL]).to_string())

    _section("NUMERIC FEATURE RANGES (min / mean / max)")
    with_features = add_text_features(clean)
    ranges = with_features[NUMERIC_RANGE_COLUMNS].agg(["min", "mean", "max"]).round(2)
    print(ranges.to_string())

    _section("DATA TYPES")
    print(with_features.dtypes.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
