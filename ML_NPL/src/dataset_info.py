"""Scan the dataset and print a structured summary.

Required artifact for Workshop #1: a ``dataset_info.py`` script that scans the
dataset directory and prints sample counts per class, feature ranges, and data
types, so a new team member can characterise the corpus from the command line.

Run it with:

    poetry run python -m src.dataset_info
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .dataset import BRONZE_DIR, load_reviews, summary
from .eda import RATING_COL, add_text_features, class_balance, column_inventory
from .preprocessing import TARGET_COL, clean_dataset

NUMERIC_RANGE_COLUMNS = ["n_words", "n_chars", "n_sentences", "unique_ratio"]


def _section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    """Hash a file in chunks, so a 31 MB corpus never lands in memory twice."""
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_bronze() -> None:
    """File inventory and checksums for the landing zone.

    The brief asks for file counts, sizes and MD5/SHA256 checksums. The point
    is that a teammate can prove they are working from the same bytes: the
    corpus is not in git, so the checksum is the only shared evidence.
    """
    _section("BRONZE LAYER — files, sizes, checksums")
    if not BRONZE_DIR.exists():
        print(f"  {BRONZE_DIR} does not exist — run download_dataset.py first")
        return
    files = sorted(path for path in BRONZE_DIR.iterdir() if path.is_file())
    print(f"  files: {len(files)}")
    for path in files:
        size_mb = path.stat().st_size / 1024**2
        print(f"\n  {path.name}")
        print(f"    size   : {size_mb:,.2f} MB ({path.stat().st_size:,} bytes)")
        print(f"    md5    : {file_digest(path, 'md5')}")
        print(f"    sha256 : {file_digest(path, 'sha256')}")


def main() -> int:
    """Print a structured description of the raw and cleaned corpus."""
    describe_bronze()

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
