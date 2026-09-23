"""Download the Disneyland reviews corpus from Kaggle and save it as a CSV.

Run it with:

    poetry run python download_dataset.py

The file lands in data/raw/disneyland_reviews.csv, encoded as UTF-8.
"""

from ml_npl.dataset import MissingKaggleCredentialsError, export_csv


def main() -> int:
    """Export the corpus and report where it landed."""
    try:
        destination = export_csv(overwrite=True)
    except MissingKaggleCredentialsError as error:
        print(error)
        return 1

    size_mb = destination.stat().st_size / 1024**2
    print(f"Saved {size_mb:.0f} MB to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
