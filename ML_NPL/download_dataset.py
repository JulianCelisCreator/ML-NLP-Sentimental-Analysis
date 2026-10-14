"""Download the Disneyland reviews corpus from Kaggle into the bronze layer.

Run it with:

    poetry run python download_dataset.py

The file lands in data/bronze/DisneylandReviews.csv, a byte-for-byte copy of
the Kaggle file (Latin-1). That is the bronze layer: the corpus as published,
never edited downstream. If the file is already there it is verified, not
downloaded again.
"""

from src.dataset import (
    DEFAULT_EXPORT_PATH,
    EXPECTED_SHA256,
    MissingKaggleCredentialsError,
    export_csv,
    file_digest,
)


def main() -> int:
    """Export the corpus if missing, then check it against the Kaggle checksum."""
    destination = DEFAULT_EXPORT_PATH
    if destination.exists():
        print(f"Already present: {destination}")
    else:
        try:
            export_csv(destination)
        except MissingKaggleCredentialsError as error:
            print(error)
            return 1
        size_mb = destination.stat().st_size / 1024**2
        print(f"Saved {size_mb:.0f} MB to {destination}")

    if file_digest(destination) != EXPECTED_SHA256:
        print(
            "WARNING: the SHA256 does not match the published Kaggle file.\n"
            f"Delete {destination} and run this script again."
        )
        return 1
    print("SHA256 matches the published Kaggle file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
