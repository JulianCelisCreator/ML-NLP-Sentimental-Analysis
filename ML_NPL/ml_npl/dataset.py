"""Acquisition and validation of the raw Disneyland reviews corpus.

Workshop #0 deliverable: pull the dataset from Kaggle, verify it matches the
schema the rest of the pipeline assumes, and expose it as a pandas DataFrame.
Every later workshop should build on :func:`load_reviews` instead of reading
the CSV directly, so the cleaning and validation rules live in one place.
"""

from __future__ import annotations

from pathlib import Path

import kagglehub
import pandas as pd
from kagglehub.exceptions import KaggleApiHTTPError

DATASET_HANDLE = "arushchillar/disneyland-reviews"

# Anchored to the package, not to the working directory: notebooks run with
# their own folder as cwd, and a relative path would scatter copies around.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXPORT_PATH = PROJECT_ROOT / "data" / "raw" / "disneyland_reviews.csv"

# The published CSV is Latin-1 encoded. Only 4 of the 42,656 rows carry
# non-ASCII bytes, all of them in Reviewer_Location, but that is enough to make
# the UTF-8 default fail on the whole file.
CSV_ENCODING = "latin-1"

EXPECTED_COLUMNS = (
    "Review_ID",
    "Rating",
    "Year_Month",
    "Reviewer_Location",
    "Review_Text",
    "Branch",
)

RATING_RANGE = (1, 5)

# Year_Month encodes absent dates as this literal string rather than as an
# empty field, so pandas reads them as ordinary values and isna() misses them.
MISSING_SENTINEL = "missing"

_CREDENTIALS_HELP = (
    "Kaggle rejected the request, which almost always means no usable "
    "credentials were found (it answers 404, not 401, to anonymous "
    "requests). kagglehub accepts any of:\n"
    "  - ~/.kaggle/access_token holding the API token\n"
    "  - ~/.kaggle/kaggle.json with 'username' and 'key'\n"
    "  - KAGGLE_API_TOKEN, or KAGGLE_USERNAME plus KAGGLE_KEY\n"
    "Generate one at kaggle.com -> Settings -> API, chmod 600 the file, and "
    "keep it out of the repository."
)

# Kaggle answers 404 for resources an unauthenticated caller cannot see, so a
# missing token is indistinguishable from a bad handle at this layer.
_AUTH_FAILURE_STATUS = frozenset({401, 403, 404})


class MissingKaggleCredentialsError(RuntimeError):
    """Raised when Kaggle refuses the download for lack of valid credentials."""


def download_raw() -> Path:
    """Download the dataset and return the local cache directory."""
    try:
        return Path(kagglehub.dataset_download(DATASET_HANDLE))
    except KaggleApiHTTPError as error:
        status = getattr(error.response, "status_code", None)
        if status in _AUTH_FAILURE_STATUS:
            raise MissingKaggleCredentialsError(_CREDENTIALS_HELP) from error
        raise


def _find_reviews_csv(root: Path) -> Path:
    """Locate the reviews CSV inside the downloaded dataset.

    The dataset ships a single CSV. Picking the largest one rather than
    hard-coding a filename keeps this working if Kaggle renames the file.
    """
    candidates = list(root.rglob("*.csv"))
    if not candidates:
        msg = f"No CSV file found under {root}"
        raise FileNotFoundError(msg)
    return max(candidates, key=lambda path: path.stat().st_size)


def validate(frame: pd.DataFrame) -> None:
    """Fail loudly if the corpus does not match the assumed schema."""
    missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    if missing:
        msg = f"Missing expected columns {missing}; found {list(frame.columns)}"
        raise ValueError(msg)

    low, high = RATING_RANGE
    ratings = frame["Rating"].dropna()
    if not ratings.between(low, high).all():
        observed = sorted(ratings.unique().tolist())
        msg = f"Ratings outside {low}-{high}: observed {observed}"
        raise ValueError(msg)


def load_reviews(*, validate_schema: bool = True) -> pd.DataFrame:
    """Return the raw reviews as a DataFrame, downloading them if needed."""
    csv_path = _find_reviews_csv(download_raw())
    frame = pd.read_csv(csv_path, encoding=CSV_ENCODING)
    if validate_schema:
        validate(frame)
    return frame


def export_csv(
    destination: Path | str = DEFAULT_EXPORT_PATH,
    *,
    overwrite: bool = False,
) -> Path:
    """Write the whole corpus to a CSV inside the project and return its path.

    The Kaggle copy is Latin-1; this export is UTF-8, so every later step reads
    it without the encoding dance. The file is reproducible from the dataset
    handle, so it belongs in .gitignore rather than in a commit.
    """
    destination = Path(destination)
    if destination.exists() and not overwrite:
        msg = f"{destination} already exists; pass overwrite=True to replace it"
        raise FileExistsError(msg)

    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = load_reviews()
    frame.to_csv(destination, index=False, encoding="utf-8")
    return destination


def summary(frame: pd.DataFrame) -> str:
    """Short human-readable description used for the EDA report."""
    ratings = frame["Rating"].value_counts().sort_index()
    majority_share = ratings.max() / len(frame)
    # str.count over non-space runs rather than str.split().str.len(): same
    # numbers, but it survives null review bodies once cleaning introduces them.
    words = frame["Review_Text"].str.count(r"\S+")

    lines = [
        f"rows: {len(frame):,}",
        f"columns: {list(frame.columns)}",
        f"branches: {sorted(frame['Branch'].unique().tolist())}",
        f"rating distribution:\n{ratings}",
        f"majority-class baseline (accuracy to beat): {majority_share:.1%}",
        f"null values per column:\n{frame.isna().sum()}",
        # isna() reports nothing for these: the sentinel is a literal string,
        # and the duplicates are exact repeats that must not straddle the
        # train/test split.
        f"'{MISSING_SENTINEL}' sentinel in Year_Month: "
        f"{(frame['Year_Month'] == MISSING_SENTINEL).sum():,}",
        f"duplicated Review_ID: {frame['Review_ID'].duplicated().sum():,}",
        f"duplicated Review_Text: {frame['Review_Text'].duplicated().sum():,}",
        f"review length in words: median {words.median():.0f}, max {words.max():,}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    reviews = load_reviews()
    print(summary(reviews))
