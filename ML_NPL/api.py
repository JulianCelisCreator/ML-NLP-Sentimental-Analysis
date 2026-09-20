"""Entry point to fetch the corpus and print a first look at it.

Run with: poetry run python api.py
The actual acquisition logic lives in ml_npl.dataset so the notebooks and the
workshop code can reuse it.
"""

from dataset import MissingKaggleCredentialsError, load_reviews, summary


def main() -> int:
    try:
        reviews = load_reviews()
    except MissingKaggleCredentialsError as error:
        print(error)
        return 1

    print(summary(reviews))
    print("\nFirst 5 records:")
    print(reviews.head())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
