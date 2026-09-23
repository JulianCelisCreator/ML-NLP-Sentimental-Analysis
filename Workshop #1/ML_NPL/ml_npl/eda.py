"""Exploratory analysis helpers for the Disneyland reviews corpus.

Every function takes a DataFrame and returns a DataFrame or Series; nothing is
mutated in place and the raw columns are never overwritten. That keeps the
notebook readable and lets the Workshop #1 preprocessing reuse the same
definitions instead of re-deriving them.

Column roles are resolved from the real schema (see COLUMN_ROLES) rather than
assumed. Two roles the assignment template asks for do not exist here:

* there is no user identifier — Review_ID is unique per review, not per person
* Year_Month has month granularity, so no day-of-week analysis is possible
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd

ID_COL = "Review_ID"
RATING_COL = "Rating"
PERIOD_COL = "Year_Month"
LOCATION_COL = "Reviewer_Location"
TEXT_COL = "Review_Text"
BRANCH_COL = "Branch"

COLUMN_ROLES: dict[str, str] = {
    ID_COL: "identifier (per review, NOT per user)",
    RATING_COL: "ordinal 1-5 score; source of the sentiment target",
    PERIOD_COL: "visit period as 'YYYY-M'; month granularity, no day",
    LOCATION_COL: "reviewer's self-declared country",
    TEXT_COL: "free-text review body; the only model input",
    BRANCH_COL: "which Disneyland park the review is about",
}

MISSING_SENTINEL = "missing"
PERIOD_PATTERN = re.compile(r"^\d{4}-\d{1,2}$")
URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)
NON_ASCII_PATTERN = re.compile(r"[^\x00-\x7F]")
# Ranges covering the common emoji blocks; enough to flag their presence.
EMOJI_PATTERN = re.compile(
    "[\U0001f300-\U0001f9ff\U00002600-\U000027bf\U0001fa70-\U0001faff]"
)
# Sentence-ending punctuation followed by whitespace or end of string. Cheaper
# than nltk.sent_tokenize on 42k rows and accurate enough for a length metric.
SENTENCE_PATTERN = re.compile(r"[.!?]+(?:\s|$)")
# Two deliberately different tokenizations, kept apart because they answer
# different questions: WORD_PATTERN measures raw length (what a reader sees),
# LEXICAL_PATTERN measures vocabulary. Mixing them makes unique/total exceed 1,
# because "well-organized,fun" is one whitespace token but three words.
WORD_PATTERN = re.compile(r"\S+")
LEXICAL_PATTERN = re.compile(r"[a-z']+")


def column_inventory(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, cardinality, nulls and the role it plays."""
    rows = []
    for column in frame.columns:
        values = frame[column]
        n_unique = values.nunique()
        n_null = int(values.isna().sum())
        rows.append(
            {
                "column": column,
                "dtype": str(values.dtype),
                "unique": n_unique,
                "nulls": n_null,
                "pct_null": round(n_null / len(frame) * 100, 2),
                "constant": n_unique <= 1,
                "unique_key": n_unique == len(frame),
                "role": COLUMN_ROLES.get(column, "unmapped"),
            }
        )
    return pd.DataFrame(rows)


def quality_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Count each data-quality issue without dropping anything.

    Treatment is suggested, never applied: the caller decides, and any dropping
    must happen after the train/test split to avoid leaking decisions.
    """
    text = frame[TEXT_COL]
    stripped = text.str.strip()
    words = text.str.count(WORD_PATTERN.pattern)
    ratings = frame[RATING_COL]

    checks = [
        ("null values (any column)", int(frame.isna().sum().sum()),
         "none present; nothing to impute"),
        ("fully duplicated rows", int(frame.duplicated().sum()),
         "drop: identical in every column, pure redundancy"),
        (f"duplicated {ID_COL}", int(frame[ID_COL].duplicated().sum()),
         "inspect: the id is meant to be unique"),
        (f"duplicated {TEXT_COL}", int(text.duplicated().sum()),
         "drop before splitting: same text on both sides inflates test scores"),
        ("empty or whitespace-only reviews", int((stripped == "").sum()),
         "drop: no signal for a text model"),
        ("reviews under 3 words", int((words < 3).sum()),
         "keep but flag: short reviews are legitimate, just low-information"),
        ("reviews over 1000 words", int((words > 1000).sum()),
         "keep: genuine long reviews; truncate at model level, not here"),
        ("symbols/digits only", int((~stripped.str.contains(r"[A-Za-z]", na=False)).sum()),
         "drop: carries no lexical content"),
        ("contains a URL", int(text.str.contains(URL_PATTERN, na=False).sum()),
         "strip the URL, keep the review"),
        ("contains emoji", int(text.str.contains(EMOJI_PATTERN, na=False).sum()),
         "keep: emoji carry sentiment; strip only for bag-of-words models"),
        ("non-ASCII characters", int(text.str.contains(NON_ASCII_PATTERN, na=False).sum()),
         "keep: normal in a multilingual reviewer base"),
        ("ratings outside 1-5", int((~ratings.between(1, 5)).sum()),
         "none present; the scale is clean"),
        (f"'{MISSING_SENTINEL}' sentinel in {PERIOD_COL}",
         int((frame[PERIOD_COL] == MISSING_SENTINEL).sum()),
         "convert to NaT: it is missing data disguised as a category"),
        ("unparseable period (excl. sentinel)",
         int((~frame[PERIOD_COL].str.match(PERIOD_PATTERN, na=False)
              & (frame[PERIOD_COL] != MISSING_SENTINEL)).sum()),
         "none present beyond the sentinel"),
    ]

    report = pd.DataFrame(checks, columns=["issue", "count", "suggested_treatment"])
    report["pct"] = (report["count"] / len(frame) * 100).round(2)
    return report[["issue", "count", "pct", "suggested_treatment"]]


def add_text_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with derived text metrics appended.

    The original columns are left untouched; every new column is additive so
    the raw text stays available for transformer tokenizers later.
    """
    out = frame.copy()
    text = out[TEXT_COL]
    lowered = text.str.lower()

    out["n_chars"] = text.str.len()
    out["n_words"] = text.str.count(WORD_PATTERN.pattern)
    out["n_sentences"] = text.str.count(SENTENCE_PATTERN.pattern).clip(lower=1)
    # findall returns NaN (a float) for a null body, and set(nan) raises. Guard
    # now: Review_Text has no nulls today, but cleaning will introduce them.
    lexical = lowered.str.findall(LEXICAL_PATTERN).map(
        lambda words: words if isinstance(words, list) else []
    )
    out["n_tokens"] = lexical.map(len)
    out["n_unique_words"] = lexical.map(lambda words: len(set(words)))
    # Divided by n_tokens, not n_words: both come from LEXICAL_PATTERN, so the
    # ratio is bounded by 1 as a type/token ratio should be.
    out["unique_ratio"] = (out["n_unique_words"] / out["n_tokens"]).round(4)
    out["mean_word_len"] = (out["n_chars"] / out["n_words"]).round(2)
    out["n_upper"] = text.str.count(r"[A-Z]")
    out["caps_ratio"] = (out["n_upper"] / out["n_chars"]).round(4)
    out["n_exclaim"] = text.str.count(r"!")
    out["n_question"] = text.str.count(r"\?")
    out["has_url"] = text.str.contains(URL_PATTERN, na=False)
    out["has_emoji"] = text.str.contains(EMOJI_PATTERN, na=False)
    return out


def sentiment_from_rating(
    ratings: pd.Series,
    scheme: str = "three_class",
) -> pd.Series:
    """Map the 1-5 rating onto a sentiment label.

    Two schemes are offered because the choice is not obvious for this corpus
    and must be justified, not defaulted:

    * ``three_class``  1-2 negative, 3 neutral, 4-5 positive
    * ``binary``       1-3 negative, 4-5 positive (drops the neutral band)

    The rating is the *source* of the label. Once used here it must never be
    fed back as a model feature — see the leakage notes in the notebook.
    """
    if scheme == "three_class":
        bins = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}
    elif scheme == "binary":
        bins = {1: "negative", 2: "negative", 3: "negative", 4: "positive", 5: "positive"}
    else:
        msg = f"Unknown scheme {scheme!r}; use 'three_class' or 'binary'"
        raise ValueError(msg)
    return ratings.map(bins).astype("category")


def class_balance(labels: pd.Series) -> pd.DataFrame:
    """Absolute and relative frequency per class, plus the imbalance ratio."""
    counts = labels.value_counts()
    table = pd.DataFrame({"count": counts, "pct": (counts / len(labels) * 100).round(2)})
    table["ratio_vs_smallest"] = (counts / counts.min()).round(1)
    return table


def parse_period(periods: pd.Series) -> pd.Series:
    """Parse 'YYYY-M' into a monthly Period, turning the sentinel into NaT."""
    cleaned = periods.where(periods != MISSING_SENTINEL)
    return pd.PeriodIndex(
        pd.to_datetime(cleaned, format="%Y-%m", errors="coerce"), freq="M"
    ).to_series(index=periods.index)


def top_ngrams(
    documents: Sequence[str],
    ngram_range: tuple[int, int] = (1, 1),
    top_n: int = 20,
    stop_words: str | list[str] | None = "english",
) -> pd.DataFrame:
    # pylint: disable=import-outside-toplevel
    """Most frequent n-grams, counted with scikit-learn's tokenizer.

    Imported lazily so that importing this module stays cheap for callers that
    only need the cheaper statistics.
    """
    from sklearn.feature_extraction.text import (  # noqa: PLC0415
        CountVectorizer,
    )

    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words)
    matrix = vectorizer.fit_transform(documents)
    frequencies = np.asarray(matrix.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    return (
        pd.DataFrame({"ngram": terms, "count": frequencies})
        .sort_values("count", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
