"""Feature engineering for the classical baselines.

Everything that is *learned* from data lives here and is fitted on the
training split only: the TF-IDF vocabulary, the numeric scaler, and the class
weights. Validation and test are transformed with those fitted objects, never
re-fitted — that is the leakage boundary described in the Workshop #1 report.

Input splits come from :mod:`src.preprocessing` (``split_dataset`` /
``load_splits``). The descriptive text metrics are reused from :mod:`src.eda`
so the notebook and the model see the same definitions.
"""

from __future__ import annotations

import re

import numpy as np

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from .eda import add_text_features, TEXT_COL

# Numeric text metrics fed to the scaler. n_chars is intentionally omitted:
# it correlates 1.00 with n_words (see the leakage section of the report), so
# keeping both would just destabilise the linear models.
NUMERIC_FEATURES = [
    "n_words",
    "n_sentences",
    "n_tokens",
    "n_unique_words",
    "unique_ratio",
    "mean_word_len",
    "n_upper",
    "caps_ratio",
    "n_exclaim",
    "n_question",
]


def normalize_text(text):
    """
    Step 6: Normalize text by converting to lowercase
    and removing URLs.
    """

    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    return text


def extract_features(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
):
    """
    Generate text and numerical features using the training,
    validation and test datasets.
    """

    # --- PHASE 3: Features ---

    # Step 5: Add text features
    X_train = add_text_features(X_train.copy())
    X_val = add_text_features(X_val.copy())
    X_test = add_text_features(X_test.copy())

    # Step 6: Normalize text
    X_train["text_normalised"] = X_train[TEXT_COL].apply(
        normalize_text
    )

    X_val["text_normalised"] = X_val[TEXT_COL].apply(
        normalize_text
    )

    X_test["text_normalised"] = X_test[TEXT_COL].apply(
        normalize_text
    )

    # --- PHASE 4: Learn parameters ---

    # Step 7: TF-IDF vectorization
    vectorizer = TfidfVectorizer()

    # Learn vocabulary only from training data
    X_train_tfidf = vectorizer.fit_transform(
        X_train["text_normalised"]
    )

    # Apply the learned vocabulary to validation and test
    X_val_tfidf = vectorizer.transform(
        X_val["text_normalised"]
    )

    X_test_tfidf = vectorizer.transform(
        X_test["text_normalised"]
    )

    # Step 8: Scale numerical features (NUMERIC_FEATURES; n_chars dropped as
    # redundant with n_words).
    scaler = StandardScaler()

    # Learn scaling parameters only from training data
    X_train_numeric = scaler.fit_transform(
        X_train[NUMERIC_FEATURES]
    )

    # Apply the learned scaling to validation and test
    X_val_numeric = scaler.transform(
        X_val[NUMERIC_FEATURES]
    )

    X_test_numeric = scaler.transform(
        X_test[NUMERIC_FEATURES]
    )

    # Convert numerical features to sparse matrices
    # so they can be combined with TF-IDF.
    X_train_numeric = np.asarray(X_train_numeric)
    X_val_numeric = np.asarray(X_val_numeric)
    X_test_numeric = np.asarray(X_test_numeric)

    # Combine TF-IDF and numerical features
    X_train_features = hstack([
        X_train_tfidf,
        X_train_numeric
    ])

    X_val_features = hstack([
        X_val_tfidf,
        X_val_numeric
    ])

    X_test_features = hstack([
        X_test_tfidf,
        X_test_numeric
    ])

    # Step 9: Manage class imbalance
    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights_dict = dict(
        zip(classes, weights)
    )

    return (
        X_train_features,
        X_val_features,
        X_test_features,
        y_train,
        y_val,
        y_test,
        vectorizer,
        scaler,
        class_weights_dict
    )