"""Supervised baselines for Workshop #1.

Baseline 1 (this file, for now): Logistic Regression on the prepared features.
Random Forest and a third model (Naive Bayes) slot in later through the same
``evaluate`` helper.

Why these choices:

* **Evaluation on validation, test untouched.** The test split is the final
  anchor; touching it now would quietly contaminate it. Baselines are compared
  on validation and test is kept for the very end.
* **macro-F1 is the headline, not accuracy.** 79.5% of reviews are positive, so
  a model that always predicts "positive" scores 79.5% accuracy while being
  useless. macro-F1 weights the 8.5% negative class equally, so it cannot be
  ignored for free. The majority-class baseline is reported precisely to make
  that floor visible.
* **class_weight="balanced"** (computed in features.py) pushes back on the
  9.4:1 imbalance instead of resampling, which would risk leaking across the
  split.

Run it (from ML_NPL/):

    python -m src.baselines
"""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

from .features import extract_features
from .preprocessing import RANDOM_STATE, TARGET_COL, build_splits, load_splits

# Fixed class order so every confusion matrix and report reads the same way.
LABELS = ["negative", "neutral", "positive"]


def load_feature_splits():
    """Rebuild the train/val/test subsets from the frozen split and vectorise.

    Loads the cleaned corpus and the saved indices, slices each subset, then
    hands them to features.extract_features (which fits TF-IDF, the scaler and
    the class weights on the TRAIN split only).
    """
    clean = build_splits()  # idempotent: regenerates splits.json if missing
    splits = load_splits()

    def subset(name):
        rows = clean.loc[splits[name]]
        return rows, rows[TARGET_COL]

    x_train_df, y_train = subset("train")
    x_val_df, y_val = subset("validation")
    x_test_df, y_test = subset("test")

    (
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
        vectorizer,
        scaler,
        class_weights,
    ) = extract_features(x_train_df, x_val_df, x_test_df, y_train, y_val, y_test)

    return {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "class_weights": class_weights,
    }


def evaluate(name, model, x_val, y_val) -> dict:
    """Print a per-class report + confusion matrix; return headline metrics.

    Reusable for every baseline so the models are compared on identical terms.
    """
    y_pred = model.predict(x_val)

    macro_f1 = f1_score(y_val, y_pred, average="macro", labels=LABELS)
    accuracy = (y_pred == y_val.to_numpy()).mean()

    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(f"accuracy : {accuracy:.4f}")
    print(f"macro-F1 : {macro_f1:.4f}  <- headline metric\n")
    print(classification_report(y_val, y_pred, labels=LABELS, zero_division=0))

    print("confusion matrix (rows = true, cols = predicted):")
    matrix = confusion_matrix(y_val, y_pred, labels=LABELS)
    header = "            " + "".join(f"{lab[:8]:>10}" for lab in LABELS)
    print(header)
    for lab, row in zip(LABELS, matrix):
        print(f"{lab:>10}  " + "".join(f"{v:>10,}" for v in row))

    # The matrix and predictions are returned too, so a notebook can draw the
    # heatmap without re-training the model.
    return {
        "model": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion": matrix,
        "labels": LABELS,
    }


def run() -> list[dict]:
    """Train and evaluate the baselines on the validation split."""
    data = load_feature_splits()
    results = []

    # Baseline 0 — majority class. The floor every real model must beat:
    # ~79.5% accuracy with a macro-F1 near 0.30 (it never predicts the
    # minority classes), which is exactly why accuracy alone is misleading.
    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(data["x_train"], data["y_train"])
    results.append(
        evaluate("Baseline 0 - Majority class", majority, data["x_val"], data["y_val"])
    )

    # Baseline 1 — Logistic Regression (linear model required by the PDF).
    # class_weight balanced to fight the imbalance; lbfgs handles the
    # multinomial (3-class) case and the sparse TF-IDF matrix fine.
    logreg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    logreg.fit(data["x_train"], data["y_train"])
    results.append(
        evaluate("Baseline 1 - Logistic Regression", logreg, data["x_val"], data["y_val"])
    )

    # Comparison table (macro-F1 is what ranks them).
    print(f"\n{'=' * 60}\nSUMMARY (validation)\n{'=' * 60}")
    print(f"{'model':<36}{'accuracy':>10}{'macro-F1':>10}")
    for r in sorted(results, key=lambda r: r["macro_f1"], reverse=True):
        print(f"{r['model']:<36}{r['accuracy']:>10.4f}{r['macro_f1']:>10.4f}")

    return results


if __name__ == "__main__":
    run()
