"""Supervised baselines for Workshop #1.

Four models, all evaluated through the same ``evaluate`` helper: a
majority-class floor, Logistic Regression (linear), Random Forest (tree
ensemble) and Multinomial Naive Bayes (the model of our choice).

Why these choices:

* **Evaluation on validation, test untouched.** The test split is the final
  anchor; touching it now would quietly contaminate it. Baselines are compared
  on validation and test is kept for the very end.
* **macro-F1 is the headline, not accuracy.** 79.5% of reviews are positive, so
  a model that always predicts "positive" scores 79.5% accuracy while being
  useless. macro-F1 weights the 8.5% negative class equally, so it cannot be
  ignored for free. The majority-class baseline is reported precisely to make
  that floor visible.
* **Balanced class weights** (computed from y_train in features.py) push back
  on the 9.4:1 imbalance instead of resampling, which would risk leaking
  across the split.

Run it (from ML_NPL/):

    python -m src.baselines                # validation: model comparison
    python -m src.baselines --split test   # test: final numbers, for Workshop #2's table
"""

from __future__ import annotations

import argparse
import random

import numpy as np
from scipy.sparse import hstack
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler

from .features import NUMERIC_FEATURES, extract_features
from .preprocessing import RANDOM_STATE, TARGET_COL, build_splits, load_splits

# The brief asks for seeds to be set for Python, numpy and sklearn at the start
# of every script. sklearn has no global seed: it reads numpy's when a
# random_state is not given, and every estimator here passes RANDOM_STATE
# explicitly anyway.
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Fixed class order so every confusion matrix and report reads the same way.
LABELS = ["negative", "neutral", "positive"]


def load_feature_splits():
    """Rebuild the train/val/test subsets from the frozen split and vectorise.

    Loads the cleaned corpus and the saved indices, slices each subset, then
    hands them to features.extract_features (which fits TF-IDF, the scaler and
    the class weights on the TRAIN split only).
    """
    clean = build_splits()  # writes silver/gold only if they are missing
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
        _vectorizer,  # fitted objects are returned for reuse; unused here
        _scaler,
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

    # AUC-ROC one-vs-rest, required by the brief. It needs probabilities, not
    # labels, so a model without predict_proba simply reports nothing rather
    # than a fabricated number.
    auc = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_val)
        auc = roc_auc_score(
            y_val, probabilities, multi_class="ovr", average="macro", labels=LABELS
        )

    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(f"accuracy : {accuracy:.4f}")
    print(f"AUC-ROC  : {auc:.4f} (one-vs-rest, macro)" if auc is not None
          else "AUC-ROC  : n/a (model exposes no probabilities)")
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
        "auc_roc": auc,
        "macro_f1": macro_f1,
        "confusion": matrix,
        "labels": LABELS,
    }


def _scale_for_naive_bayes(x_train, x_eval):
    """Min-max scale the feature matrix to [0, 1] without densifying it."""
    x_train, x_eval = x_train.tocsc(), x_eval.tocsc()
    n_text = x_train.shape[1] - len(NUMERIC_FEATURES)

    text_scaler = MaxAbsScaler()
    numeric_scaler = MinMaxScaler()
    train = hstack([
        text_scaler.fit_transform(x_train[:, :n_text]),
        numeric_scaler.fit_transform(x_train[:, n_text:].toarray()),
    ]).tocsr()
    val = hstack([
        text_scaler.transform(x_eval[:, :n_text]),
        numeric_scaler.transform(x_eval[:, n_text:].toarray()),
    ]).tocsr()
    return train, val


def run(split: str = "validation") -> list[dict]:
    """Train on train, evaluate on `split` ("validation" by default, or "test")."""
    data = load_feature_splits()
    key = {"validation": "val", "test": "test"}[split]
    x_eval, y_eval = data[f"x_{key}"], data[f"y_{key}"]
    results = []

    # Baseline 0 — majority class. The floor every real model must beat:
    # ~79.5% accuracy with a macro-F1 near 0.30 (it never predicts the
    # minority classes), which is exactly why accuracy alone is misleading.
    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(data["x_train"], data["y_train"])
    results.append(
        evaluate("Baseline 0 - Majority class", majority, x_eval, y_eval)
    )

    # Baseline 1 — Logistic Regression (linear model required by the PDF).
    # Balanced class weights (fitted on y_train in features.py) fight the
    # imbalance; lbfgs handles the multinomial (3-class) case and the sparse
    # TF-IDF matrix fine.
    logreg = LogisticRegression(
        max_iter=1000,
        class_weight=data["class_weights"],
        random_state=RANDOM_STATE,
    )
    logreg.fit(data["x_train"], data["y_train"])
    results.append(
        evaluate("Baseline 1 - Logistic Regression", logreg, x_eval, y_eval)
    )

    # Baseline 2 — Random Forest, the tree-based ensemble the brief requires.
    #
    # max_features is left at the sklearn default (sqrt): with 35,605 columns,
    # 99.75% of them zero, sampling a subset per split is what keeps the trees
    # decorrelated. balanced_subsample reweights inside each bootstrap sample
    # rather than once globally, which suits bagging. max_depth caps growth so
    # a tree cannot memorise single reviews.
    rndmforestcl = RandomForestClassifier(
        n_estimators=400,
        max_depth=50,
        criterion="gini",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rndmforestcl.fit(data["x_train"], data["y_train"])
    results.append(
        evaluate("Baseline 2 - Random Forest", rndmforestcl, x_eval, y_eval)
    )

    # Baseline 3 — Multinomial Naive Bayes, the "model of your choice" slot.
    #
    # It is the classical text-classification baseline and a genuinely
    # different family: generative and probabilistic, where the other two are
    # discriminative. It cannot take negative inputs, and StandardScaler
    # produces them, so every column is re-scaled to [0, 1] here, fitted on
    # TRAIN only like every other learned transform.
    #
    # The matrix stays sparse: densifying it costs ~17 GB. Every TF-IDF column
    # has a minimum of 0, so MaxAbsScaler (sparse-safe) gives the same result
    # as MinMaxScaler on that block; only the 10 numeric columns go dense.
    x_train_nb, x_eval_nb = _scale_for_naive_bayes(data["x_train"], x_eval)

    naive_bayes = MultinomialNB()
    naive_bayes.fit(x_train_nb, data["y_train"])
    results.append(
        evaluate("Baseline 3 - Multinomial Naive Bayes", naive_bayes, x_eval_nb, y_eval)
    )

    # Comparison table (macro-F1 is what ranks them).
    print(f"\n{'=' * 60}\nSUMMARY ({split})\n{'=' * 60}")
    print(f"{'model':<38}{'accuracy':>10}{'AUC-ROC':>10}{'macro-F1':>10}")
    for r in sorted(results, key=lambda r: r["macro_f1"], reverse=True):
        auc_text = f"{r['auc_roc']:>10.4f}" if r["auc_roc"] is not None else f"{'n/a':>10}"
        print(f"{r['model']:<38}{r['accuracy']:>10.4f}{auc_text}{r['macro_f1']:>10.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Workshop #1 baselines")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    run(parser.parse_args().split)
