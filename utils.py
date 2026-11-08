"""Shared helpers for training and for the Streamlit app."""

import platform
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET = "Revenue"

NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Weekend",
]

# Integer codes such as Browser or Region are identifiers, not quantities.
CATEGORICAL_FEATURES = [
    "Month",
    "VisitorType",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

METRIC_COLUMNS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


_TRUE_FALSE = {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}


def _to_binary(series, column):
    """Accept True/False, 'TRUE'/'FALSE' or 0/1 values; reject anything else."""
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map(_TRUE_FALSE)
    bad = series[mapped.isna()].unique()
    if len(bad):
        raise ValueError(
            f"Column '{column}' must contain only True/False or 1/0. "
            f"Unexpected values: {list(bad[:5])}"
        )
    return mapped.astype(int)


def validate_dataframe(df, require_target=True):
    """Raise ValueError with a readable message if the upload cannot be scored."""
    if df is None or df.empty:
        raise ValueError("The uploaded CSV is empty.")

    needed = FEATURES + ([TARGET] if require_target else [])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))

    nulls = df[needed].isna().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        detail = ", ".join(f"{col} ({n})" for col, n in nulls.items())
        raise ValueError("Empty values found in: " + detail)

    plain_numeric = [c for c in NUMERIC_FEATURES if c != "Weekend"]
    for col in plain_numeric:
        if pd.to_numeric(df[col], errors="coerce").isna().any():
            raise ValueError(f"Column '{col}' must be numeric.")


def split_features_target(df, require_target=True):
    """Return (X, y) from a raw dataframe; y is None when no target is requested."""
    validate_dataframe(df, require_target)

    X = df[FEATURES].copy()
    X["Weekend"] = _to_binary(X["Weekend"], "Weekend")
    y = _to_binary(df[TARGET], TARGET) if TARGET in df.columns else None
    return X, y


def environment_summary():
    """One-line description of where the code is running (used as run evidence)."""
    return (
        f"{platform.system()} {platform.release()} | host {platform.node()} | "
        f"Python {sys.version.split()[0]} | scikit-learn {sklearn.__version__} | "
        f"pandas {pd.__version__}"
    )


def score_model(model, X):
    """Return hard predictions and a continuous score for the positive class."""
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    else:
        y_score = model.decision_function(X)
    return y_pred, y_score


def compute_metrics(y_true, y_pred, y_score):
    """The six metrics required by the assignment."""
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = np.nan  # only one class present in y_true
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def metrics_table(results):
    """results: {model name: metrics dict} -> tidy dataframe."""
    return pd.DataFrame(results).T[METRIC_COLUMNS]
