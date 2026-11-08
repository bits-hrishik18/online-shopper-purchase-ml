"""Train the five classifiers, save them, and write the held-out test set.

Run:  python train_models.py
      python train_models.py --output-dir ~/ml_run     # leaves the repo files untouched
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from utils import (
    CATEGORICAL_FEATURES,
    MODEL_FILES,
    NUMERIC_FEATURES,
    TARGET,
    compute_metrics,
    environment_summary,
    metrics_table,
    score_model,
    split_features_target,
)

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "online_shoppers_intention.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def make_preprocessor():
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_models():
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=20, random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel="rbf", random_state=RANDOM_STATE),
    }
    return {
        name: Pipeline([("prep", make_preprocessor()), ("clf", clf)])
        for name, clf in classifiers.items()
    }


def main(output_dir):
    print("Environment:", environment_summary())
    raw = pd.read_csv(DATA_FILE)
    before = len(raw)
    raw = raw.drop_duplicates().reset_index(drop=True)
    print(f"Rows: {before} -> {len(raw)} after dropping exact duplicates")

    X, y = split_features_target(raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"Positive rate - train {y_train.mean():.3f}, test {y_test.mean():.3f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"
    model_dir.mkdir(exist_ok=True)

    # The uploaded test file keeps the raw column format plus the target.
    test_df = raw.loc[X_test.index].copy()
    test_df.to_csv(output_dir / "test_data.csv", index=False)

    results = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        joblib.dump(model, model_dir / MODEL_FILES[name])
        y_pred, y_score = score_model(model, X_test)
        results[name] = compute_metrics(y_test, y_pred, y_score)
        print(f"Trained and saved: {name}")

    # The app compares these versions with the ones it runs on.
    info = {"scikit_learn": sklearn.__version__, "pandas": pd.__version__}
    (model_dir / "model_info.json").write_text(json.dumps(info, indent=2))

    table = metrics_table(results)
    table.round(4).to_csv(output_dir / "metrics.csv", index_label="ML Model Name")
    print("\n" + table.round(4).to_string())
    print(f"\nTarget column: {TARGET} | Output folder: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT,
        help="where model/, test_data.csv and metrics.csv are written (default: repo root)",
    )
    main(parser.parse_args().output_dir.expanduser().resolve())
