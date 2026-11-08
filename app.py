"""Streamlit app: evaluate pre-trained classifiers on an uploaded test CSV."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
)

from utils import (
    FEATURES,
    METRIC_COLUMNS,
    MODEL_FILES,
    TARGET,
    compute_metrics,
    metrics_table,
    score_model,
    split_features_target,
)

ROOT = Path(__file__).resolve().parent
SAMPLE_FILE = ROOT / "test_data.csv"
INFO_FILE = ROOT / "model" / "model_info.json"

st.set_page_config(page_title="Shopper Purchase Predictor", page_icon="🛒", layout="wide")


@st.cache_resource
def load_model(name):
    return joblib.load(ROOT / "model" / MODEL_FILES[name])


@st.cache_data
def evaluate(name, df):
    """Score one saved model on the uploaded data; nothing is retrained."""
    X, y = split_features_target(df)
    y_pred, y_score = score_model(load_model(name), X)
    return y, y_pred, y_score


def version_note():
    """Warn when the models were trained with a different scikit-learn."""
    if not INFO_FILE.exists():
        return None
    trained = json.loads(INFO_FILE.read_text()).get("scikit_learn")
    if trained and trained != sklearn.__version__:
        return (
            f"Models were trained with scikit-learn {trained}, but this app runs "
            f"{sklearn.__version__}. If predictions fail to load, retrain with "
            "`python train_models.py` in the same environment."
        )
    return None


st.title("🛒 Online Shopper Purchase Intention")
st.write(
    "Predict whether a website visit ends in a purchase (`Revenue`). "
    "Pick a saved model, optionally upload your own labelled test CSV, and "
    "inspect how it performs."
)

with st.sidebar:
    st.header("1. Test data")
    uploaded = st.file_uploader("Upload test CSV", type="csv")
    st.header("2. Model")
    model_name = st.selectbox("Choose a classifier", list(MODEL_FILES))
    st.caption(f"The CSV needs these columns plus `{TARGET}`:")
    st.code(", ".join(FEATURES), language=None)

if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
    except Exception:
        st.error("Could not read that file as a CSV. Please upload a valid .csv file.")
        st.stop()
    st.caption(f"Showing results for **{uploaded.name}**.")
elif SAMPLE_FILE.exists():
    data = pd.read_csv(SAMPLE_FILE)
    st.caption("Showing the bundled test set. Upload a CSV in the sidebar to replace it.")
else:
    st.info("Upload a test CSV in the sidebar to begin.")
    st.stop()

note = version_note()
if note:
    st.warning(note)

try:
    y_true, y_pred, y_score = evaluate(model_name, data)
except ValueError as err:
    st.error(str(err))
    st.stop()
except FileNotFoundError:
    st.error("Saved model files are missing. Run `python train_models.py` first.")
    st.stop()
except Exception as err:  # e.g. a model pickled by an incompatible library version
    st.error(f"Could not score the data with {model_name}: {err}")
    st.stop()

st.subheader(f"{model_name} on {len(data):,} rows")
with st.expander("Preview uploaded data"):
    st.dataframe(data.head(20), width="stretch")

metrics = compute_metrics(y_true, y_pred, y_score)
columns = st.columns(len(METRIC_COLUMNS))
for col, key in zip(columns, METRIC_COLUMNS):
    col.metric(key, "n/a" if pd.isna(metrics[key]) else f"{metrics[key]:.3f}")

tab_cm, tab_report, tab_roc, tab_all = st.tabs(
    ["Confusion matrix", "Classification report", "ROC curve", "Compare all models"]
)

with tab_cm:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred, labels=[0, 1]),
        display_labels=["No purchase", "Purchase"],
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(model_name)
    st.pyplot(fig)

with tab_report:
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["No purchase", "Purchase"],
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).T.round(3), width="stretch")

with tab_roc:
    if y_true.nunique() < 2:
        st.warning("ROC needs both classes in the uploaded target column.")
    else:
        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(y_true, y_score, ax=ax, name=model_name)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        st.pyplot(fig)

with tab_all:
    results = {}
    for name in MODEL_FILES:
        yt, yp, ys = evaluate(name, data)
        results[name] = compute_metrics(yt, yp, ys)
    table = metrics_table(results)
    st.dataframe(
        table.style.format("{:.3f}").highlight_max(axis=0, color="#cfe8cf"),
        width="stretch",
    )
    st.caption("Best value per metric is highlighted. Models are evaluated, not retrained.")
