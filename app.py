from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
DATASET_PATH = HERE / "dataset.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_nb.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.set_page_config(page_title="CellScope | Classification Studio", page_icon="🧬", layout="wide")
st.title("CellScope — Breast Tumour Classification Studio")
st.caption("Machine Learning Assignment 2 | Wisconsin Diagnostic Breast Cancer dataset")
st.markdown(
    "Explore five classification approaches using a held-out labeled test set. "
    "This educational demonstration is **not a medical diagnostic tool**."
)


@st.cache_resource
def fetch_model(model_name):
    # Rebuild only Logistic Regression in the Streamlit runtime to avoid
    # loading a serialized estimator created with a different sklearn version.
    if model_name == "Logistic Regression":
        if not DATASET_PATH.exists():
            raise FileNotFoundError(
                "dataset.csv was not found beside app.py. Add dataset.csv to the GitHub repository root."
            )
        data = pd.read_csv(DATASET_PATH)
        if "target" not in data.columns:
            raise ValueError("dataset.csv must contain a 'target' column.")
        X = data.drop(columns=["target"])
        y = data["target"].astype(int)
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.20, random_state=27, stratify=y
        )
        estimator = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, C=0.8, random_state=27)),
        ])
        estimator.fit(X_train, y_train)
        return estimator

    filename = MODEL_FILES[model_name]
    return joblib.load(MODEL_DIR / filename)


left, right = st.columns([1, 1])
with left:
    chosen = st.selectbox("Choose a classifier", list(MODEL_FILES))
with right:
    uploaded = st.file_uploader("Upload labeled test data (CSV)", type=["csv"])

if uploaded is None:
    st.info("Upload the included `test_data.csv` to calculate evaluation results.")
    st.stop()

try:
    frame = pd.read_csv(uploaded)
except Exception as error:
    st.error(f"Unable to read this CSV: {error}")
    st.stop()

if "target" not in frame.columns:
    st.error("The uploaded file must include the `target` column (0 = malignant, 1 = benign).")
    st.stop()
if not set(frame["target"].dropna().astype(int).unique()).issubset({0, 1}):
    st.error("Target values must be 0 or 1.")
    st.stop()

try:
    estimator = fetch_model(chosen)
    expected = list(estimator.feature_names_in_)
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        st.error("Missing feature columns: " + ", ".join(missing))
        st.stop()
    features = frame[expected]
    truth = frame["target"].astype(int)
    predicted = estimator.predict(features)
    probabilities = estimator.predict_proba(features)[:, 1]
except Exception as error:
    st.error(f"Model evaluation failed: {error}")
    st.stop()

auc = roc_auc_score(truth, probabilities) if truth.nunique() == 2 else None
scores = {
    "Accuracy": accuracy_score(truth, predicted),
    "AUC": auc,
    "Precision": precision_score(truth, predicted, zero_division=0),
    "Recall": recall_score(truth, predicted, zero_division=0),
    "F1 score": f1_score(truth, predicted, zero_division=0),
    "MCC": matthews_corrcoef(truth, predicted),
}

st.subheader(f"Evaluation results — {chosen}")
metric_cols = st.columns(3)
for index, (label, value) in enumerate(scores.items()):
    metric_cols[index % 3].metric(label, "N/A" if value is None else f"{value:.4f}")

chart_col, report_col = st.columns(2)
with chart_col:
    st.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        truth, predicted, labels=[0, 1],
        display_labels=["Malignant", "Benign"],
        cmap="Purples", values_format="d", ax=ax,
    )
    st.pyplot(fig)
    plt.close(fig)
with report_col:
    st.subheader("Classification report")
    st.code(
        classification_report(
            truth, predicted, labels=[0, 1],
            target_names=["Malignant", "Benign"], zero_division=0,
        ),
        language="text",
    )

st.subheader("Compare all five models")
comparison_rows = []
for model_name in MODEL_FILES:
    current = fetch_model(model_name)
    pred = current.predict(features)
    prob = current.predict_proba(features)[:, 1]
    comparison_rows.append({
        "Model": model_name,
        "Accuracy": accuracy_score(truth, pred),
        "AUC": roc_auc_score(truth, prob) if truth.nunique() == 2 else float("nan"),
        "Precision": precision_score(truth, pred, zero_division=0),
        "Recall": recall_score(truth, pred, zero_division=0),
        "F1": f1_score(truth, pred, zero_division=0),
        "MCC": matthews_corrcoef(truth, pred),
    })

st.dataframe(
    pd.DataFrame(comparison_rows).set_index("Model").style.format("{:.4f}"),
    use_container_width=True,
)
