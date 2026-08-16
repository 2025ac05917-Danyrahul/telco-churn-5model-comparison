"""
Streamlit app for Telco customer churn classification with 5 models
Usage:   streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="Churn Classifier Demo", layout="wide")

MODEL_DIR = Path(__file__).resolve().parent / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    numeric_features = joblib.load(MODEL_DIR / "numeric_features.joblib")
    categorical_features = joblib.load(MODEL_DIR / "categorical_features.joblib")
    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}
    return preprocessor, label_encoder, numeric_features, categorical_features, models


preprocessor, label_encoder, numeric_features, categorical_features, models = load_artifacts()

st.title("📊 Customer Churn — Multi-Model Classification Demo")
st.caption(
    "Upload test data, pick a trained model, and view evaluation metrics, "
    "confusion matrix, and classification report."
)

# ---------------------------------------------------------------------
# a. Dataset upload
# ---------------------------------------------------------------------
st.sidebar.header("1️⃣ Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

# ---------------------------------------------------------------------
# b. Model selection dropdown
# ---------------------------------------------------------------------
st.sidebar.header("2️⃣ Select Model")
selected_model_name = st.sidebar.selectbox("Choose a classifier", list(models.keys()))

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Expected columns: same feature columns as the training data, "
    "plus the true label column **Churn** (Yes/No) if you want metrics computed."
)

if uploaded_file is None:
    st.info("👈 Upload the provided `test_data.csv` from the sidebar to get started.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.subheader("Preview of Uploaded Data")
st.dataframe(df.head(10))

has_labels = "Churn" in df.columns
if has_labels:
    y_true_raw = df["Churn"]
    X_input = df.drop(columns=["Churn"])
else:
    X_input = df.copy()


if "TotalCharges" in X_input.columns:
    X_input["TotalCharges"] = pd.to_numeric(X_input["TotalCharges"], errors="coerce")
if "tenure" in X_input.columns and "TotalCharges" in X_input.columns:
    X_input["AvgMonthlySpend"] = X_input["TotalCharges"] / X_input["tenure"].replace(0, 1)
    X_input["IsNewCustomer"] = (X_input["tenure"] <= 6).astype(int)
if "customerID" in X_input.columns:
    X_input = X_input.drop(columns=["customerID"])

try:
    X_proc = preprocessor.transform(X_input)
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()
except Exception as e:
    st.error(f"Could not preprocess the uploaded data: {e}")
    st.stop()

model = models[selected_model_name]
preds = model.predict(X_proc)
pred_labels = label_encoder.inverse_transform(preds)

probs = model.predict_proba(X_proc)[:, 1] if hasattr(model, "predict_proba") else None

st.subheader(f"🔮 Predictions — {selected_model_name}")
result_df = X_input.copy()
result_df["Predicted_Churn"] = pred_labels
if probs is not None:
    result_df["Churn_Probability"] = np.round(probs, 3)
st.dataframe(result_df.head(20))

# ---------------------------------------------------------------------
# c. Evaluation metrics + d. confusion matrix / classification report
# ---------------------------------------------------------------------
if has_labels:
    y_true = label_encoder.transform(y_true_raw)

    st.subheader("📈 Evaluation Metrics")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    acc = accuracy_score(y_true, preds)
    auc = roc_auc_score(y_true, probs) if probs is not None else float("nan")
    prec = precision_score(y_true, preds)
    rec = recall_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    mcc = matthews_corrcoef(y_true, preds)

    col1.metric("Accuracy", f"{acc:.3f}")
    col2.metric("AUC", f"{auc:.3f}")
    col3.metric("Precision", f"{prec:.3f}")
    col4.metric("Recall", f"{rec:.3f}")
    col5.metric("F1 Score", f"{f1:.3f}")
    col6.metric("MCC", f"{mcc:.3f}")

    left, right = st.columns(2)

    with left:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, preds)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with right:
        st.subheader("Classification Report")
        report = classification_report(
            y_true, preds, target_names=label_encoder.classes_, output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose().round(3))
else:
    st.warning(
        "No 'Churn' column found in the uploaded file — showing predictions only. "
        "Include the true label column to see metrics, confusion matrix, and report."
    )

st.markdown("---")
st.subheader("🏆 All Models — Saved Comparison (from training run)")
try:
    metrics_df = pd.read_csv(MODEL_DIR / "metrics.csv")
    st.dataframe(metrics_df)
except FileNotFoundError:
    st.caption("metrics.csv not found — run model/train_models.py first.")
