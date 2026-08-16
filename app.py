"""
app.py — Streamlit demo app for the Breast Cancer classification project.

Features:
  a. CSV upload of test data
  b. Model selection dropdown (5 trained classifiers)
  c. Display of evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
  d. Confusion matrix + classification report
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Breast Cancer Classifier Comparison", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "model/random_forest.pkl",
}


@st.cache_resource
def load_artifacts():
    scaler = joblib.load("model/scaler.pkl")
    with open("model/feature_names.json") as f:
        meta = json.load(f)
    models = {name: joblib.load(path) for name, path in MODEL_FILES.items()}
    return scaler, meta, models


scaler, meta, models = load_artifacts()
feature_names = meta["feature_names"]
target_names = meta["target_names"]

st.title("🔬 Breast Cancer Classification — Model Comparison")
st.caption(
    "Dataset: Breast Cancer Wisconsin (Diagnostic) — 30 features, 569 instances, "
    "binary classification (malignant vs benign)."
)

# -----------------------------------------------------------------
# a. Dataset upload
# -----------------------------------------------------------------
st.sidebar.header("1. Upload Test Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload test_data.csv (must include a 'target' column)", type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("No file uploaded — using the bundled test_data.csv sample.")
    df = pd.read_csv("test_data.csv")

missing_cols = [c for c in feature_names if c not in df.columns]
if missing_cols:
    st.error(f"Uploaded file is missing required feature columns: {missing_cols}")
    st.stop()

has_target = "target" in df.columns

st.subheader("Preview of Uploaded Data")
st.dataframe(df.head(10), use_container_width=True)

X = df[feature_names]
X_scaled = scaler.transform(X)

# -----------------------------------------------------------------
# b. Model selection dropdown
# -----------------------------------------------------------------
st.sidebar.header("2. Choose a Model")
model_choice = st.sidebar.selectbox("Model", list(models.keys()))
model = models[model_choice]

y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)[:, 1]

st.subheader(f"Predictions — {model_choice}")
result_df = df.copy()
result_df["predicted_label"] = [target_names[p] for p in y_pred]
result_df["predicted_probability_benign"] = np.round(y_proba, 4)
st.dataframe(result_df[["predicted_label", "predicted_probability_benign"]].head(20),
             use_container_width=True)

# -----------------------------------------------------------------
# c. Evaluation metrics (only possible if ground-truth labels present)
# -----------------------------------------------------------------
st.subheader("Evaluation Metrics")

if has_target:
    y_true = df["target"]
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Accuracy", f"{acc:.4f}")
    m2.metric("AUC", f"{auc:.4f}")
    m3.metric("Precision", f"{prec:.4f}")
    m4.metric("Recall", f"{rec:.4f}")
    m5.metric("F1 Score", f"{f1:.4f}")
    m6.metric("MCC", f"{mcc:.4f}")

    # -----------------------------------------------------------------
    # d. Confusion matrix + classification report
    # -----------------------------------------------------------------
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=target_names, yticklabels=target_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with col_b:
        st.markdown("**Classification Report**")
        report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
else:
    st.warning("Uploaded CSV has no 'target' column — showing predictions only, "
               "no evaluation metrics can be computed.")

# -----------------------------------------------------------------
# All-model comparison table (precomputed on the standard test split)
# -----------------------------------------------------------------
st.subheader("All Models — Comparison on Held-Out Test Split")
comparison_df = pd.read_csv("model/metrics_comparison.csv")
st.dataframe(comparison_df, use_container_width=True)
st.bar_chart(comparison_df.set_index("ML Model Name")[["Accuracy", "AUC", "F1"]])
