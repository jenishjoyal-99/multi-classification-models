"""
train_models.py
----------------
Trains 5 classification models on the Breast Cancer Wisconsin (Diagnostic)
dataset, evaluates each with 6 metrics, and saves:
  - trained model files (model/*.pkl)
  - a fitted StandardScaler (model/scaler.pkl)
  - the held-out test split as test_data.csv (used by the Streamlit app)
  - a metrics comparison table (model/metrics_comparison.csv)

Dataset: sklearn.datasets.load_breast_cancer
  - 569 instances (>= 500 required)
  - 30 numeric features (>= 12 required)
  - Binary classification: malignant (0) vs benign (1)
Source: originally donated to UCI ML Repository by Dr. William H. Wolberg,
University of Wisconsin Hospitals.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target  # 0 = malignant, 1 = benign
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Save the held-out test split (features + true label) as test_data.csv
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv("../test_data.csv", index=False)
print(f"Saved test_data.csv with shape {test_df.shape}")

# Scale features (helps LogisticRegression & KNN; harmless for tree/NB/RF)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")

# ---------------------------------------------------------------
# 2. Define models
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
}

results = []
saved_names = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl",
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append({
        "ML Model Name": name,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4),
    })

    joblib.dump(model, saved_names[name])
    print(f"\n=== {name} ===")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=data.target_names))

# ---------------------------------------------------------------
# 3. Save comparison table
# ---------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("metrics_comparison.csv", index=False)
print("\nComparison table:\n", results_df.to_string(index=False))

# Save feature names + target names for the Streamlit app
with open("feature_names.json", "w") as f:
    json.dump({
        "feature_names": feature_names,
        "target_names": list(data.target_names),
    }, f, indent=2)

print("\nAll models trained and saved successfully.")
