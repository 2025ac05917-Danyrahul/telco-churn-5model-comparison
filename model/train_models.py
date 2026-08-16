"""
train_models.py
----------------
Trains 5 classification models on the Telco Customer Churn dataset and
saves:
  - trained model objects (model/*.joblib)
  - the fitted preprocessing pipeline (model/preprocessor.joblib)
  - the label encoder (model/label_encoder.joblib)
  - a metrics comparison table (model/metrics.csv)
  - a held-out test_data.csv (used later by the Streamlit app)

Run once locally / on BITS Virtual Lab:
    python model/train_models.py
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

DATA_PATH = Path(__file__).resolve().parent.parent / "telco.csv"
MODEL_DIR = Path(__file__).resolve().parent
OUT_DIR = Path(__file__).resolve().parent.parent

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# TotalCharges has some blank strings for brand-new customers -> coerce to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Drop the ID column (not a feature)
df = df.drop(columns=["customerID"])

# A couple of light, defensible engineered features (keeps this from being a
# pure copy of the raw dataset and gives the models a bit more signal)
df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
df["IsNewCustomer"] = (df["tenure"] <= 6).astype(int)

target_col = "Churn"
y_raw = df[target_col]
X = df.drop(columns=[target_col])

le = LabelEncoder()
y = le.fit_transform(y_raw)  # No -> 0, Yes -> 1

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
print(f"Total features: {len(numeric_features) + len(categorical_features)} | Rows: {len(X)}")

# ---------------------------------------------------------------------
# 2. Train / test split (test split is what we export as test_data.csv)
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Save the raw (unprocessed) test split as test_data.csv for the assignment
# submission + for uploading into the Streamlit app.
test_export = X_test.copy()
test_export[target_col] = le.inverse_transform(y_test)
test_export.to_csv(OUT_DIR / "test_data.csv", index=False)
print(f"Saved test_data.csv with {len(test_export)} rows")

# ---------------------------------------------------------------------
# 3. Preprocessing pipeline
# ---------------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# Naive Bayes needs dense input
if hasattr(X_train_proc, "toarray"):
    X_train_dense = X_train_proc.toarray()
    X_test_dense = X_test_proc.toarray()
else:
    X_train_dense = X_train_proc
    X_test_dense = X_test_proc

# ---------------------------------------------------------------------
# 4. Define models
# ---------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=15),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=RANDOM_STATE),
}

results = []
fitted_models = {}

for name, model in models.items():
    model.fit(X_train_dense, y_train)
    preds = model.predict(X_test_dense)
    probs = model.predict_proba(X_test_dense)[:, 1] if hasattr(model, "predict_proba") else preds

    metrics = {
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "AUC": round(roc_auc_score(y_test, probs), 4),
        "Precision": round(precision_score(y_test, preds), 4),
        "Recall": round(recall_score(y_test, preds), 4),
        "F1": round(f1_score(y_test, preds), 4),
        "MCC": round(matthews_corrcoef(y_test, preds), 4),
    }
    results.append(metrics)
    fitted_models[name] = model
    print(metrics)

metrics_df = pd.DataFrame(results)
metrics_df.to_csv(MODEL_DIR / "metrics.csv", index=False)
print("\nSaved metrics.csv")
print(metrics_df.to_string(index=False))

# ---------------------------------------------------------------------
# 5. Persist everything the Streamlit app needs
# ---------------------------------------------------------------------
joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")
joblib.dump(le, MODEL_DIR / "label_encoder.joblib")
joblib.dump(list(numeric_features), MODEL_DIR / "numeric_features.joblib")
joblib.dump(list(categorical_features), MODEL_DIR / "categorical_features.joblib")

for name, model in fitted_models.items():
    fname = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, MODEL_DIR / fname)

print("\nAll models and preprocessing artifacts saved to /model")
