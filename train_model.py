"""
train_model.py
----------------
This script trains a Credit Card Fraud Detection model using the CSV
dataset stored in dataset/creditcard.csv.

Steps performed:
1. Load the dataset with Pandas.
2. Check for missing values.
3. Separate features (X) and target (y).
4. Split into train/test sets.
5. Handle class imbalance (fraud is rare) using class_weight='balanced'.
6. Train a Random Forest classifier.
7. Evaluate using Accuracy, Precision, Recall, F1-score, Confusion Matrix.
8. Save the trained model (+ encoders + metadata) using joblib.

Run this with:
    python train_model.py
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import joblib
import os

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------
DATASET_PATH = os.path.join("dataset", "creditcard.csv")
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading dataset from:", DATASET_PATH)
df = pd.read_csv(DATASET_PATH)
print("Dataset shape:", df.shape)
print("\nColumns found:", list(df.columns))

# ---------------------------------------------------------
# 2. Check missing values
# ---------------------------------------------------------
print("\nMissing values per column:")
print(df.isnull().sum())

# Drop rows with missing values if any exist (dataset is clean, but this
# makes the script robust to other CSVs too).
df = df.dropna()

# ---------------------------------------------------------
# Column configuration
# ---------------------------------------------------------
# transaction_id is just a row identifier -> not useful for prediction
ID_COLUMN = "transaction_id"
TARGET_COLUMN = "is_fraud"
CATEGORICAL_COLUMNS = ["merchant_category"]
NUMERIC_COLUMNS = [
    "amount",
    "transaction_hour",
    "foreign_transaction",
    "location_mismatch",
    "device_trust_score",
    "velocity_last_24h",
    "cardholder_age",
]

if ID_COLUMN in df.columns:
    df = df.drop(columns=[ID_COLUMN])

# ---------------------------------------------------------
# 3. Encode categorical column(s)
# ---------------------------------------------------------
label_encoders = {}
for col in CATEGORICAL_COLUMNS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"\nEncoded '{col}' classes:", list(le.classes_))

FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

# ---------------------------------------------------------
# 4. Separate features (X) and target (y)
# ---------------------------------------------------------
X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

print("\nClass distribution (0 = Normal, 1 = Fraud):")
print(y.value_counts())
print(y.value_counts(normalize=True) * 100)

# ---------------------------------------------------------
# 5. Train/test split (stratified because fraud is rare)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# ---------------------------------------------------------
# 6. Handle class imbalance + Train model
# ---------------------------------------------------------
# Fraud is rare (~1.5%), so we use class_weight='balanced' which tells the
# Random Forest to pay much more attention to the minority (fraud) class
# instead of just predicting "Normal" every time.
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 7. Evaluate the model
# ---------------------------------------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("\n================ MODEL EVALUATION ================")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print("\nConfusion Matrix (rows=actual, cols=predicted):")
print(cm)
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

# ---------------------------------------------------------
# Dataset-level statistics for the dashboard
# ---------------------------------------------------------
total_transactions = int(len(df))
normal_count = int((df[TARGET_COLUMN] == 0).sum())
fraud_count = int((df[TARGET_COLUMN] == 1).sum())
fraud_percentage = round((fraud_count / total_transactions) * 100, 2)

stats = {
    "total_transactions": total_transactions,
    "normal_count": normal_count,
    "fraud_count": fraud_count,
    "fraud_percentage": fraud_percentage,
    "accuracy": round(accuracy * 100, 2),
    "precision": round(precision * 100, 2),
    "recall": round(recall * 100, 2),
    "f1_score": round(f1 * 100, 2),
    "confusion_matrix": cm.tolist(),
    "feature_columns": FEATURE_COLUMNS,
    "numeric_columns": NUMERIC_COLUMNS,
    "categorical_columns": CATEGORICAL_COLUMNS,
    "merchant_category_options": list(label_encoders["merchant_category"].classes_),
    # ranges are used by the frontend form for min/max hints
    "feature_ranges": {
        col: {"min": float(df[col].min()), "max": float(df[col].max())}
        for col in NUMERIC_COLUMNS
    },
}

with open(os.path.join(MODEL_DIR, "stats.json"), "w") as f:
    json.dump(stats, f, indent=2)

# ---------------------------------------------------------
# 8. Save trained model + encoders using joblib
# ---------------------------------------------------------
joblib.dump(model, os.path.join(MODEL_DIR, "fraud_model.pkl"))
joblib.dump(label_encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))
joblib.dump(FEATURE_COLUMNS, os.path.join(MODEL_DIR, "feature_columns.pkl"))

print("\nSaved model files to the 'model/' folder:")
print(" - fraud_model.pkl       (trained Random Forest model)")
print(" - label_encoders.pkl    (encoders for categorical columns)")
print(" - feature_columns.pkl   (exact column order the model expects)")
print(" - stats.json            (dashboard statistics)")
print("\nTraining complete! You can now run: python app.py")
