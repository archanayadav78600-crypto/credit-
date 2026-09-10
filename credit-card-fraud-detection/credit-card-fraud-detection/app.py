"""
app.py
-------
Flask backend for the Credit Card Fraud Detection web app.

Routes:
    /            -> Home page
    /detect      -> Fraud detection form (GET) + handle prediction (POST)
    /result      -> Shows the last prediction result
    /dashboard   -> Dataset & model statistics dashboard
    /api/predict -> JSON API used by script.js (optional AJAX flow)

Run with:
    python app.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify

app = Flask(__name__)

MODEL_DIR = "model"

# ---------------------------------------------------------
# Load trained model + supporting files (once, at startup)
# ---------------------------------------------------------
model = joblib.load(os.path.join(MODEL_DIR, "fraud_model.pkl"))
label_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))

with open(os.path.join(MODEL_DIR, "stats.json")) as f:
    stats = json.load(f)

NUMERIC_COLUMNS = stats["numeric_columns"]
CATEGORICAL_COLUMNS = stats["categorical_columns"]
MERCHANT_OPTIONS = stats["merchant_category_options"]
FEATURE_RANGES = stats["feature_ranges"]

# Keep the most recent prediction in memory so the /result page can show it
# after a redirect (simple approach, no database needed for this project).
last_result = {}


def build_feature_vector(form_data):
    """Turn the submitted form fields into a single-row DataFrame the
    model can predict on (a DataFrame keeps the exact column names/order
    the model was trained with, avoiding sklearn's "missing feature
    names" warning)."""
    row = {}
    for col in feature_columns:
        if col in CATEGORICAL_COLUMNS:
            raw_value = form_data.get(col)
            encoder = label_encoders[col]
            row[col] = encoder.transform([raw_value])[0]
        else:
            row[col] = float(form_data.get(col))
    return pd.DataFrame([row], columns=feature_columns)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["GET", "POST"])
def detect():
    if request.method == "POST":
        try:
            features = build_feature_vector(request.form)
            prediction = int(model.predict(features)[0])
            probability = float(model.predict_proba(features)[0][1])  # P(fraud)

            last_result["prediction"] = "Fraud" if prediction == 1 else "Normal"
            last_result["is_fraud"] = prediction == 1
            last_result["risk_percentage"] = round(probability * 100, 2)
            last_result["submitted_values"] = {
                col: request.form.get(col) for col in feature_columns
            }

            return redirect(url_for("result"))
        except Exception as exc:
            error_message = f"Could not process the form: {exc}"
            return render_template(
                "detect.html",
                numeric_columns=NUMERIC_COLUMNS,
                categorical_columns=CATEGORICAL_COLUMNS,
                merchant_options=MERCHANT_OPTIONS,
                feature_ranges=FEATURE_RANGES,
                error=error_message,
            )

    return render_template(
        "detect.html",
        numeric_columns=NUMERIC_COLUMNS,
        categorical_columns=CATEGORICAL_COLUMNS,
        merchant_options=MERCHANT_OPTIONS,
        feature_ranges=FEATURE_RANGES,
        error=None,
    )


@app.route("/result")
def result():
    if not last_result:
        return redirect(url_for("detect"))
    return render_template("result.html", result=last_result)


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", stats=stats)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Optional JSON endpoint - handy if you extend the frontend with AJAX."""
    data = request.get_json(force=True)
    try:
        features = build_feature_vector(data)
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])
        return jsonify(
            {
                "prediction": "Fraud" if prediction == 1 else "Normal",
                "risk_percentage": round(probability * 100, 2),
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
