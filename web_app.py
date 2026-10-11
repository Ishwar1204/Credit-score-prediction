"""
Flask Web Application for Credit Score Prediction System.
Built in alignment with Ishwar Kumavat's ML Report (BE Sem VII, Computer Engineering).
"""
import os
import sys
import json
import io
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from src.predict import load_model, predict_single, DEFAULT_MODEL_PATH
from src.preprocessing import clean_dataframe, COMMON_LOAN_TYPES

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"), static_folder=os.path.join(BASE_DIR, "static"))
CORS(app)

MODEL_ARTIFACTS = None


def get_model():
    global MODEL_ARTIFACTS
    if MODEL_ARTIFACTS is None:
        MODEL_ARTIFACTS = load_model(DEFAULT_MODEL_PATH)
    return MODEL_ARTIFACTS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/model-info", methods=["GET"])
def model_info():
    """Return model benchmarks, training metrics, report metadata, and feature importances."""
    metrics_path = os.path.join(BASE_DIR, "models", "training_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)
    else:
        metrics_data = {}

    model_pkg = get_model()
    return jsonify({
        "model_name": model_pkg.get("model_name", "LightGBM"),
        "metrics": metrics_data,
        "loan_types": COMMON_LOAN_TYPES,
        "report_info": {
            "title": "Credit Score Prediction Using Machine Learning Algorithms",
            "student_name": "Ishwar Kumavat",
            "roll_no": "91",
            "moodle_id": "23102034",
            "academic_year": "2026-27",
            "year_semester": "BE - Semester VII",
            "department": "Department of Computer Engineering",
            "instructor": "Prof. Merlin Jacob",
            "experiment_no": "01",
            "target_classes": ["Good", "Average", "Bad"],
            "report_models": ["Logistic_Regression", "Decision_Tree", "Random_Forest", "XGBoost", "LightGBM"]
        }
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Single customer assessment."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No data provided"}), 400

        model_pkg = get_model()
        result = predict_single(data, model_artifacts=model_pkg)
        return jsonify({
            "status": "success",
            "data": result
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict-batch", methods=["POST"])
def api_predict_batch():
    """Batch prediction from CSV upload or using default test.csv."""
    try:
        model_pkg = get_model()
        model = model_pkg["model"]
        imputer = model_pkg.get("imputer", None)
        scaler = model_pkg.get("scaler", None)
        encoders = model_pkg["encoders"]
        feature_names = model_pkg["feature_names"]
        target_inv_map = model_pkg.get("target_inv_map", {0: 'Bad', 1: 'Average', 2: 'Good'})

        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            df_raw = pd.read_csv(file)
        else:
            test_path = os.path.join(BASE_DIR, "test.csv")
            if not os.path.exists(test_path):
                return jsonify({"error": "No file uploaded and test.csv not found"}), 400
            df_raw = pd.read_csv(test_path, nrows=5000)

        id_series = df_raw["ID"] if "ID" in df_raw.columns else pd.Series(range(len(df_raw)))
        df_clean = clean_dataframe(df_raw, is_train=False)

        for col, le in encoders.items():
            if col in df_clean.columns:
                known = set(le.classes_)
                s = df_clean[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
                df_clean[col] = le.transform(s)

        for col in feature_names:
            if col not in df_clean.columns:
                df_clean[col] = np.nan

        X = df_clean[feature_names].copy()
        if imputer is not None:
            X = imputer.transform(X)
        if scaler is not None:
            X = scaler.transform(X)

        preds = np.asarray(model.predict(X)).ravel().astype(int)
        probs = model.predict_proba(X) if hasattr(model, "predict_proba") else None

        pred_labels = [target_inv_map[int(p)] for p in preds]

        out_df = pd.DataFrame({
            "ID": id_series,
            "Predicted_Credit_Score": pred_labels
        })

        if probs is not None:
            for idx, lbl in target_inv_map.items():
                out_df[f"Prob_{lbl}"] = (probs[:, idx] * 100).round(2)

        summary_counts = out_df["Predicted_Credit_Score"].value_counts().to_dict()
        preview_records = out_df.head(50).to_dict(orient="records")

        csv_buffer = io.StringIO()
        out_df.to_csv(csv_buffer, index=False)
        csv_str = csv_buffer.getvalue()

        return jsonify({
            "status": "success",
            "total_records": len(out_df),
            "summary": summary_counts,
            "preview": preview_records,
            "csv_data": csv_str
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/download-batch", methods=["POST"])
def download_batch():
    """Return CSV file attachment from posted csv_data."""
    csv_data = request.form.get("csv_data", "")
    if not csv_data:
        return "No data to download", 400

    bio = io.BytesIO(csv_data.encode('utf-8'))
    return send_file(
        bio,
        mimetype="text/csv",
        as_attachment=True,
        download_name="credit_score_predictions.csv"
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f"Credit Score AI Web App running at: http://localhost:{port}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
