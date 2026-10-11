"""
Flask REST API for Credit Score Prediction.
Provides endpoints for single and batch predictions.
"""
import os
import sys
from flask import Flask, request, jsonify

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.predict import load_model, predict_single, DEFAULT_MODEL_PATH

app = Flask(__name__)
model_artifacts = None


@app.before_request
def init_model():
    global model_artifacts
    if model_artifacts is None:
        model_artifacts = load_model(DEFAULT_MODEL_PATH)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": model_artifacts is not None})


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Accepts JSON body of customer features.
    Returns credit score prediction, probabilities, and risk factors.
    """
    data = request.get_json(force=True)
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON payload. Expected customer feature dictionary."}), 400

    try:
        result = predict_single(data, model_artifacts=model_artifacts)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    POST /predict_batch
    Accepts JSON body: {"records": [ {...}, {...} ]}
    """
    data = request.get_json(force=True)
    records = data.get("records", [])
    if not isinstance(records, list) or len(records) == 0:
        return jsonify({"error": "Payload must contain a non-empty 'records' list."}), 400

    try:
        results = [predict_single(r, model_artifacts=model_artifacts) for r in records]
        return jsonify({"predictions": results, "count": len(results)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Credit Score Prediction API on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
