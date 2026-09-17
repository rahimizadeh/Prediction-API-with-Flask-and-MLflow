"""Flask API for salary prediction using an MLflow-registered model."""

import os
from functools import lru_cache

import mlflow
import pandas as pd
from flask import Flask, jsonify, request

FEATURES = ["experience", "age", "interview_score"]
DEFAULT_TRACKING_URI = "http://127.0.0.1:8080"
DEFAULT_MODEL_URI = "models:/SalaryRandomForest@champion"


@lru_cache(maxsize=1)
def _load_model():
    """Load the configured model once, on the first prediction request."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    model_uri = os.getenv("MODEL_URI", DEFAULT_MODEL_URI)
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.sklearn.load_model(model_uri)


def create_app(model=None):
    """Create the Flask app.

    ``model`` is injectable so unit tests can run without a live MLflow server.
    """
    app = Flask(__name__)
    app.config["MODEL"] = model

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"}), 200

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be valid JSON."}), 400

        missing = [name for name in FEATURES if name not in payload]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

        try:
            values = {name: float(payload[name]) for name in FEATURES}
        except (TypeError, ValueError):
            return jsonify({"error": "All feature values must be numeric."}), 400

        model_instance = app.config.get("MODEL") or _load_model()
        input_frame = pd.DataFrame([values], columns=FEATURES)
        prediction = float(model_instance.predict(input_frame)[0])
        return jsonify({"prediction": round(prediction, 2)}), 200

    return app


app = create_app()

if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5001"))
    app.run(host=host, port=port, debug=False)
