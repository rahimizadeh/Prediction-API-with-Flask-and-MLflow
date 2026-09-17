"""Command-line prediction helper for the registered salary model."""

import argparse
import os

import mlflow
import pandas as pd

FEATURES = ["experience", "age", "interview_score"]
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
MODEL_URI = os.getenv("MODEL_URI", "models:/SalaryRandomForest@champion")


def predict_salary(experience: float, age: float, interview_score: float) -> float:
    mlflow.set_tracking_uri(TRACKING_URI)
    model = mlflow.sklearn.load_model(MODEL_URI)
    frame = pd.DataFrame(
        [{
            "experience": experience,
            "age": age,
            "interview_score": interview_score,
        }],
        columns=FEATURES,
    )
    return float(model.predict(frame)[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experience", type=float, required=True)
    parser.add_argument("--age", type=float, required=True)
    parser.add_argument("--interview-score", type=float, required=True)
    args = parser.parse_args()

    result = predict_salary(args.experience, args.age, args.interview_score)
    print(f"Predicted salary: ${result:,.2f}")
