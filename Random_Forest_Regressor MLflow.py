"""Train and register a RandomForest salary model with MLflow."""

import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path(os.getenv("DATA_PATH", "Salary_predict.csv"))
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "SalaryPrediction")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "SalaryRandomForest")
FEATURES = ["experience", "age", "interview_score"]
TARGET = "Salary"


def main():
    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "mse": mean_squared_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        mlflow.log_params({"n_estimators": 200, "random_state": 42})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=X_train.head(3),
        )
        print(f"Run ID: {run.info.run_id}")
        print({name: round(value, 4) for name, value in metrics.items()})
        print(f"Registered model: {MODEL_NAME}")
        print("Create or move the alias 'champion' to the desired registered model version before serving.")


if __name__ == "__main__":
    main()
