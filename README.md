# Prediction API with Flask and MLflow

A small end-to-end ML project that trains a Random Forest salary regressor, tracks/registers it with MLflow, and serves predictions through Flask.

## Dataset

The committed dataset is `Salary_predict.csv` with these features:

- `experience`
- `age`
- `interview_score`
- target: `Salary`

## Local setup

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run MLflow

```bash
mlflow server --host 127.0.0.1 --port 8080
```

In another terminal, train and register the model:

```bash
python "Random_Forest_Regressor MLflow.py"
```

The registered model name is `SalaryRandomForest`. Before serving, assign the alias `champion` to the model version you want to deploy, or override `MODEL_URI` with another valid MLflow model URI.

## Run the API

```bash
python Application_Server.py
```

Health check:

```bash
curl http://127.0.0.1:5001/health
```

Prediction example:

```bash
curl -X POST http://127.0.0.1:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"experience":4,"age":30,"interview_score":8}'
```

Or run:

```bash
python client.py
```

## Tests

```bash
pytest -q
```

The API tests inject a dummy model, so they do not require a running MLflow server.

## Docker

```bash
docker compose up --build
```

The Flask container uses `MLFLOW_TRACKING_URI=http://mlflow-server:8080` and `MODEL_URI=models:/SalaryRandomForest@champion`.

## Environment variables

- `MLFLOW_TRACKING_URI`
- `MLFLOW_EXPERIMENT_NAME`
- `MLFLOW_MODEL_NAME`
- `MODEL_URI`
- `DATA_PATH`
- `FLASK_HOST`
- `FLASK_PORT`

## CI

GitHub Actions runs `pytest` on pushes and pull requests using `.github/workflows/ci.yml`.
