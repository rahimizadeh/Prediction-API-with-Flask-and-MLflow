# Prediction-API-with-Flask-and-MLflow
An end-to-end machine learning project demonstrating model lifecycle management with MLflow and production deployment using Flask.
[![MLflow](https://img.shields.io/badge/MLflow-%23FF6F00.svg?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Flask](https://img.shields.io/badge/Flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)

An end-to-end machine learning project demonstrating model lifecycle management with MLflow and production deployment using Flask.

## Key Features
- **MLflow Integration**: Full experiment tracking and model registry
- **Flask REST API**: Production-ready model serving
- **Model Versioning**: Track different model versions in registry
- **Visualization**: Built-in result plotting and tree visualization
- **CI/CD Ready**: Easily deployable architecture

## Project Architecture
graph TD  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A[Training Script] -->|Logs to| B[MLflow Tracking Server]  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;B -->|Stores| C[Model Registry]  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;C -->|Serves| D[Flask API]  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;D -->|Responds to| E[Client Applications]  

🚀 Getting Started  
**Prerequisites**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **Python 3.8+**
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - **MLflow server (local/remote)**  
**Required packages:**
pip install -r requirements.txt
Project Structure

├── MLflow_Components/
│   ├── RandomForestRegressor.py   # Training with MLflow tracking
│   └── predict.py                # Model loading from registry
├── Flask_Components/
│   ├── Application_Server.py      # REST API endpoint
│   └── client.py                  # API test client
└── Position_Salaries.csv          # Sample dataset  
🔧 MLflow Implementation
Model Training & Tracking
bash
Copy
python MLflow_Components/RandomForestRegressor.py
Tracks parameters (n_estimators, random_state)

Logs metrics (OOB score, MSE, R²)

Stores artifacts (plots, decision trees)

Registers model in MLflow Model Registry

MLflow UI

Key MLflow Features
Experiment comparison

Model versioning

Artifact storage

Model staging (Staging/Production/Archived)

🌐 Flask API Deployment
Start REST API Server
bash
Copy
python Flask_Components/Application_Server.py
API Endpoint
http
Copy
POST http://localhost:5001/predict
Content-Type: application/json

{
    "level": 6.5
}
Example Response
json
Copy
{
    "prediction": 175000.0
}
Test Client
bash
Copy
python Flask_Components/client.py
🛠️ MLflow + Flask Integration
The Flask API directly loads models from MLflow Model Registry:

python

# Load production-stage model from registry
model = mlflow.sklearn.load_model("models:/SalaryPredictor/Production")
📈 Model Lifecycle Management
Develop model in experiments

Register best model in registry

Promote to Production stage

Serve through Flask API

Monitor and iterate

📚 Resources
MLflow Documentation

Flask Documentation

Sklearn RandomForestRegressor
