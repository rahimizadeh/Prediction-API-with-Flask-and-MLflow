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
  
&nbsp;&nbsp;&nbsp;**Prerequisites**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Python 3.8+  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - MLflow server (local/remote)  
&nbsp;&nbsp;&nbsp;**Required packages:**
```bash
pip install -r requirements.txt
 ```
&nbsp;&nbsp;&nbsp;**Project Structure**  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── MLflow_Components/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│   &nbsp;&nbsp;&nbsp;├── RandomForestRegressor.py   &nbsp;&nbsp;&nbsp;# Training with MLflow tracking  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│   &nbsp;&nbsp;&nbsp;└── predict.py                &nbsp;&nbsp;&nbsp;# Model loading from registry  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Flask_Components/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│   &nbsp;&nbsp;&nbsp;├── Application_Server.py      &nbsp;&nbsp;&nbsp;# REST API endpoint  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│  &nbsp;&nbsp;&nbsp;└── client.py                  &nbsp;&nbsp;&nbsp;# API test client  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── Position_Salaries.csv          &nbsp;&nbsp;&nbsp;# Sample dataset  

🔧 MLflow Implementation  
  
&nbsp;&nbsp;&nbsp;**Model Training & Tracking**
```bash
python MLflow_Components/RandomForestRegressor.py
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Tracks parameters (n_estimators, random_state)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Logs metrics (OOB score, MSE, R²)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Stores artifacts (plots, decision trees)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Registers model in MLflow Model Registry  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- MLflow UI  
  
**Key MLflow Features**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Experiment comparison  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Model versioning  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Artifact storage  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Model staging (Staging/Production/Archived)  

🌐 Flask API Deployment  

&nbsp;&nbsp;&nbsp; **Start REST API Server**
```bash
python Flask_Components/Application_Server.py
```  
&nbsp;&nbsp;&nbsp; **API Endpoint**  
```bash
POST http://localhost:5001/predict
```
&nbsp;&nbsp;&nbsp; **Content-Type: application/json**  
```bash

{
    "level": 6.5
}
```
&nbsp;&nbsp;&nbsp; Example Response  
```bash
{
   "prediction": 175000.0
}
```
**Test Client**
```bash
python Flask_Components/client.py
```
🛠️ MLflow + Flask Integration  
**The Flask API directly loads models from MLflow Model Registry:**
Load production-stage model from registry
``` bash
 model = mlflow.sklearn.load_model("models:/Your_Model/Production")
```
📈 Model Lifecycle Management  
**Develop model in experiments**  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1. Register best model in registry  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2. Promote to Production stage  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3. Serve through Flask API  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 4. Monitor and iterate  

📚 Resources  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[MLflow Documentation
](https://mlflow.org/docs/latest/index.html)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Flask Documentation
](https://flask.palletsprojects.com/)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Sklearn RandomForestRegressor
](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
