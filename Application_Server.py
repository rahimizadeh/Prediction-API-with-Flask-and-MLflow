# Deploy with a Web Framework (Flask)
# REST API Deployment
# This is Server side. Run this code on a Terminal as python Application_Server.py


from flask import Flask, request, jsonify
import mlflow
import numpy as np

mlflow.set_tracking_uri("http://127.0.0.1:8080")  

app = Flask(__name__)

# load the model from the MLflow model registry at startup.
# Example: model = mlflow.sklearn.load_model("models:/RandomForestRegressorModel/Production")

model = mlflow.sklearn.load_model("models:/Your_Model/Model_stage") 

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    level = float(data['level'])
    input_data = np.array([[level]])
    prediction = model.predict(input_data)[0]
    return jsonify({"prediction": round(prediction, 2)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)