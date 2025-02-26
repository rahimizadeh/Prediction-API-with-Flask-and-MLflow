# This code loads a ML model by its version number or stage and predicts an input value.
# To test the prediction function run on cmd Directory of the code. Example: python predict.py --level 6.5 

import mlflow
import pandas as pd
import numpy as np
import argparse

# Setting training URI
mlflow.set_tracking_uri("http://127.0.0.1:8080")  


# Load the model from the Model Registry
# The model can be loaded by using actual Version number or the stage of a model. 

def predict_salary(level: float):

    # model = mlflow.sklearn.load_model("models:/RandomForestRegressorModel/26")  # Using actual version number of the model
    # model = mlflow.sklearn.load_model("models:/RandomForestRegressorModel/Production") # Using the stage of the model

    model = mlflow.sklearn.load_model("models:/Your_Model/Model_stage") # Using the stage of the model 

    # Prepare input 
    input_data = np.array([[level]])  
    
    # Make prediction
    prediction = model.predict(input_data)
    return prediction[0]  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=float, required=True)
    args = parser.parse_args()
    
    result = predict_salary(args.level)
    print(f"Predicted Salary for Level {args.level}: ${result:,.2f}")