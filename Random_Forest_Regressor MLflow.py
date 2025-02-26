"""
Random Forest Regressor Implementation with MLflow Tracking

This script:
1. Loads and preprocesses position/salary data
2. Trains a Random Forest Regressor model
3. Logs parameters, metrics, and artifacts with MLflow
4. Generates visualization plots
5. Registers the trained model in MLflow Model Registry

Key Features:
- Out-of-Bag (OOB) error calculation
- MSE and R² metrics tracking
- Model versioning and registry
- Decision tree visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from sklearn.tree import plot_tree

# Set the tracking URI and experiment

mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("Set an name for expriment") # EXample: mlflow.set_experiment("RandomForestRegressorModel")

# Start an MLflow run

with mlflow.start_run():
    try:
        # Load data from csv file (Position_Salaries.csv)
        # Example: df = pd.read_csv('G:/00 MLflow/Random Forest Resressor/Position_Salaries.csv')

        df = pd.read_csv('./Position_Salaries.csv') # Load data from the current directory
        print("Original Data:")
        print(df)

        # Correct feature selection
        X = df[['Level']].values  # Double brackets to keep 2D shape
        y = df['Salary'].values

        # Log parameters
        n_estimators = 10
        random_state = 0
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)

        # Train model
        regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, oob_score=True)
        regressor.fit(X, y)  # Use correct features

        # Calculate metrics
        oob_score = regressor.oob_score_
        predictions = regressor.predict(X)
        mse = mean_squared_error(y, regressor.oob_prediction_)
        r2 = r2_score(y, predictions)

        # Log metrics

        '''The OOB score is a valuable metric for assessing the performance of Random Forest models.
           It provides an estimate of how well the model will generalize to unseen data without requiring a separate validation set. 
           A high OOB score (close to 1) indicates that the Random Forest model is performing well and generalizing effectively.
        '''
        mlflow.log_metric("oob_score", oob_score)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        print(f'\n Out-of-Bag Score: {oob_score}')
        print(f'Mean Squared Error: {mse}')
        print(f'R-squared: {r2}')

        # Visualization

        X_grid = np.arange(min(X), max(X), 0.01).reshape(-1, 1)
        plt.scatter(X, y, color='blue')
        plt.plot(X_grid, regressor.predict(X_grid), color='green')
        plt.title("Random Forest Regression Results")
        plt.xlabel('Position level')
        plt.ylabel('Salary')
        plt.savefig("random_forest_regression_results.png")
        mlflow.log_artifact("random_forest_regression_results.png")
        plt.show()

        # Plot the decision tree
        
        tree_to_plot = regressor.estimators_[0]
        plt.figure(figsize=(20, 10))
        plot_tree(tree_to_plot, feature_names=df.columns.tolist(), filled=True, rounded=True, fontsize=10)
        plt.title("Decision Tree from Random Forest")
        plt.savefig("decision_tree.png")
        mlflow.log_artifact("decision_tree.png")
        plt.show()

        # Log the model

    ''' mlflow.sklearn.log_model():
        This logs the trained model to the MLflow tracking server. 
        The model is saved in the specified path (random_forest_model). '''
        
        model_path = "random_forest_model"
        mlflow.sklearn.log_model(regressor, model_path)

        # Register the model in the MLflow Model Registry

    ''' mlflow.register_model(): This registers the logged model in the MLflow Model Registry. 
        Example: mlflow.register_model(model_uri, "RandomForestRegressorModel")
        The model_uri is constructed using the run_id (which is automatically generated for each run) and the model path. '''
    
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_path}"
        mlflow.register_model(model_uri, "Your_model")
        print("-"* 40 + "RUN INFO" + "-"* 40)
        print("\n", mlflow.active_run()._info)
        print("-"* 80)
    except Exception as e:
        print(f"An error occurred: {e}")