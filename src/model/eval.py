import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from mlflow.models import infer_signature

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill any NaN values
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise


def load_model(model_path: str):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', model_path)
        return model
    except Exception as e:
        logger.error('Error loading model from %s: %s', model_path, e)
        raise




def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading parameters from %s: %s', params_path, e)
        raise


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the model and log classification metrics and confusion matrix."""
    try:
        # Predict and calculate metrics
        y_pred = model.predict(X_test)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.debug('Model evaluation completed')

        return rmse, mae, r2
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise



def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        # Create a dictionary with the info you want to save
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }
        # Save the dictionary as a JSON file
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    mlflow.set_experiment('dvc-pipeline-runs')
    
    with mlflow.start_run() as run:
        try:
            print("Eval started")
            # Load parameters from YAML file
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
            params = load_params(os.path.join(root_dir, 'params.yaml'))

            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)
            
            # Load model and vectorizer
            model = load_model(os.path.join(root_dir, 'xgboost_model.pkl'))

            # Load test data for signature inference
            test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))

            # Prepare test data
            X_test = test_data.drop("Price", axis=1)
            y_test = test_data["Price"]

            # Create a DataFrame for signature inference (using first few rows as an example)
            input_example = pd.DataFrame(X_test[:5], columns=X_test.columns)
            # Infer the signature
            signature = infer_signature(input_example, model.predict(X_test[:5]))  # <--- Added for signature
            # Log model with signature
            mlflow.sklearn.log_model(
                model,
                "xgboost_model",
                signature=signature,  # <--- Added for signature
                input_example=input_example  # <--- Added input example
            )

            # Save model info
            # artifact_uri = mlflow.get_artifact_uri()
            model_path = "xgboost_model"
            save_model_info(run.info.run_id, model_path, 'experiment_info.json')


            # Evaluate model and get metrics
            rmse, mae, r2  = evaluate_model(model, X_test, y_test)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("R2", r2)

            # Add important tags
            mlflow.set_tag("model_type", "XGBoost")
            mlflow.set_tag("task", "Cost Prediction")
            mlflow.set_tag("dataset", "Car_Details")

        except Exception as e:
            logger.error(f"Failed to complete model evaluation: {e}")
            print(f"Error: {e}")

if __name__ == '__main__':
    main()