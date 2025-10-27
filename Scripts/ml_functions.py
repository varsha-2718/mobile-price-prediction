import os
import pickle
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dotenv import load_dotenv
from helper_functions import log_info, log_error
from math import sqrt

# Load environment variables
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, os.getenv('ARTIFACTS_DIR'))
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.pkl")

def training_pipeline(X_train, y_train):
    """
    Trains an XGBoost Regressor and saves the model.
    """
    try:
        model = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective='reg:squarederror',
            random_state=42
        )
        model.fit(X_train, y_train)

        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)

        log_info(f"Regression model trained and saved at {MODEL_PATH}")
        return model
    except Exception as e:
        log_error(f"Error during model training: {e}")
        raise

def load_model():
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        log_info("Model loaded successfully.")
        return model
    except FileNotFoundError:
        log_error(f"Model file not found at {MODEL_PATH}")
        raise

def prediction_pipeline(X_val):
    try:
        model = load_model()
        return model.predict(X_val)
    except Exception as e:
        log_error(f"Error during prediction: {e}")
        raise

def evaluation_matrices(predictions, y_val):
    try:
        rmse = sqrt(mean_squared_error(y_val, predictions))
        mae = mean_absolute_error(y_val, predictions)
        r2 = r2_score(y_val, predictions)

        log_info("Regression model evaluation completed.")
        log_info(f"RMSE: {rmse}")
        log_info(f"MAE: {mae}")
        log_info(f"R² Score: {r2}")

        return rmse, mae, r2
    except Exception as e:
        log_error(f"Error during evaluation: {e}")
        raise
