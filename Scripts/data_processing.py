import os
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from helper_functions import log_info, log_error

# Load environment variables
load_dotenv()

# Define base paths dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, os.getenv('ARTIFACTS_DIR'))

# Ensure Artifacts directory exists
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Define paths for pipeline and label encoder
PIPELINE_PATH = os.path.join(ARTIFACTS_DIR, "data_pipeline.pkl")
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")

def create_data_pipeline():
    """
    Creates a preprocessing pipeline for the mobile price regression dataset.
    """
    # Match actual feature names in your dataset
    numerical_features = [
        'Rating',
        'Spec_score',
        'Ram',
        'Battery',
        'Display',
        'Camera',
        'External_Memory',
        'Android_version',
        'Inbuilt_memory'
    ]

    # Add binary/categorical features that are already encoded (0/1 or categorical)
    categorical_features = [
        'No_of_sim',
        'fast_charging'
    ]

    all_features = numerical_features + categorical_features

    pipeline = Pipeline(steps=[
        ("scaler", MinMaxScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", pipeline, all_features)
    ])

    log_info("Data processing pipeline created successfully.")
    return preprocessor


def save_pipeline(pipeline):
    """
    Saves the preprocessing pipeline.
    """
    with open(PIPELINE_PATH, 'wb') as file:
        pickle.dump(pipeline, file)
    log_info(f"Pipeline saved at {PIPELINE_PATH}")

def encode_response_variable(y):
    """
    Encodes target variable 'price_range' and saves the label encoder.
    """
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    log_info(f"Label encoder saved at {LABEL_ENCODER_PATH}")
    return y_encoded

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the dataset into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
