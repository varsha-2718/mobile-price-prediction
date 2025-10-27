import os
import pandas as pd
from data_processing import (
    create_data_pipeline,
    split_data,
    save_pipeline
)
from ml_functions import training_pipeline, prediction_pipeline, evaluation_matrices
from dotenv import load_dotenv
import re
import numpy as np

# Load environment variables
load_dotenv()

# Define base directory and dataset path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, os.getenv('DATA_DIR'))
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')

# Load dataset
df = pd.read_csv("data/train.csv")
# Drop irrelevant columns and isolate features/target
X = df.drop(['Unnamed: 0', 'Name', 'Processor_name', 'Price'], axis=1)

# Clean target variable: remove commas and convert to float
y = df['Price'].replace(',', '', regex=True).astype(float)

# Helper functions for cleaning and feature extraction
def extract_number(text):
    if pd.isnull(text):
        return np.nan
    match = re.search(r'\d+\.?\d*', text)
    return float(match.group()) if match else np.nan

def external_memory_to_gb(text):
    if pd.isnull(text):
        return 0
    tb_match = re.search(r'upto (\d+\.?\d*) TB', text)
    gb_match = re.search(r'upto (\d+\.?\d*) GB', text)
    if tb_match:
        return float(tb_match.group(1)) * 1024
    elif gb_match:
        return float(gb_match.group(1))
    return 0

def count_sims(text):
    if pd.isnull(text):
        return 0
    if 'Dual' in text or 'dual' in text:
        return 2
    elif 'Single' in text or 'single' in text:
        return 1
    return len(re.findall(r'\d+', text))

def extract_main_camera_mp(text):
    if pd.isnull(text):
        return np.nan
    match = re.search(r'(\d+)\s*MP', text)
    return float(match.group(1)) if match else np.nan

def extract_numeric_version(text):
    if pd.isnull(text):
        return np.nan
    match = re.search(r'(\d+(\.\d+)?)', text)
    return float(match.group(1)) if match else np.nan

# Apply transformations
X['Ram'] = X['Ram'].apply(extract_number)
X['Battery'] = X['Battery'].apply(extract_number)
X['Display'] = X['Display'].apply(extract_number)
X['Inbuilt_memory'] = X['Inbuilt_memory'].apply(extract_number)
X['External_Memory'] = X['External_Memory'].apply(external_memory_to_gb)
X['fast_charging'] = X['fast_charging'].apply(extract_number)
X['No_of_sim'] = X['No_of_sim'].apply(count_sims)
X['Camera'] = X['Camera'].apply(extract_main_camera_mp)
X['Android_version'] = X['Android_version'].apply(extract_numeric_version)

# Create and fit pipeline
pipeline = create_data_pipeline()
pipeline.fit(X)
save_pipeline(pipeline)

# Transform features
X_transformed = pipeline.transform(X)

# Split data
X_train, X_val, y_train, y_val = split_data(X_transformed, y)

# Train model
model = training_pipeline(X_train, y_train)

# Predict and evaluate
preds = prediction_pipeline(X_val)
evaluation_matrices(preds, y_val)
