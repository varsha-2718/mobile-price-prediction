import streamlit as st
import pandas as pd
import pickle
import os
from dotenv import load_dotenv
from helper_functions import log_info, log_error

# Load environment variables
load_dotenv()

# Define base paths dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
ARTIFACTS_DIR = os.path.join(BASE_DIR, os.getenv('ARTIFACTS_DIR'))
DATA_OUTPUT_DIR = os.path.join(BASE_DIR, os.getenv('DATA_DIR'), "output")

# Ensure output directory exists
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

# Define model and pipeline paths
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.pkl")
PIPELINE_PATH = os.path.join(ARTIFACTS_DIR, "data_pipeline.pkl")

def load_artifact(filepath):
    try:
        with open(filepath, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        log_error(f"Artifact not found: {filepath}")
        st.error(f"Error: Artifact not found: {filepath}")
        return None

def prepare_input(input_data):
    # Columns your pipeline expects, from your main.py transformations
    required_cols = [
        'External_Memory', 'Inbuilt_memory', 'Android_version', 'Spec_score',
        'No_of_sim', 'fast_charging', 'Ram', 'Camera', 'Battery', 'Rating', 'Display'
    ]

    mapped_input = {
        'External_Memory': input_data.get('External_Memory', 0),
        'Inbuilt_memory': input_data.get('Inbuilt_memory', 0),
        'Android_version': input_data.get('Android_version', 0),
        'Spec_score': input_data.get('Spec_score', 0),
        'No_of_sim': input_data.get('No_of_sim', 1),
        'fast_charging': input_data.get('fast_charging', 0),
        'Ram': input_data.get('Ram', 0),
        'Camera': input_data.get('Camera', 0),
        'Battery': input_data.get('Battery', 0),
        'Rating': input_data.get('Rating', 0),
        'Display': input_data.get('Display', 0),
    }

    df = pd.DataFrame([mapped_input], columns=required_cols)
    return df

def predict_price(input_data):
    pipeline = load_artifact(PIPELINE_PATH)
    model = load_artifact(MODEL_PATH)
    
    if not pipeline or not model:
        return None

    df_input = prepare_input(input_data)
    transformed_input = pipeline.transform(df_input)
    prediction = model.predict(transformed_input)
    return prediction[0]

# Streamlit UI
st.title("📱 Mobile Phone Price Prediction App")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Single Prediction", "Batch Prediction"])

if page == "Single Prediction":
    st.header("Enter Phone Specifications")

    external_memory = st.number_input("External Memory (GB)", min_value=0, max_value=2048, value=0)
    inbuilt_memory = st.number_input("Inbuilt Memory (GB)", min_value=0, max_value=512, value=16)
    android_version = st.number_input("Android Version (e.g., 10.0)", min_value=0.0, max_value=20.0, value=10.0)
    spec_score = st.number_input("Spec Score", min_value=0, max_value=100, value=50)
    no_of_sim = st.number_input("Number of SIMs", min_value=1, max_value=4, value=2)
    fast_charging = st.selectbox("Fast Charging (0=No, 1=Yes)", [0, 1])
    ram = st.number_input("RAM (MB)", min_value=256, max_value=16384, value=2048)
    camera = st.number_input("Camera (MP)", min_value=0, max_value=108, value=12)
    battery = st.number_input("Battery (mAh)", min_value=500, max_value=6000, value=3000)
    rating = st.number_input("User Rating (1-5)", min_value=1, max_value=5, value=3)
    display = st.number_input("Display Size (inch)", min_value=3, max_value=10, value=6)

    if st.button("Predict Price Category"):
        input_data = {
            'External_Memory': external_memory,
            'Inbuilt_memory': inbuilt_memory,
            'Android_version': android_version,
            'Spec_score': spec_score,
            'No_of_sim': no_of_sim,
            'fast_charging': fast_charging,
            'Ram': ram,
            'Camera': camera,
            'Battery': battery,
            'Rating': rating,
            'Display': display,
        }
        prediction = predict_price(input_data)
        if prediction is not None:
            st.success(f"Predicted Price Category: {prediction}")
            log_info(f"Predicted Price Category: {prediction}")

elif page == "Batch Prediction":
    st.header("Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV for Batch Prediction", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Important: Check if all required columns are present in batch file
        required_cols = [
            'External_Memory', 'Inbuilt_memory', 'Android_version', 'Spec_score',
            'No_of_sim', 'fast_charging', 'Ram', 'Camera', 'Battery', 'Rating', 'Display'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns in uploaded file: {missing_cols}")
        else:
            pipeline = load_artifact(PIPELINE_PATH)
            model = load_artifact(MODEL_PATH)

            if pipeline and model:
                transformed_data = pipeline.transform(df[required_cols])
                predictions = model.predict(transformed_data)
                df['Predicted Price Category'] = predictions

                output_file = os.path.join(DATA_OUTPUT_DIR, "batch_predictions.csv")
                df.to_csv(output_file, index=False)

                st.write(df)
                st.success(f"Batch Prediction Completed! Results saved at {output_file}")
                log_info("Batch Prediction Completed Successfully!")
