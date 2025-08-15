# app.py
from flask import Flask, request, jsonify
import joblib
import os
import numpy as np
import pandas as pd

app = Flask(__name__)

# Define paths to your trained model, scaler, and feature names
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'cvd_risk_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'models', 'cvd_scaler.pkl')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), 'models', 'cvd_model_features.joblib')

# Load the model, scaler, and feature names globally when the app starts
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    # The 'model_features' loaded here *should* include 'bmi' if saved that way
    # However, we'll manually filter it out for validation if we calculate it here.
    loaded_model_features = joblib.load(FEATURES_PATH)
    model_features = [f for f in loaded_model_features if f != 'bmi'] # <--- IMPORTANT: Filter 'bmi' if you calculate it
    print(f"Model loaded successfully from {MODEL_PATH}")
    print(f"Scaler loaded successfully from {SCALER_PATH}")
    print(f"Model features (excluding BMI for input validation): {model_features}")
except FileNotFoundError as e:
    print(f"Error: One or more ML components not found: {e}. Make sure to run train_risk_model.py first.")
    model = None
    scaler = None
    loaded_model_features = None # Set to None for safety
    model_features = None # Set to None for safety
except Exception as e:
    print(f"An unexpected error occurred while loading ML components: {e}")
    model = None
    scaler = None
    loaded_model_features = None # Set to None for safety
    model_features = None # Set to None for safety

@app.route('/')
def home():
    return "MedAI ML Service is running!"

@app.route('/predict_risk', methods=['POST'])
def predict_risk():
    if model is None or scaler is None or loaded_model_features is None: # Use loaded_model_features for main check
        return jsonify({'error': 'ML components not loaded. Please check the server startup logs.'}), 500

    data = request.get_json(force=True)

    # Validate input data against expected features (excluding BMI, which we'll calculate)
    # We still need 'height' and 'weight' to be present to calculate BMI
    required_features_for_input = [f for f in loaded_model_features if f != 'bmi'] # Define expected inputs
    missing_features = [f for f in required_features_for_input if f not in data]
    if missing_features:
        return jsonify({'error': f'Missing required features: {", ".join(missing_features)}. Please provide all of {required_features_for_input}'}), 400

    # Calculate BMI if height and weight are provided
    height_cm = data.get('height')
    weight_kg = data.get('weight')

    if height_cm is None or weight_kg is None or height_cm <= 0: # Basic validation for calculation
        return jsonify({'error': 'Height and weight are required to calculate BMI, and height must be positive.'}), 400

    bmi_value = weight_kg / ((height_cm / 100)**2) # Convert cm to meters for BMI formula
    data['bmi'] = bmi_value # Add calculated BMI to the data dictionary

    # Create a Pandas DataFrame from the input data, ensuring correct feature order
    # Now use loaded_model_features which includes BMI, as data now has it
    try:
        input_df = pd.DataFrame([data], columns=loaded_model_features) # Use the full list including 'bmi'
    except Exception as e:
        return jsonify({'error': f'Error processing input data into DataFrame: {e}'}), 400

    # Identify numerical features that need scaling (must match train_risk_model.py)
    numerical_features_to_scale = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi'] # BMI is now included here

    # Separate numerical and non-numerical features
    numerical_data = input_df[numerical_features_to_scale]
    non_numerical_data = input_df.drop(columns=numerical_features_to_scale)

    # Scale the numerical part of the input data
    try:
        input_scaled_numerical = scaler.transform(numerical_data)
        scaled_numerical_df = pd.DataFrame(input_scaled_numerical, columns=numerical_features_to_scale, index=input_df.index)
    except Exception as e:
        return jsonify({'error': f'Error scaling numerical input features: {e}'}), 500

    # Combine scaled numerical features with original non-numerical features
    final_input_df = pd.concat([scaled_numerical_df, non_numerical_data], axis=1)
    final_input_df = final_input_df[loaded_model_features] # Reorder columns to match training features

    try:
        prediction_proba = model.predict_proba(final_input_df).tolist()
        prediction_class = model.predict(final_input_df).tolist()
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {e}'}), 500

    # Determine if an alert should be sent (Rule-based on prediction and raw values)
    send_alert = False
    predicted_risk_is_high = bool(prediction_class[0])

    if predicted_risk_is_high and data.get('ap_hi', 0) >= 160:
        send_alert = True
    elif data.get('ap_lo', 0) >= 110:
        send_alert = True
    elif predicted_risk_is_high and data.get('cholesterol', 1) == 3:
        send_alert = True

    return jsonify({
        'prediction_class': prediction_class[0],
        'prediction_probabilities': {
            'low_risk_proba': prediction_proba[0][0],
            'high_risk_proba': prediction_proba[0][1]
        },
        'send_alert': send_alert,
        'message': 'CVD Risk prediction and alert status based on patient data.'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)