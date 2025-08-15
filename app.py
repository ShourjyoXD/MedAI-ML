# app.py
from flask import Flask, request, jsonify
import joblib
import os
import numpy as np

app = Flask(__name__)

# Define the path to your trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'dummy_model.pkl')

# Load the model globally when the app starts
# This avoids reloading the model for every prediction request
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Make sure to run create_dummy_model.py.")
    model = None # Set model to None to handle cases where it's not loaded
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return "MedAI ML Service is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'ML model not loaded'}), 500

    data = request.get_json(force=True)

    if not data or 'features' not in data:
        return jsonify({'error': 'Invalid input: "features" field is required'}), 400

    features = data['features']

    # Basic input validation: ensure features is a list
    if not isinstance(features, list):
        return jsonify({'error': 'Invalid input: "features" must be a list'}), 400

    # Convert features to a NumPy array, assuming a single sample for now
    try:
        # Reshape for scikit-learn if it expects 2D array (e.g., [[feature1]])
        input_data = np.array(features).reshape(1, -1)
    except Exception as e:
        return jsonify({'error': f'Error processing input features: {e}'}), 400

    try:
        prediction = model.predict(input_data).tolist() # .tolist() to make it JSON serializable
        # If your model also outputs probabilities
        # prediction_proba = model.predict_proba(input_data).tolist()
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {e}'}), 500

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    # You can change the port if 5000 is taken by your Node.js backend
    app.run(host='0.0.0.0', port=5001, debug=True)