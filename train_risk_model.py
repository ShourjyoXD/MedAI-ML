import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
# import kagglehub # <--- Can comment out or remove this line

# --- 0. NO LONGER Downloading via KaggleHub directly in script if data is local ---
# If you manually moved cardio_train.csv to MedAI-ML/data/, use this path:
CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'cardio_train.csv')

# --- 1. Data Loading (Real Dataset) ---
print(f"\nLoading data from {CSV_FILE_PATH}...")
try:
    df = pd.read_csv(CSV_FILE_PATH, sep=';') # The dataset uses semicolon as a separator
except FileNotFoundError:
    print(f"Error: {CSV_FILE_PATH} not found. Make sure you downloaded the dataset and placed it in the 'data' folder inside MedAI-ML.")
    exit()

print("Original Data head:")
print(df.head())
print(f"Original Data shape: {df.shape}")

# --- 1.1 Data Preprocessing & Cleaning ---
# Drop 'id' column as it's not a feature
df = df.drop('id', axis=1)

# Convert 'age' from days to years
df['age'] = df['age'] / 365.25 # Convert days to years for better interpretability
df['age'] = df['age'].astype(int)

# Calculate BMI
df['bmi'] = df['weight'] / ((df['height'] / 100)**2) # height is in cm, convert to meters

# Basic outlier removal for blood pressure (optional, but good practice for real data)
df = df[(df['ap_hi'] < 250) & (df['ap_hi'] > 70)] # Systolic reasonable range
df = df[(df['ap_lo'] < 180) & (df['ap_lo'] > 40)]  # Diastolic reasonable range
df = df[df['ap_lo'] < df['ap_hi']] # Diastolic must be less than Systolic

# Handle 'gender' if it's not already 0/1 (dataset has 1 and 2)
df['gender'] = df['gender'].map({1: 0, 2: 1}) # 1: male -> 0, 2: female -> 1

print("\nProcessed Data head:")
print(df.head())
print(f"Processed Data shape: {df.shape}")

print("\nCVD (Risk) distribution in processed data:")
print(df['cardio'].value_counts(normalize=True))

# --- 2. Feature Selection & Target ---
FEATURES = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
            'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi']
TARGET = 'cardio'

X = df[FEATURES]
y = df[TARGET]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTraining data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Scale numerical features
numerical_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']

scaler = StandardScaler()
X_train_scaled_numerical = scaler.fit_transform(X_train[numerical_features])
X_test_scaled_numerical = scaler.transform(X_test[numerical_features])

# Reconstruct X_train_scaled and X_test_scaled DataFrames
X_train_scaled = pd.DataFrame(X_train_scaled_numerical, columns=numerical_features, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled_numerical, columns=numerical_features, index=X_test.index)

# Add back the non-scaled features
for col in X.columns:
    if col not in numerical_features:
        X_train_scaled[col] = X_train[col]
        X_test_scaled[col] = X_test[col]

# Ensure the order of columns is consistent before training
X_train_scaled = X_train_scaled[FEATURES]
X_test_scaled = X_test_scaled[FEATURES]

print("\nFeatures scaled and prepared.")

# --- 3. Model Training ---
print("Training Logistic Regression model...")
model = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
model.fit(X_train_scaled, y_train)
print("Model training complete.")

# --- 4. Model Evaluation ---
print("\nEvaluating model performance...")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)

# --- 5. Save the Model and Scaler ---
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, 'cvd_risk_model.pkl')
scaler_path = os.path.join(models_dir, 'cvd_scaler.pkl')
features_path = os.path.join(models_dir, 'cvd_model_features.joblib')

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(FEATURES, features_path)

print(f"\nModel saved to: {model_path}")
print(f"Scaler saved to: {scaler_path}")
print(f"Features saved to: {features_path}")
print("Training script finished.")