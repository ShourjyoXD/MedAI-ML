import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'cardio_train.csv')


print(f"\nLoading data from {CSV_FILE_PATH}...")
try:
    df = pd.read_csv(CSV_FILE_PATH, sep=';') 
except FileNotFoundError:
    print(f"Error: {CSV_FILE_PATH} not found. Make sure you downloaded the dataset and placed it in the 'data' folder inside MedAI-ML.")
    exit()

print("Original Data head:")
print(df.head())
print(f"Original Data shape: {df.shape}")

df = df.drop('id', axis=1)

df['age'] = df['age'] / 365.25 
df['age'] = df['age'].astype(int)
df['bmi'] = df['weight'] / ((df['height'] / 100)**2) 

df = df[(df['ap_hi'] < 250) & (df['ap_hi'] > 70)] 
df = df[(df['ap_lo'] < 180) & (df['ap_lo'] > 40)]  
df = df[df['ap_lo'] < df['ap_hi']] 

df['gender'] = df['gender'].map({1: 0, 2: 1}) 

print("\nProcessed Data head:")
print(df.head())
print(f"Processed Data shape: {df.shape}")

print("\nCVD (Risk) distribution in processed data:")
print(df['cardio'].value_counts(normalize=True))

FEATURES = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
            'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi']
TARGET = 'cardio'

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTraining data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

numerical_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']

scaler = StandardScaler()
X_train_scaled_numerical = scaler.fit_transform(X_train[numerical_features])
X_test_scaled_numerical = scaler.transform(X_test[numerical_features])

X_train_scaled = pd.DataFrame(X_train_scaled_numerical, columns=numerical_features, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled_numerical, columns=numerical_features, index=X_test.index)


for col in X.columns:
    if col not in numerical_features:
        X_train_scaled[col] = X_train[col]
        X_test_scaled[col] = X_test[col]


X_train_scaled = X_train_scaled[FEATURES]
X_test_scaled = X_test_scaled[FEATURES]

print("\nFeatures scaled and prepared.")

print("Training Logistic Regression model...")
model = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
model.fit(X_train_scaled, y_train)
print("Model training complete.")

print("\nEvaluating model performance...")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)

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