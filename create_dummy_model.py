# create_dummy_model.py
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np

# Create a simple dummy model
# This model will just predict based on a single input feature
model = LogisticRegression()
# Dummy data for training (model needs to be trained before saving)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 1, 1, 1])
model.fit(X, y)

# Save the model
model_path = 'models/dummy_model.pkl'
joblib.dump(model, model_path)

print(f"Dummy model saved to {model_path}")