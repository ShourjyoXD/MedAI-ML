# MedAI-ML
MedAI - CVD Risk Prediction ML Model
🚀 Project Overview
This repository holds the Machine Learning model and Flask API for the MedAI project. It's designed to predict Cardiovascular Disease (CVD) risk from user health data, serving as the intelligent core of the MedAI mobile app.

✨ Features
CVD Risk Prediction: Uses a pre-trained scikit-learn model for risk assessment.

Data Preprocessing: Integrates a StandardScaler to normalize incoming data, matching training conditions.

Flask API: Provides a straightforward RESTful API (/predict) for easy integration.

Model Persistence: Loads models and scalers from .pkl files for efficient startup.

🛠️ Technologies Used
Python 3.x

Flask: API framework.

scikit-learn: ML models and preprocessing.

NumPy, Pandas, joblib: For data handling and model loading.

python-dotenv: For environment variable management.

📦 Project Structure
MedAI-ML/
├── app.py                     # Flask app, ML model serving
├── models/
│   ├── cvd_risk_model.pkl     # Pre-trained ML model
│   └── cvd_scaler.pkl         # Pre-trained data scaler
├── requirements.txt           # Python dependencies
└── .env                       # Environment variables
⚙️ Setup and Installation
Clone the repository:

Bash

git clone https://github.com/YourGitHubUsername/MedAI-ML.git
cd MedAI-ML
Create & activate virtual environment:

Bash

python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # macOS/Linux
Install dependencies:

Bash

pip install -r requirements.txt
(If no requirements.txt, create one via pip freeze > requirements.txt after installing flask, scikit-learn, numpy, pandas, joblib, python-dotenv)

Place models: Ensure cvd_risk_model.pkl and cvd_scaler.pkl are in the models/ directory.

Run Flask app:

Bash

python app.py
Server runs on http://192.168.0.114:5001 locally.

💻 API Endpoint
POST /predict
Description: Predicts CVD risk from user health data.

URL (Local): http://192.168.0.114:5001/predict

Method: POST

Content-Type: application/json

Request Body Example:

JSON

{
    "age": 55, "gender": 1, "height": 170, "weight": 80,
    "ap_hi": 140, "ap_lo": 90, "cholesterol": 1, "gluc": 1,
    "smoke": 0, "alco": 0, "active": 1
}
Response Body Example (Success):

JSON

{ "prediction": 0.75, "risk_category": "High", "message": "Prediction successful" }
☁️ Deployment
This model can be deployed to various cloud platforms with free tiers for personal projects, such as Google Cloud Run, Render, Hugging Face Spaces, or Vercel (Serverless Functions). After deployment, remember to update your React Native frontend's API_BASE_URL (e.g., in api/axiosInstance.js) to point to your new public URL.

🤝 Contribution
Contributions are welcome! Fork the repo, make changes, and submit a pull request.

📄 License
MIT License

📧 Contact
For questions, contact shourjyochakraborty2006@gmail.com