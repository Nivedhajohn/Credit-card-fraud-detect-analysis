from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Define input structure
class Transaction(BaseModel):
    amt: float
    merchant: str
    category: str
    gender: str
    city_pop: int
    city: str
    state: str
    zip: int
    lat: float
    long: float

# Load your trained model
model = joblib.load("fraud_model.pkl")

@app.post("/predict")
def predict_fraud(data: Transaction):
    # Example preprocessing (adapt to your model)
    features = [[
        data.amt, data.city_pop, data.lat, data.long
    ]]
    prediction = model.predict(features)[0]
    result = "Fraudulent" if prediction == 1 else "Legit"
    return {"prediction": result}
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # <-- Import the middleware
from pydantic import BaseModel
# ... (other imports like joblib, pandas)

# --- Define input schema ---
class Transaction(BaseModel):
    # ... (your transaction fields)
    pass

# --- Create FastAPI app ---
app = FastAPI()

# app.py

# ... (Existing imports like fastapi, joblib, requests, os)

# --------------------------------------------------------------------
# ⚠️ CONFIGURATION: PASTE YOUR DIRECT DOWNLOAD LINKS HERE
# --------------------------------------------------------------------

# 1. Replace "YOUR_RF_MODEL_DIRECT_LINK_HERE" with the direct link for fraud_rf.joblib
MODEL_URL = "https://https://drive.google.com/uc?export=download&id=1jiQAF_Q4I0iQrHQSOrLIBcF4wS_HSBcM"
# 2. Replace "YOUR_SCALER_MODEL_DIRECT_LINK_HERE" with the direct link for scaler.joblib
SCALER_URL = "https://drive.google.com/uc?export=download&id=1SUgmB2tkUIhTEegU8Gmn38o8ogwsqEG5"

# 3. Replace "YOUR_FEATURES_MODEL_DIRECT_LINK_HERE" with the direct link for feature_columns.joblib
FEATURES_URL = "https://drive.google.com/uc?export=download&id=155IUwlrDturmpOgwM6NEY_ZC9Kwg3Lfc"

MODEL_PATH = "models/fraud_rf.joblib"
SCALER_PATH = "models/scaler.joblib"
FEATURES_PATH = "models/feature_columns.joblib"

# --------------------------------------------------------------------

# ... (The rest of your code, including the download_file function and the app = FastAPI() line)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, GET, etc.
    allow_headers=["*"],  # Allows all headers
)

# ----------------------------------------------------

# --- Load trained artifacts ---
model = joblib.load("models/fraud_rf.joblib")
scaler = joblib.load("models/scaler.joblib")
features = joblib.load("models/feature_columns.joblib")

# --- Prediction endpoint ---
@app.post("/predict")
def predict(transaction: Transaction):
    # ... (your prediction logic)
    pass
