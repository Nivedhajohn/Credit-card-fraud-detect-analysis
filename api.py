# scripts/api.py
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from .preprocessing import preprocess_inference_pipeline

# --- 1. Load Model Artifacts (Global Scope) ---
try:
    # Load all the necessary objects saved from train_model.py
    MODEL = joblib.load("models/fraud_rf.joblib")
    SCALER = joblib.load("models/scaler.joblib")
    FEATURE_COLUMNS = joblib.load("models/feature_columns.joblib")
    print("✅ Model artifacts loaded successfully.")
except FileNotFoundError:
    print("❌ Error: Model artifacts not found. Run 'python scripts/train_model.py --csv data/credit_card.csv' first.")
    exit()

# --- 2. Define Input Schema (Pydantic) ---
# This ensures API inputs are validated and structured correctly.
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

# --- 3. Initialize FastAPI App ---
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time inference API for credit card fraud detection.",
    version="1.0.0"
)

# --- 4. Define API Endpoints ---

@app.get("/")
def home():
    """Simple health check endpoint."""
    return {"status": "ok", "model_loaded": True}

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    """
    Accepts a single transaction and returns the probability of fraud.
    """
    # 1. Convert Pydantic object to a dictionary and then to a pandas DataFrame
    input_data_dict = transaction.dict()
    input_df = pd.DataFrame([input_data_dict])
    
    # 2. Preprocess the data using the trained scaler and feature list
    input_scaled = preprocess_inference(input_df, SCALER, FEATURE_COLUMNS)

    # 3. Predict probability
    # predict_proba returns [[prob_0, prob_1]], we want prob_1 (fraud)
    proba = MODEL.predict_proba(input_scaled)[0, 1]
    
    # 4. Return result
    return {
        "fraud_probability": round(proba, 4),
        "prediction": "Fraud" if proba > 0.5 else "Not Fraud" # Use a default threshold of 0.5
    }
    # scripts/api.py
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# Import the feature engineering functions from the local preprocessing module
from .preprocessing import preprocess_inference_pipeline 

# --- 1. Load Model Artifacts ---
# Define paths relative to the project root (CCFD/)
MODEL_PATH = "models/fraud_rf.joblib"
SCALER_PATH = "models/scaler.joblib"
FEATURE_COLUMNS_PATH = "models/feature_columns.joblib"

# Load model artifacts (assuming train_model has successfully run)
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
except FileNotFoundError as e:
    print(f"ERROR: Model artifacts not found. Did you run train_model.py first? Missing file: {e.filename}")
    raise e

# --- 2. Define API Schemas ---
# Define the expected input fields for a single transaction
class TransactionInput(BaseModel):
    # Include all features your model was trained on
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
    # Add any other required features here...

class PredictionOutput(BaseModel):
    is_fraud_proba: float
    prediction: int

# --- 3. Initialize FastAPI ---
app = FastAPI(title="Credit Card Fraud Detection API")

# --- 4. Define Prediction Endpoint ---
@app.post("/predict", response_model=PredictionOutput)
def predict_fraud(transaction: TransactionInput):
    """
    Accepts a single transaction and returns the fraud probability.
    """
    # 1. Convert input to DataFrame (1 row)
    sample_df = pd.DataFrame([transaction.dict()])

    # 2. Preprocess (using the function from preprocessing.py)
    # This step applies feature engineering, aligns columns, and scales the data.
    X_processed = preprocess_inference_pipeline(sample_df, scaler, feature_columns)

    # 3. Predict
    proba = model.predict_proba(X_processed)[0, 1]
    prediction = 1 if proba > 0.5 else 0 # Simple threshold

    return {
        "is_fraud_proba": np.float64(proba),
        "prediction": prediction
    }

# Health Check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True}