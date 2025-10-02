# inference_example.py
import joblib
import pandas as pd
from train_model import preprocess_dataframe  # reuse same preprocessing function

MODEL_PATH = "models/fraud_rf.joblib"
SCALER_PATH = "models/scaler.joblib"
FEATURES_PATH = "models/feature_columns.joblib"

clf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_cols = joblib.load(FEATURES_PATH)

# Example transaction (fill with realistic values from your dataset)
example = {
    "trans_date_trans_time": "2020-06-21 12:14:25",
    "cc_num": 1234567890123456,
    "merchant": "Amazon",
    "category": "electronics",
    "amt": 125.5,
    "first": "John",
    "last": "Doe",
    "gender": "M",
    "street": "123 Main St",
    "city": "Somecity",
    "state": "CA",
    "zip": "90001",
    "lat": 34.05,
    "long": -118.25,
    "city_pop": 100000,
    "job": "Engineer",
    "dob": "1980-01-01",
    "trans_num": 111111,
    "unix_time": 1592734465,
    "merch_lat": 34.0522,
    "merch_long": -118.2437
}

df_ex = pd.DataFrame([example])
df_proc = preprocess_dataframe(df_ex, drop_pii=True, verbose=False)
# Ensure feature columns same order:
X = df_proc.reindex(columns=feature_cols, fill_value=0)
X_scaled = scaler.transform(X)
prob = clf.predict_proba(X_scaled)[:,1][0]
print("Fraud probability:", prob)
