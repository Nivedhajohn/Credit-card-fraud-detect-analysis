# scripts/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- Helper Functions (Moved from train_model.py) ---
def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in kilometers."""
    # ... (Your haversine_km implementation) ...
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a.clip(0,1)))
    R = 6371.0
    return R * c

def frequency_encode(series):
    """Vectorized frequency encoding (Warning: relies on global value counts)."""
    vc = series.value_counts(dropna=True)
    return series.map(vc).fillna(0)

def safe_parse_datetime(s):
    return pd.to_datetime(s, errors="coerce")


# --- Feature Engineering Pipeline (Moved from train_model.py) ---
def preprocess_dataframe(df, drop_pii=True, verbose=True):
    # This is the exact copy of your current logic
    df = df.copy()
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    
    # parse datetimes
    if "trans_date_trans_time" in df.columns:
        df["trans_date_trans_time"] = safe_parse_datetime(df["trans_date_trans_time"])
        df["hour"] = df["trans_date_trans_time"].dt.hour.fillna(-1).astype(int)
        df["day"] = df["trans_date_trans_time"].dt.day.fillna(-1).astype(int)
        df["day_of_week"] = df["trans_date_trans_time"].dt.weekday.fillna(-1).astype(int)
        df["month"] = df["trans_date_trans_time"].dt.month.fillna(-1).astype(int)
        df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
    
    # age from dob
    if "dob" in df.columns and "trans_date_trans_time" in df.columns:
        df["dob"] = safe_parse_datetime(df["dob"])
        df["age"] = ((df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25)
        df["age"] = df["age"].fillna(df["age"].median())

    # compute distance
    if {"lat","long","merch_lat","merch_long"}.issubset(df.columns):
        df["distance_km"] = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
        df["distance_km"] = df["distance_km"].fillna(0.0)
    
    # map gender
    if "gender" in df.columns:
        df["gender_mapped"] = df["gender"].map({"M":1,"F":0,"Male":1,"Female":0}).fillna(-1).astype(int)
        df = df.drop(columns=["gender"]) 

    # frequency encode (CRITICAL: This is non-robust for API without saved maps!)
    freq_cols = [c for c in ["merchant","job","city","category","zip","state"] if c in df.columns]
    for c in freq_cols:
        df[f"{c}_freq"] = frequency_encode(df[c]) # This will only count the single row in the API!
    
    # fill numeric missing with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols:
        df[c] = df[c].fillna(df[c].median())

    # drop PII / original columns (mandatory for inference)
    drop_cols = []
    for col in ["first","last","street","cc_num","trans_num","trans_date_trans_time","dob","merchant","job","city","category","zip","state","lat","long","merch_lat","merch_long"]:
        if col in df.columns:
            drop_cols.append(col)
    if drop_pii:
        df = df.drop(columns=drop_cols, errors="ignore")
    
    df = df.fillna(0)
    return df

# --- Inference Pipeline (NEW) ---
def preprocess_inference_pipeline(sample_df, scaler, feature_columns):
    """
    Applies the full feature engineering, scaling, and alignment for API input.
    """
    # 1. Feature Engineering
    df_engineered = preprocess_dataframe(sample_df, drop_pii=True, verbose=False)
    
    # 2. Alignment
    # Must ensure the input is exactly the same shape and order as the training data
    df_aligned = df_engineered.reindex(columns=feature_columns, fill_value=0)
    
    # 3. Scaling
    sample_scaled = scaler.transform(df_aligned)
    
    return sample_scaled