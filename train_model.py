# train_model.py
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from sklearn.utils import compute_class_weight
import warnings
warnings.filterwarnings("ignore")
from .preprocessing import preprocess_dataframe



# --- Main training routine ---
def train(csv_path="creditcard.csv",
          model_out="models/fraud_rf.joblib",
          scaler_out="models/scaler.joblib",
          features_out="models/feature_columns.joblib",
          use_smote=False,
          random_state=42):
    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)

    df = pd.read_csv("creditcard.csv")
    target_col = "is_fraud"
    if target_col not in df.columns:
        raise ValueError(f"Expected target column '{target_col}' not found in CSV.")

    print("Loaded:", csv_path, "shape:", df.shape)

    # Preprocess & feature engineer
    df_proc = preprocess_dataframe(df, drop_pii=True, verbose=True)
    print("After preprocessing shape:", df_proc.shape)

    # Prepare X, y
    y = df[target_col].astype(int).values
    # remove target column if it accidentally persisted in df_proc
    if target_col in df_proc.columns:
        df_proc = df_proc.drop(columns=[target_col])
    X = df_proc.copy()

    # Save the feature columns order for later inference
    feature_columns = X.columns.tolist()

    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state
    )
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Train target proportion:", (y_train.sum()/len(y_train)))

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Optionally use SMOTE if requested AND imblearn present
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=random_state, n_jobs=-1)
            X_train_scaled, y_train = sm.fit_resample(X_train_scaled, y_train)
            print("SMOTE applied. New positive count:", y_train.sum(), "Total:", len(y_train))
        except Exception as e:
            print("imblearn not installed or SMOTE failed:", e)
            print("Proceeding without SMOTE and using class weight in model.")

    # Class weights (if not using SMOTE)
    class_weight = "balanced" if not use_smote else None

    # Model training (RandomForest baseline)
    clf = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=random_state,
        class_weight=class_weight
    )
    print("Training RandomForest...")
    clf.fit(X_train_scaled, y_train)

    # Predictions & metrics
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]
    y_pred = clf.predict(X_test_scaled)

    print("\n--- Evaluation on test set ---")
    print(classification_report(y_test, y_pred, digits=4))
    try:
        roc = roc_auc_score(y_test, y_proba)
    except Exception:
        roc = None
    pr_auc = average_precision_score(y_test, y_proba)
    print(f"ROC AUC: {roc:.4f}" if roc is not None else "ROC AUC: (could not compute)")
    print(f"PR AUC (average precision): {pr_auc:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Save artifacts
    joblib.dump(clf, model_out)
    joblib.dump(scaler, scaler_out)
    joblib.dump(feature_columns, features_out)
    print(f"\nSaved model -> {model_out}")
    print(f"Saved scaler -> {scaler_out}")
    print(f"Saved feature columns list -> {features_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--csv", default="credit card.csv", help="Path to CSV (default: credit card.csv)")
    parser.add_argument("--out_model", default="models/fraud_rf.joblib", help="Output model path")
    parser.add_argument("--out_scaler", default="models/scaler.joblib", help="Output scaler path")
    parser.add_argument("--out_features", default="models/feature_columns.joblib", help="Output feature columns list")
    parser.add_argument("--use_smote", action="store_true", help="Use SMOTE if imblearn available")
    args = parser.parse_args()
    train(csv_path=args.csv, model_out=args.out_model, scaler_out=args.out_scaler,
          features_out=args.out_features, use_smote=args.use_smote)
   


