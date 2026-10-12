"""
File: src/model_trainer.py
Purpose: Train and benchmark Linear Regression vs. Random Forest Regressor using chronological splitting.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    """Computes RMSE, MAE, MAPE (%), and R² score."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    r2 = float(r2_score(y_true, y_pred))

    print(f"\n--- {model_name} Performance ---")
    print(f"RMSE : {rmse:.2f} kW")
    print(f"MAE  : {mae:.2f} kW")
    print(f"MAPE : {mape:.2f} %")
    print(f"R²   : {r2:.4f}")

    return {"model": model_name, "rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def train_and_evaluate_models(df: pd.DataFrame, model_save_path: str = "models/random_forest_energy.pkl"):
    """
    Performs strict chronological 80/20 train-test split, trains baseline and ensemble models,
    and persists the trained Random Forest pipeline.
    """
    split_idx = int(len(df) * 0.80)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    print(f"[SPLIT] Training samples: {len(X_train):,} | Testing samples: {len(X_test):,}")

    # 1. Baseline Model: Multiple Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_metrics = evaluate_predictions(y_test.values, lr_preds, "Linear Regression (Baseline)")

    # 2. Primary Industry Model: Random Forest Regressor
    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=14,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_metrics = evaluate_predictions(y_test.values, rf_preds, "Random Forest Regressor")

    # Save predictions and feature importances for downstream visualization
    test_df["predicted_lr_kw"] = np.round(lr_preds, 2)
    test_df["predicted_rf_kw"] = np.round(rf_preds, 2)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(rf_model, model_save_path)
    print(f"\n[SAVED] Trained Random Forest model saved to '{model_save_path}'")

    feature_importance_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": rf_model.feature_importances_
    }).sort_values("importance", ascending=False)

    return rf_model, test_df, feature_importance_df, [lr_metrics, rf_metrics]
