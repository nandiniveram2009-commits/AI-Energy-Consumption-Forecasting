import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates industry-standard time-series features for supervised ML forecasting.
    """
    feat_df = df.copy()

    # 1. Calendar & Operational Regime Features
    feat_df["hour"] = feat_df["timestamp"].dt.hour
    feat_df["day_of_week"] = feat_df["timestamp"].dt.dayofweek
    feat_df["month"] = feat_df["timestamp"].dt.month
    feat_df["is_weekend"] = (feat_df["day_of_week"] >= 5).astype(int)

    # Peak tariff window (17:00 to 21:00 is high-cost grid stress period)
    feat_df["is_peak_hour"] = feat_df["hour"].between(17, 21).astype(int)

    # 2. Cyclical Encoding of Hour (so 23:00 and 00:00 are mathematically close)
    feat_df["hour_sin"] = np.sin(2 * np.pi * feat_df["hour"] / 24.0)
    feat_df["hour_cos"] = np.cos(2 * np.pi * feat_df["hour"] / 24.0)

    # 3. Autoregressive Lag Features (Crucial for high R² accuracy)
    feat_df["lag_1h"] = feat_df["energy_consumption_kw"].shift(1)
    feat_df["lag_24h"] = feat_df["energy_consumption_kw"].shift(24)    # Same hour yesterday
    feat_df["lag_168h"] = feat_df["energy_consumption_kw"].shift(168)  # Same hour last week

    # 4. Rolling Window Statistics (shifted by 1 to prevent data leakage!)
    shifted_target = feat_df["energy_consumption_kw"].shift(1)
    feat_df["rolling_mean_6h"] = shifted_target.rolling(window=6).mean()
    feat_df["rolling_mean_24h"] = shifted_target.rolling(window=24).mean()
    feat_df["rolling_std_24h"] = shifted_target.rolling(window=24).std()

    # Drop initial rows with NaN caused by 168-hour lookback window
    feat_df = feat_df.dropna().reset_index(drop=True)
    print(f"[FEATURES] Engineered {feat_df.shape[1]} columns across {len(feat_df):,} valid rows.")
    return feat_df


FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_peak_hour",
    "hour_sin",
    "hour_cos",
    "temperature_c",
    "humidity_pct",
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_6h",
    "rolling_mean_24h",
    "rolling_std_24h",
]

TARGET_COLUMN = "energy_consumption_kw"
