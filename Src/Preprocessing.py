import pandas as pd
import numpy as np


def load_and_clean_data(filepath: str = "data/raw_energy_consumption.csv") -> pd.DataFrame:
    """
    Loads raw energy CSV, parses timestamps, interpolates missing values,
    and smooths extreme sensor glitches using IQR clipping.
    """
    df = pd.read_csv(filepath)

    # 1. Convert timestamp to datetime and sort chronologically
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 2. Check and report missing values before cleaning
    missing_before = df["energy_consumption_kw"].isna().sum()
    print(f"[PREPROCESS] Missing readings detected: {missing_before}")

    # 3. Time-based linear interpolation (ideal for continuous hourly energy sensors)
    df["energy_consumption_kw"] = df["energy_consumption_kw"].interpolate(method="linear")
    df["temperature_c"] = df["temperature_c"].interpolate(method="linear")
    df["humidity_pct"] = df["humidity_pct"].interpolate(method="linear")

    # Backfill/forward-fill any edge NaNs at index 0
    df = df.bfill().ffill()

    # 4. Cap extreme physical sensor spikes using Interquartile Range (IQR)
    q1 = df["energy_consumption_kw"].quantile(0.01)
    q99 = df["energy_consumption_kw"].quantile(0.99)
    iqr = q99 - q1
    lower_bound = max(50.0, q1 - 1.5 * iqr)
    upper_bound = q99 + 1.5 * iqr

    df["energy_consumption_kw"] = np.clip(df["energy_consumption_kw"], lower_bound, upper_bound)

    print(f"[PREPROCESS] Cleaned dataset shape: {df.shape} | Missing after: {df.isna().sum().sum()}")
    return df
  
