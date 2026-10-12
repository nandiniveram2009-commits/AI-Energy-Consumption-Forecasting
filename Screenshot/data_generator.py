import os
import numpy as np
import pandas as pd


def generate_virtual_energy_dataset(
    output_path: str = "data/raw_energy_consumption.csv",
    hours: int = 8760,  # 365 days * 24 hours
    seed: int = 42
) -> pd.DataFrame:
    """
    Simulates 1 year of hourly building/grid electricity consumption (kW)
    with realistic diurnal cycles, seasonal temperature variations, and peak tariff windows.
    """
    np.random.seed(seed)
    timestamps = pd.date_range(start="2025-01-01 00:00:00", periods=hours, freq="h")

    hour = timestamps.hour.values
    day_of_week = timestamps.dayofweek.values
    day_of_year = timestamps.dayofyear.values
    is_weekend = (day_of_week >= 5).astype(int)

    # 1. Simulate seasonal + daily ambient temperature (Celsius)
    seasonal_temp = 22.0 + 10.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365.0)
    daily_temp_swing = 5.5 * np.sin(2 * np.pi * (hour - 9) / 24.0)
    temperature_c = seasonal_temp + daily_temp_swing + np.random.normal(0, 1.4, size=hours)

    # 2. Simulate relative humidity (%)
    humidity_pct = np.clip(65.0 - 1.2 * daily_temp_swing + np.random.normal(0, 4.5, size=hours), 25.0, 95.0)

    # 3. Simulate bimodal human/industrial activity curve (morning 8 AM & evening 7:30 PM peaks)
    morning_peak = 95.0 * np.exp(-((hour - 8.0) ** 2) / 5.0)
    evening_peak = 165.0 * np.exp(-((hour - 19.5) ** 2) / 6.5)
    base_load_kw = 380.0

    # 4. HVAC cooling/heating non-linear demand when temperature deviates from 21°C
    hvac_load_kw = 9.2 * np.maximum(0, temperature_c - 21.5) + 4.5 * np.maximum(0, 15.0 - temperature_c)

    # 5. Weekend reduction factor for commercial/industrial zones
    weekend_multiplier = np.where(is_weekend == 1, 0.84, 1.0)

    # 6. Combine components + random sensor noise
    noise_kw = np.random.normal(0, 14.0, size=hours)
    energy_kw = (base_load_kw + morning_peak + evening_peak + hvac_load_kw) * weekend_multiplier + noise_kw
    energy_kw = np.round(np.maximum(150.0, energy_kw), 2)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "temperature_c": np.round(temperature_c, 2),
        "humidity_pct": np.round(humidity_pct, 2),
        "energy_consumption_kw": energy_kw
    })

    # Inject a tiny fraction (0.2%) of missing values to demonstrate real-world data cleaning
    missing_indices = np.random.choice(df.index, size=int(hours * 0.002), replace=False)
    df.loc[missing_indices, "energy_consumption_kw"] = np.nan

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Generated {len(df):,} hourly records -> saved to '{output_path}'")
    return df


if __name__ == "__main__":
    generate_virtual_energy_dataset()
  
