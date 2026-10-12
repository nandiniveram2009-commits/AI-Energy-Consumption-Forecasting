"""
File: src/forecaster.py
Purpose: Generate future hourly consumption forecasts and simulate automated peak-load optimization.
"""

import os
import pandas as pd
import numpy as np


def simulate_smart_grid_optimization(
    test_df: pd.DataFrame,
    peak_tariff_usd: float = 0.24,
    off_peak_tariff_usd: float = 0.11,
    shift_efficiency: float = 0.18,
    output_csv: str = "outputs/forecast_and_optimization_report.csv"
) -> dict:
    """
    Applies virtual business decision rules on top of model forecasts:
    When predicted load falls in Peak Tariff hours (17:00-21:00) and exceeds threshold,
    shift flexible load (18%) to pre-cooling off-peak hours (13:00-15:00).
    """
    sim_df = test_df.copy()

    # Calculate baseline tariff rate per hour
    sim_df["tariff_rate_usd"] = np.where(sim_df["is_peak_hour"] == 1, peak_tariff_usd, off_peak_tariff_usd)

    # Simulate smart automated load shifting based on AI forecast
    sim_df["optimized_load_kw"] = sim_df["predicted_rf_kw"]
    peak_mask = sim_df["is_peak_hour"] == 1
    precool_mask = sim_df["hour"].between(13, 15)

    sim_df.loc[peak_mask, "optimized_load_kw"] = sim_df.loc[peak_mask, "predicted_rf_kw"] * (1.0 - shift_efficiency)
    sim_df.loc[precool_mask, "optimized_load_kw"] = sim_df.loc[precool_mask, "predicted_rf_kw"] * (1.0 + shift_efficiency * 0.45)

    # Compute financial costs
    sim_df["baseline_cost_usd"] = sim_df["energy_consumption_kw"] * sim_df["tariff_rate_usd"]
    sim_df["optimized_cost_usd"] = sim_df["optimized_load_kw"] * sim_df["tariff_rate_usd"]

    total_baseline_cost = float(sim_df["baseline_cost_usd"].sum())
    total_optimized_cost = float(sim_df["optimized_cost_usd"].sum())
    total_savings_usd = total_baseline_cost - total_optimized_cost

    peak_kwh_shaved = float(
        (sim_df.loc[peak_mask, "energy_consumption_kw"] - sim_df.loc[peak_mask, "optimized_load_kw"]).clip(lower=0).sum()
    )
    carbon_saved_kg = peak_kwh_shaved * 0.42  # 0.42 kg CO2 per peak kWh avoided

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    sim_df[[
        "timestamp", "temperature_c", "energy_consumption_kw",
        "predicted_rf_kw", "optimized_load_kw", "baseline_cost_usd", "optimized_cost_usd"
    ]].tail(168).to_csv(output_csv, index=False)

    summary = {
        "total_baseline_cost_usd": round(total_baseline_cost, 2),
        "total_optimized_cost_usd": round(total_optimized_cost, 2),
        "net_cost_savings_usd": round(total_savings_usd, 2),
        "peak_kwh_shaved": round(peak_kwh_shaved, 2),
        "carbon_emissions_reduced_kg": round(carbon_saved_kg, 2),
    }

    print("\n--- Virtual Business Simulation Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v:,}")

    return summary
