"""
File: src/visualizer.py
Purpose: Generate and save high-resolution analytics charts for GitHub portfolio proof.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def generate_portfolio_plots(
    test_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
    output_dir: str = "images"
):
    """Creates 3 industry-ready plots in images/ directory."""
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Actual vs Predicted Time-Series Horizon (Last 120 Hours)
    subset = test_df.tail(120)
    fig, ax = plt.subplots(figsize=(13, 5), dpi=150)
    ax.plot(subset["timestamp"], subset["energy_consumption_kw"], label="Actual Meter Load (kW)", color="#0f172a", linewidth=2)
    ax.plot(subset["timestamp"], subset["predicted_rf_kw"], label="Random Forest Forecast (kW)", color="#10b981", linestyle="--", linewidth=2)
    ax.set_title("AI Energy Consumption Forecast: Actual vs. Predicted (120-Hour Horizon)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Power Demand (kW)")
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/actual_vs_predicted.png")
    plt.close()

    # 2. Feature Importance Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    sns.barplot(
        data=feature_importance_df.head(10),
        x="importance",
        y="feature",
        palette="viridis",
        ax=ax
    )
    ax.set_title("Top 10 Predictive Features Driving Energy Demand", fontsize=13, fontweight="bold")
    ax.set_xlabel("Relative Importance Score")
    ax.set_ylabel("Engineered Feature")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png")
    plt.close()

    # 3. Prediction Error Residuals Scatter Plot
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    ax.scatter(test_df["energy_consumption_kw"], test_df["predicted_rf_kw"], alpha=0.35, color="#0284c7", edgecolors="none")
    min_val = test_df["energy_consumption_kw"].min()
    max_val = test_df["energy_consumption_kw"].max()
    ax.plot([min_val, max_val], [min_val, max_val], color="#dc2626", linestyle="--", linewidth=1.8, label="Ideal 1:1 Fit")
    ax.set_title("Model Calibration: Actual vs. Predicted Scatter", fontsize=13, fontweight="bold")
    ax.set_xlabel("Actual Consumption (kW)")
    ax.set_ylabel("Predicted Consumption (kW)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/regression_calibration_scatter.png")
    plt.close()

    print(f"[VISUALIZER] Saved 3 portfolio charts to '{output_dir}/'")
