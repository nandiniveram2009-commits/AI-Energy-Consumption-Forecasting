"""
File: main.py
Project: AI-Powered Energy Consumption Forecasting System
Usage: python main.py
"""

import json
import os
from src.data_generator import generate_virtual_energy_dataset
from src.preprocessing import load_and_clean_data
from src.feature_engineering import engineer_features
from src.model_trainer import train_and_evaluate_models
from src.forecaster import simulate_smart_grid_optimization
from src.visualizer import generate_portfolio_plots


def main():
    print("=" * 65)
    print("  AI-POWERED ENERGY CONSUMPTION FORECASTING SYSTEM - PIPELINE")
    print("=" * 65)

    # Phase 1: Generate or Load Smart-Meter Dataset
    raw_csv_path = "data/raw_energy_consumption.csv"
    if not os.path.exists(raw_csv_path):
        generate_virtual_energy_dataset(output_path=raw_csv_path)

    # Phase 2: Data Cleaning & Preprocessing
    clean_df = load_and_clean_data(filepath=raw_csv_path)

    # Phase 3: Time-Series Feature Engineering
    featured_df = engineer_features(clean_df)

    # Phase 4 & 5: Model Training & Evaluation (Linear Regression vs. Random Forest)
    rf_model, test_df, importance_df, metrics_list = train_and_evaluate_models(featured_df)

    # Phase 6: Virtual Smart-Grid Load Shifting Simulation
    business_summary = simulate_smart_grid_optimization(test_df)

    # Phase 7: Save Evaluation Metrics JSON for GitHub Proof
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/evaluation_metrics.json", "w") as f:
        json.dump({"models": metrics_list, "business_impact": business_summary}, f, indent=2)

    # Phase 8: Generate Portfolio Plots
    generate_portfolio_plots(test_df, importance_df, output_dir="images")

    print("\n[SUCCESS] Full pipeline completed! Check 'outputs/' and 'images/' folders.")


if __name__ == "__main__":
    main()
