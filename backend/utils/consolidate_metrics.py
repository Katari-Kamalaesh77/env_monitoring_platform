import pandas as pd
import os

# Create a directory for the consolidated metrics if it doesn't exist
os.makedirs("consolidated_metrics", exist_ok=True)

# Load the evaluation metrics for each model (from saved CSVs)
prophet_metrics = pd.read_csv(r"D:\env_monitoring_platform\backend\weather_data_files\weather_model_prophet_metrics.csv")
sarima_metrics = pd.read_csv(r"D:\env_monitoring_platform\backend\weather_data_files\weather_model_sarima_metrics.csv")
xgboost_metrics = pd.read_csv(r"D:\env_monitoring_platform\backend\weather_data_files\xgboost_weather_multivariate_metrics.csv")

# Combine the metrics into one DataFrame
all_metrics = pd.concat([prophet_metrics, sarima_metrics, xgboost_metrics], ignore_index=True)

# Save the consolidated metrics to a CSV file
consolidated_metrics_file = "D:/env_monitoring_platform/backend/weather_data_files/weather_forecasting_metrics.csv"
all_metrics.to_csv(consolidated_metrics_file, index=False)

print(f"Consolidated evaluation metrics saved to {consolidated_metrics_file}")
