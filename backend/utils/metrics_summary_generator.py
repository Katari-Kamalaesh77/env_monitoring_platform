import pandas as pd
import os

# Create a directory for the consolidated metrics if it doesn't exist
os.makedirs("consolidated_air_metrics", exist_ok=True)

# Load each model's metrics
prophet_metrics = pd.read_csv(r"D:\env_monitoring_platform\backend\ml\prophet_air_metrics.csv")
sarima_metrics = pd.read_csv(r"D:\env_monitoring_platform\backend\ml\sarima_air_metrics.csv")
xgboost_metrics = pd.read_csv(r"D:\env_monitoring_platform\backend\ml\xgboost_air_metrics.csv")

# Fix xgboost_metrics (index was used instead of 'Pollutant' column)
if 'Unnamed: 0' in xgboost_metrics.columns:
    xgboost_metrics.rename(columns={'Unnamed: 0': 'Pollutant'}, inplace=True)

# Add a model label column to each
prophet_metrics['Model'] = 'Prophet'
sarima_metrics['Model'] = 'SARIMA'
xgboost_metrics['Model'] = 'XGBoost'

# Combine all
all_metrics = pd.concat([prophet_metrics, sarima_metrics, xgboost_metrics], ignore_index=True)

# Reorder columns
all_metrics = all_metrics[['Model', 'Pollutant', 'RMSE', 'MAE', 'MAPE']]

# Save the consolidated CSV
consolidated_path = r"D:\env_monitoring_platform\backend\ml\air_forecasting_metrics.csv"
all_metrics.to_csv(consolidated_path, index=False)

print(f" Consolidated evaluation metrics saved to {consolidated_path}")
