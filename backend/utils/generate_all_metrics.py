import sys
import os
import pandas as pd

# ✅ Add full backend path
sys.path.append('D:/env_monitoring_platform/backend/')

from ml.prophet_univariate import run_prophet_forecast
from ml.sarima_univariate import run_sarima_forecast
from ml.xgboost_univariate import run_xgboost_forecast

# 🔹 Run each model
print("\nRunning Prophet forecasts...")
prophet_metrics = run_prophet_forecast()

print("\nRunning SARIMA forecasts...")
sarima_metrics = run_sarima_forecast()

print("\nRunning XGBoost forecasts...")
xgboost_metrics = run_xgboost_forecast()

# 🔹 Helper to format metrics
def format_metrics(metrics_dict, model_name):
    rows = []
    for variable, metrics in metrics_dict.items():
        rows.append({
            'Variable': variable,
            'Model': model_name,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'MAPE': metrics['MAPE']
        })
    return rows

# 🔹 Combine all into one summary DataFrame
all_metrics = (
    format_metrics(prophet_metrics, 'Prophet') +
    format_metrics(sarima_metrics, 'SARIMA') +
    format_metrics(xgboost_metrics, 'XGBoost')
)
summary_df = pd.DataFrame(all_metrics)

# 🔹 Save to disk
output_dir = 'D:/env_monitoring_platform/backend/ml/metrics_summary'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'forecasting_metrics_summary.csv')
summary_df.to_csv(output_path, index=False)

print("\n Forecasting metrics summary saved to:")
print(output_path)
