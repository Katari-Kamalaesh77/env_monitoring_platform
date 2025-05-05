import pandas as pd

# Load individual model metrics
prophet_metrics = pd.read_csv(r"D:\env_monitoring_platform\backend\weather_data_files\weather_model_prophet_metrics.csv")
sarima_metrics = pd.read_csv(r"D:\env_monitoring_platform\backend\weather_data_files\weather_model_sarima_metrics.csv")
xgboost_metrics = pd.read_csv(r"D:\env_monitoring_platform\backend\weather_data_files\weather_model_xgboost_metrics.csv")
xgb_multi_metrics = pd.read_csv(r"D:\env_monitoring_platform\backend\weather_data_files\xgboost_weather_multivariate_metrics.csv")

# Standardize and deduplicate
prophet_metrics.rename(columns={"Variable": "Pollutant"}, inplace=True)
sarima_metrics.rename(columns={"Variable": "Pollutant"}, inplace=True)
xgboost_metrics.rename(columns={"Variable": "Pollutant"}, inplace=True)
xgb_multi_metrics.rename(columns={"Variable": "Pollutant"}, inplace=True)

# Drop duplicate entries
prophet_metrics.drop_duplicates(subset=["Pollutant"], inplace=True)
sarima_metrics.drop_duplicates(subset=["Pollutant"], inplace=True)
xgboost_metrics.drop_duplicates(subset=["Pollutant"], inplace=True)
xgb_multi_metrics.drop_duplicates(subset=["Pollutant"], inplace=True)

# Merge Prophet and SARIMA
merged_ps = pd.merge(prophet_metrics, sarima_metrics, on="Pollutant", suffixes=('_Prophet', '_SARIMA'))

# Merge with XGBoost univariate
merged_all = pd.merge(merged_ps, xgboost_metrics, on="Pollutant")
merged_all.rename(columns={
    "RMSE": "RMSE_XGBoost",
    "MAE": "MAE_XGBoost",
    "R²": "R2_XGBoost"
}, inplace=True)
merged_all.drop(columns=["Model_Prophet", "Model_SARIMA", "Model"], errors='ignore', inplace=True)

# Merge with XGBoost multivariate
final_all = pd.merge(merged_all, xgb_multi_metrics, on="Pollutant", suffixes=('', '_XGBMulti'))
final_all.rename(columns={
    "RMSE": "RMSE_XGBMulti",
    "MAE": "MAE_XGBMulti",
    "R²": "R2_XGBMulti"
}, inplace=True)

# Save final summary
output_path = r"D:\env_monitoring_platform\backend\weather_data_files\consolidated_weather_model_metrics_summary.csv"
final_all.to_csv(output_path, index=False)
print(f" FINAL consolidated summary saved to: {output_path}")
