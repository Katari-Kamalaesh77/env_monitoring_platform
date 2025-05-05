import pandas as pd

# Load metrics
uni_df = pd.read_csv(r"D:\env_monitoring_platform\backend\ml\air_forecasting_metrics.csv")   # from univariate models
multi_df = pd.read_csv(r"D:\env_monitoring_platform\backend\ml\xgb_multivariate_metrics.csv")   # from multivariate model

# Add 'Model' column
# Make sure univariate CSV already includes model names: 'Prophet', 'SARIMA', 'XGBoost'
multi_df['Model'] = 'XGBoost-Multivariate'

# Merge
combined_df = pd.concat([uni_df, multi_df], ignore_index=True)

# Save to CSV
combined_df.to_csv("air_quality_combined_metrics.csv", index=False)
print("Saved combined metrics to 'air_quality_combined_metrics.csv'")
