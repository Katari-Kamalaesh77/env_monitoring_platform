import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# === Load cleaned weather data ===
file_path = "D:/env_monitoring_platform/backend/weather_data_files/cleaned_weather_data.csv"
df = pd.read_csv(file_path, parse_dates=["DATE"])
df = df.sort_values("DATE")

# === Forecast settings ===
forecast_horizon = 30  # Days to predict
lags = 30  # Number of previous days to use as features

# === Directory for organized plots and metrics ===
plot_dir = "D:/env_monitoring_platform/backend/plots/weather/xgboost"
os.makedirs(plot_dir, exist_ok=True)

# === Initialize list to save metrics ===
metrics_list = []

# === Lag feature creation function ===
def create_lag_features(series, lags):
    df_lagged = pd.DataFrame()
    for i in range(1, lags + 1):
        df_lagged[f"lag_{i}"] = series.shift(i)
    df_lagged["y"] = series.values
    df_lagged = df_lagged.dropna()
    return df_lagged

# === Training and forecasting function ===
def train_and_forecast(df, column, forecast_horizon, lags):
    print(f"\nTraining XGBoost model for: {column}")

    # Create lag features
    lagged_df = create_lag_features(df[column], lags)
    
    # Train-test split
    X = lagged_df.drop("y", axis=1)
    y = lagged_df["y"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=forecast_horizon, shuffle=False)
    
    # Model
    model = XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Save metrics to list
    metrics_list.append([column, mae, mse, r2])

    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2:.4f}")

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(df["DATE"].iloc[-forecast_horizon:], y_test, label="Actual", marker="o")
    plt.plot(df["DATE"].iloc[-forecast_horizon:], y_pred, label="Predicted", marker="x")
    plt.title(f"XGBoost Forecast - {column}")
    plt.xlabel("Date")
    plt.ylabel(column)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{column}_forecast.png")
    plt.close()

# === Run for all 3 weather variables ===
for col in ["Max_Temperature", "Min_Temperature", "Average_Temperature"]:
    if col in df.columns:
        train_and_forecast(df, col, forecast_horizon, lags)
    else:
        print(f"Column '{col}' not found in dataset!")

# === Save individual model metrics ===
metrics_df = pd.DataFrame(metrics_list, columns=["Variable", "MAE", "MSE", "R²"])
metrics_df["Model"] = "XGBoost"
metrics_df["RMSE"] = np.sqrt(metrics_df["MSE"])
metrics_df = metrics_df[["Variable", "Model", "RMSE", "MAE", "R²"]]  # Reorder for consistency

# Save to XGBoost-specific file
metrics_df.to_csv(f"{plot_dir}/xgboost_weather_metrics.csv", index=False)

# (Optional) Append to summary file
summary_path = "D:/env_monitoring_platform/backend/weather_data_files/weather_model_xgboost_metrics.csv"
if os.path.exists(summary_path):
    existing = pd.read_csv(summary_path)
    combined = pd.concat([existing, metrics_df], ignore_index=True)
else:
    combined = metrics_df

combined.to_csv(summary_path, index=False)

print("XGBoost univariate forecasting complete.")
