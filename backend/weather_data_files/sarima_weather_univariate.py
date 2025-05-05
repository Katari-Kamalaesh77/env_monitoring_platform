import pandas as pd
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Load the cleaned weather data
df = pd.read_csv("D:/env_monitoring_platform/backend/weather_data_files/cleaned_weather_data.csv", parse_dates=["DATE"], low_memory=False)

# Aggregate to daily average if multiple entries exist per day
daily_weather = df.groupby("DATE").agg({
    "Max_Temperature": "mean", 
    "Min_Temperature": "mean", 
    "Average_Temperature": "mean"
}).reset_index()

# Setup output folders
base_path = "D:/env_monitoring_platform/backend/plots/weather/sarima/"
combined_csv_path = "D:/env_monitoring_platform/backend/weather_data_files/"
plot_dir = os.path.join(base_path, "forecast_plots")
csv_dir = os.path.join(base_path, "forecast_csv")

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

# Initialize the dictionary to store metrics for all variables
metrics_dict = {
    "Variable": [],
    "Model": [],
    "RMSE": [],
    "MAE": []
}

# Loop over each weather variable to run SARIMA
for column in ["Max_Temperature", "Min_Temperature", "Average_Temperature"]:
    print(f"\nRunning SARIMA for: {column}")

    # Prepare the data
    data = daily_weather[["DATE", column]].dropna()
    data = data.rename(columns={column: "y"})
    data = data.set_index("DATE")

    # Define the SARIMA model (fixed parameters for now)
    sarima_model = SARIMAX(data["y"], 
                           order=(1, 0, 0), 
                           seasonal_order=(1, 0, 1, 12),
                           enforce_stationarity=False, 
                           enforce_invertibility=False)
    
    results = sarima_model.fit()
    print(results.summary())

    # Save predictions (in-sample)
    y_pred = results.fittedvalues
    result_df = data.copy()
    result_df["SARIMA_Predicted"] = y_pred
    result_csv_path = os.path.join(csv_dir, f"{column}_sarima_forecast.csv")
    result_df.to_csv(result_csv_path)

    # Evaluate
    rmse = np.sqrt(mean_squared_error(data["y"], y_pred))
    mae = mean_absolute_error(data["y"], y_pred)

    metrics_dict["Variable"].append(column)
    metrics_dict["Model"].append("SARIMA")
    metrics_dict["RMSE"].append(round(rmse, 4))
    metrics_dict["MAE"].append(round(mae, 4))

    # Save plot
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data["y"], label="Actual")
    plt.plot(data.index, y_pred, label="Fitted", color='red')
    plt.title(f"SARIMA Model for {column}")
    plt.xlabel("Date")
    plt.ylabel(f"{column}")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(plot_dir, f"{column}_sarima_plot.png")
    plt.savefig(plot_path)
    plt.close()

# Save metrics
metrics_df = pd.DataFrame(metrics_dict)
metrics_csv = os.path.join(combined_csv_path, "weather_model_sarima_metrics.csv")
metrics_df.to_csv(metrics_csv, index=False)

print(" SARIMA modeling, metrics, and plots saved under:")
print(f" - CSVs: {csv_dir}")
print(f" - Plots: {plot_dir}")
print(f" - Metrics: {metrics_csv}")
