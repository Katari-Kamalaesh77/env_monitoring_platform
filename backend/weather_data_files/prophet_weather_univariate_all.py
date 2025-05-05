import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# Load cleaned weather data
df = pd.read_csv("D:/env_monitoring_platform/backend/weather_data_files/cleaned_weather_data.csv", parse_dates=["DATE"])

# Dictionary of variables to forecast
forecast_targets = {
    "Max_Temperature": "tmax",
    "Min_Temperature": "tmin",
    "Average_Temperature": "tavg"
}

# Output folder structure
base_plot_dir = "D:/env_monitoring_platform/backend/plots/weather/prophet/"
forecast_csv_dir = os.path.join(base_plot_dir, "forecast_csv")
forecast_plot_dir = os.path.join(base_plot_dir, "forecast_plots")
os.makedirs(forecast_csv_dir, exist_ok=True)
os.makedirs(forecast_plot_dir, exist_ok=True)

# Path for metrics CSV
metrics_file = "D:/env_monitoring_platform/backend/weather_data_files/weather_model_prophet_metrics.csv"
all_metrics = []

for column, tag in forecast_targets.items():
    print(f"\n Forecasting {column} using Prophet...")

    # Filter and prepare data
    df_temp = df[["DATE", column]].dropna().rename(columns={"DATE": "ds", column: "y"})

    # Fit Prophet model
    model = Prophet()
    model.fit(df_temp)

    # Create future dataframe (365 days ahead)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    # Save forecast to CSV
    forecast_output_file = os.path.join(forecast_csv_dir, f"{tag}_prophet_forecast.csv")
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(forecast_output_file, index=False)

    # Plot forecast and save image
    fig = model.plot(forecast)
    plt.title(f"Forecast for {column}")
    plot_file = os.path.join(forecast_plot_dir, f"{tag}_prophet_forecast_plot.png")
    fig.savefig(plot_file)
    plt.close()

    # Evaluate model
    df_merged = pd.merge(df_temp, forecast[["ds", "yhat"]], on="ds", how="inner")
    rmse = mean_squared_error(df_merged["y"], df_merged["yhat"], squared=False)
    mae = mean_absolute_error(df_merged["y"], df_merged["yhat"])
    mape = (abs((df_merged["y"] - df_merged["yhat"]) / df_merged["y"])).mean() * 100

    all_metrics.append({
        "Variable": column,
        "Model": "Prophet",
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MAPE": round(mape, 2)
    })

    print(f" Forecast and plot for {column} saved as:\n - {forecast_output_file}\n - {plot_file}")
    print(f" RMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.2f}%")

# Save all metrics
metrics_df = pd.DataFrame(all_metrics)
if os.path.exists(metrics_file):
    existing = pd.read_csv(metrics_file)
    updated = pd.concat([existing, metrics_df], ignore_index=True)
else:
    updated = metrics_df
updated.to_csv(metrics_file, index=False)

print(f"\n All evaluation metrics saved to: {metrics_file}")
