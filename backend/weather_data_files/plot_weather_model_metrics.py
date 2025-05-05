import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Load CSV ===
csv_path = r"D:/env_monitoring_platform/backend/weather_data_files/consolidated_weather_model_metrics_summary.csv"
df = pd.read_csv(csv_path)

# === Drop MAPE and Model columns (not needed for plotting)
df = df.drop(columns=["MAPE", "Model"], errors="ignore")

# === Rename R² to R2 if needed
df.columns = [col.replace("R²", "R2") for col in df.columns]

# === Convert to long format
df_long = pd.DataFrame()
for metric in ["RMSE", "MAE", "R2"]:
    # Extract matching columns
    cols = [col for col in df.columns if col.startswith(metric)]
    temp = df[["Pollutant"] + cols].copy()
    temp = temp.melt(id_vars="Pollutant", var_name="Model", value_name="Value")
    temp["Model"] = temp["Model"].str.replace(f"{metric}_", "")
    temp["Metric"] = metric
    df_long = pd.concat([df_long, temp], ignore_index=True)

# === Output directory
output_dir = r"D:/env_monitoring_platform/backend/plots/weather/bar_charts"
os.makedirs(output_dir, exist_ok=True)

# === Plotting
sns.set(style="whitegrid")
for metric in ["RMSE", "MAE", "R2"]:
    subset = df_long[df_long["Metric"] == metric]
    if subset.empty:
        print(f"No data for {metric}")
        continue

    plt.figure(figsize=(12, 6))
    sns.barplot(data=subset, x="Pollutant", y="Value", hue="Model", errorbar=None)
    plt.title(f"Weather Forecasting Comparison — {metric}")
    plt.xlabel("Variable")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"weather_forecasting_comparison_{metric.lower()}.png")
    plt.savefig(save_path)
    plt.close()
    print(f" Saved: {save_path}")
