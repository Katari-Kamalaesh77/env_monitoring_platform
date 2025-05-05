import pandas as pd
import os

# File paths
DATA_DIR = r"D:\env_monitoring_platform\backend\epa_data_by_state"
OUTPUT_PATH = r"D:\env_monitoring_platform\backend\ml\combined_air_data.csv"

# Mapping of parameter codes to pollutant names
pollutants = {
    "88101_all.csv": "PM2.5",
    "81102_all.csv": "PM10",
    "42101_all.csv": "CO",
    "42602_all.csv": "NO2",
    "42401_all.csv": "SO2",
    "44201_all.csv": "O3"
}

# Combine all pollutant files
df_combined = None
for filename, pollutant in pollutants.items():
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        print(f"Missing file: {file_path}")
        continue

    df = pd.read_csv(file_path, usecols=["date_local", "arithmetic_mean"])
    df = df.rename(columns={"arithmetic_mean": pollutant})
    df["date_local"] = pd.to_datetime(df["date_local"])
    
    # Average across stations per date
    df = df.groupby("date_local")[pollutant].mean().reset_index()

    if df_combined is None:
        df_combined = df
    else:
        df_combined = pd.merge(df_combined, df, on="date_local", how="outer")

# Sort and save
df_combined = df_combined.sort_values("date_local")
df_combined.to_csv(OUTPUT_PATH, index=False)
print(f"Combined air data saved to: {OUTPUT_PATH}")
