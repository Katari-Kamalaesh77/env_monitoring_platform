import pandas as pd
import numpy as np
from datetime import datetime

# Load combined weather data
combined_file = "processed_weather_data.csv"  # Assumes raw data is already combined here
df = pd.read_csv(combined_file, parse_dates=["DATE"], low_memory=False)

# Drop rows with missing date
df = df.dropna(subset=["DATE"])

# Basic cleaning - Fill missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Feature Engineering
# Temperature Range
if "TMAX_C" in df.columns and "TMIN_C" in df.columns:
    df["TEMP_RANGE_C"] = df["TMAX_C"] - df["TMIN_C"]

# Precipitation indicator
if "PRCP_mm" in df.columns:
    df["RAIN_INDICATOR"] = (df["PRCP_mm"] > 0).astype(int)

# Rolling Averages (7-day window)
rolling_features = ["TMAX_C", "TMIN_C", "TAVG_C", "PRCP_mm"]
for col in rolling_features:
    if col in df.columns:
        df[f"{col}_ROLL7"] = df[col].rolling(window=7, min_periods=1).mean()

# Tag season
def get_season(date):
    Y = 2000  # dummy leap year to handle Feb 29
    seasons = [(datetime(Y, 1, 1), datetime(Y, 3, 20), 'Winter'),
               (datetime(Y, 3, 21), datetime(Y, 6, 20), 'Spring'),
               (datetime(Y, 6, 21), datetime(Y, 9, 22), 'Summer'),
               (datetime(Y, 9, 23), datetime(Y, 12, 20), 'Fall'),
               (datetime(Y, 12, 21), datetime(Y, 12, 31), 'Winter')]
    date = date.replace(year=Y)
    return next(season for start, end, season in seasons if start <= date <= end)

df["SEASON"] = df["DATE"].apply(lambda x: get_season(x))

# Tag weekday/weekend
df["WEEKDAY"] = df["DATE"].dt.weekday
df["IS_WEEKEND"] = df["WEEKDAY"].isin([5, 6]).astype(int)

# Save processed file
df.to_csv("processed_weather_data_v2.csv", index=False)
print("Cleaned and enhanced weather data saved to processed_weather_data_v2.csv")
