import pandas as pd
import os

# Choose the correct processed file
input_file = r"D:\env_monitoring_platform\backend\weather_data_files\processed_weather_data.csv"
output_file = r"D:\env_monitoring_platform\backend\weather_data_files\cleaned_weather_data.csv"

# Load the processed dataset
print(f"Reading from: {input_file}")
df = pd.read_csv(input_file, low_memory=False)

# Convert 'DATE' to datetime format
if 'DATE' in df.columns:
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df = df.dropna(subset=['DATE'])

# Drop rows with missing or invalid average temperature
if 'Average_Temperature' in df.columns:
    df = df.dropna(subset=['Average_Temperature'])

# Drop unnecessary columns if needed (example: all columns ending with "_ATTRIBUTES")
df = df.loc[:, ~df.columns.str.endswith('_ATTRIBUTES')]

# Optional: Filter by year range (e.g., keep only data from 2000 to 2024)
df['Year'] = df['DATE'].dt.year
df = df[(df['Year'] >= 2000) & (df['Year'] <= 2024)]

# Save the cleaned dataset
df.to_csv(output_file, index=False)
print(f"Cleaned data saved to: {output_file}")
