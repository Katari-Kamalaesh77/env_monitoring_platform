# weather_analysis.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set the style for seaborn
sns.set(style="whitegrid")

# Define the path to the combined weather data CSV
data_dir = r'D:\env_monitoring_platform\backend\weather_data_files'
data_file = os.path.join(data_dir, 'combined_weather_data.csv')

# Load the dataset
try:
    df = pd.read_csv(data_file, low_memory=False)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"File not found: {data_file}")
    exit()
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    exit()

# Display the first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Display missing values in each column
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Display statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Rename columns for clarity if they exist
rename_dict = {
    'TMAX': 'Max_Temperature',
    'TMIN': 'Min_Temperature',
    'PRCP': 'Precipitation',
    'SNOW': 'Snowfall',
    'SNWD': 'Snow_Depth'
}

for old_col, new_col in rename_dict.items():
    if old_col in df.columns:
        df.rename(columns={old_col: new_col}, inplace=True)
    else:
        print(f"Column '{old_col}' not found in the dataset.")

# Convert units
# Temperature from tenths of degrees Celsius to degrees Celsius
for temp_col in ['Max_Temperature', 'Min_Temperature']:
    if temp_col in df.columns:
        df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce') / 10
    else:
        print(f"Column '{temp_col}' not found in the dataset.")

# Precipitation and Snowfall from tenths of mm to mm
for precip_col in ['Precipitation']:
    if precip_col in df.columns:
        df[precip_col] = pd.to_numeric(df[precip_col], errors='coerce') / 10
    else:
        print(f"Column '{precip_col}' not found in the dataset.")

# Snowfall and Snow Depth are already in mm, convert to numeric
for snow_col in ['Snowfall', 'Snow_Depth']:
    if snow_col in df.columns:
        df[snow_col] = pd.to_numeric(df[snow_col], errors='coerce')
    else:
        print(f"Column '{snow_col}' not found in the dataset.")

# Convert 'DATE' column to datetime
if 'DATE' in df.columns:
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
else:
    print("Column 'DATE' not found in the dataset.")

# Drop rows with missing 'DATE'
df.dropna(subset=['DATE'], inplace=True)

# Add 'Year' and 'Month' columns
df['Year'] = df['DATE'].dt.year
df['Month'] = df['DATE'].dt.month

# Calculate Average Temperature
if 'Max_Temperature' in df.columns and 'Min_Temperature' in df.columns:
    df['Average_Temperature'] = df[['Max_Temperature', 'Min_Temperature']].mean(axis=1)
else:
    print("Required columns for calculating average temperature not found.")

# Analysis 1: Annual Average Temperature
if 'Average_Temperature' in df.columns:
    annual_avg_temp = df.groupby('Year')['Average_Temperature'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=annual_avg_temp, x='Year', y='Average_Temperature', marker='o')
    plt.title('Annual Average Temperature')
    plt.xlabel('Year')
    plt.ylabel('Temperature (°C)')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'annual_average_temperature.png'))
    plt.close()
    print("Annual average temperature plot saved.")
else:
    print("Average_Temperature column not found for annual analysis.")

# Analysis 2: Annual Total Precipitation
if 'Precipitation' in df.columns:
    annual_total_precip = df.groupby('Year')['Precipitation'].sum().reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=annual_total_precip, x='Year', y='Precipitation', palette='Blues_d')
    plt.title('Annual Total Precipitation')
    plt.xlabel('Year')
    plt.ylabel('Precipitation (mm)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'annual_total_precipitation.png'))
    plt.close()
    print("Annual total precipitation plot saved.")
else:
    print("Precipitation column not found for annual analysis.")

# Analysis 3: Annual Total Snowfall
if 'Snowfall' in df.columns:
    annual_total_snowfall = df.groupby('Year')['Snowfall'].sum().reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=annual_total_snowfall, x='Year', y='Snowfall', palette='coolwarm')
    plt.title('Annual Total Snowfall')
    plt.xlabel('Year')
    plt.ylabel('Snowfall (mm)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'annual_total_snowfall.png'))
    plt.close()
    print("Annual total snowfall plot saved.")
else:
    print("Snowfall column not found for annual analysis.")

# Analysis 4: Monthly Average Temperature (Heatmap)
if 'Average_Temperature' in df.columns:
    monthly_avg_temp = df.groupby(['Year', 'Month'])['Average_Temperature'].mean().unstack()
    plt.figure(figsize=(12, 8))
    sns.heatmap(monthly_avg_temp, cmap='coolwarm', annot=False, fmt=".1f")
    plt.title('Monthly Average Temperature Heatmap')
    plt.xlabel('Month')
    plt.ylabel('Year')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'monthly_avg_temp_heatmap.png'))
    plt.close()
    print("Monthly average temperature heatmap saved.")
else:
    print("Average_Temperature column not found for monthly analysis.")

# Save the cleaned and processed data
processed_file = os.path.join(data_dir, 'processed_weather_data.csv')
df.to_csv(processed_file, index=False)
print(f"Processed data saved to {processed_file}")
