import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Load the cleaned weather data
file_path = r'D:\env_monitoring_platform\backend\weather_data_files\cleaned_weather_data.csv'
df = pd.read_csv(file_path)

# ------------------- Cleanup & Normalize -------------------
# Normalize column names
df.columns = df.columns.str.strip()

# Optional: Rename for clarity
df.rename(columns={
    'Precipitation': 'PRCP_mm',
    'Snowfall': 'SNOW_mm',
    'Max_Temperature': 'TMAX_C',
    'Min_Temperature': 'TMIN_C',
    'Average_Temperature': 'TAVG_C',
    'Wind_Speed': 'AWND_mps'  # if exists
}, inplace=True)

# Convert DATE to datetime and sort
df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
df = df.sort_values('DATE')

# ------------------- Basic Info -------------------
print("Dataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# ------------------- Histograms -------------------
def plot_histograms(dataframe):
    num_cols = dataframe.select_dtypes(include=['float64', 'int64']).columns
    dataframe[num_cols].hist(bins=50, figsize=(12, 10))
    plt.suptitle('Histograms of Numerical Features')
    plt.tight_layout()
    plt.show()

plot_histograms(df)

# ------------------- Boxplots -------------------
def plot_boxplots(dataframe):
    num_cols = dataframe.select_dtypes(include=['float64', 'int64']).columns
    num_plots = len(num_cols)
    cols = 4
    rows = math.ceil(num_plots / cols)

    plt.figure(figsize=(cols * 5, rows * 4))
    for i, col in enumerate(num_cols, 1):
        plt.subplot(rows, cols, i)
        sns.boxplot(x=dataframe[col])
        plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()

selected_cols = ['PRCP_mm', 'SNOW_mm', 'TMAX_C', 'TMIN_C', 'TAVG_C']
plot_boxplots(df[selected_cols])

# ------------------- Time Series with Smoothing -------------------
def plot_timeseries_with_smoothing(dataframe, date_column, target_column, window=30):
    plt.figure(figsize=(12, 6))
    plt.plot(dataframe[date_column], dataframe[target_column], alpha=0.4, label='Daily')
    plt.plot(dataframe[date_column], dataframe[target_column].rolling(window).mean(), color='red', label=f'{window}-Day Rolling Avg')
    plt.title(f'{target_column} Over Time with Rolling Avg')
    plt.xlabel('Date')
    plt.ylabel(target_column)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_timeseries_with_smoothing(df, 'DATE', 'TMAX_C')
plot_timeseries_with_smoothing(df, 'DATE', 'TMIN_C')
plot_timeseries_with_smoothing(df, 'DATE', 'TAVG_C')
plot_timeseries_with_smoothing(df, 'DATE', 'PRCP_mm')

# ------------------- Correlation Heatmap -------------------
def plot_correlation_heatmap(dataframe):
    corr = dataframe.select_dtypes(include=['float64', 'int64']).corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

plot_correlation_heatmap(df)

# ------------------- Scatter Plot -------------------
def plot_temperature_precipitation(dataframe):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=dataframe['TMAX_C'], y=dataframe['PRCP_mm'])
    plt.title('Max Temperature vs Precipitation')
    plt.xlabel('Max Temperature (°C)')
    plt.ylabel('Precipitation (mm)')
    plt.tight_layout()
    plt.show()

plot_temperature_precipitation(df)

# ------------------- Monthly Averages -------------------
df['MONTH'] = df['DATE'].dt.month
monthly_avg = df.groupby('MONTH')[['TAVG_C', 'PRCP_mm', 'SNOW_mm']].mean()

monthly_avg.plot(kind='line', marker='o', figsize=(10, 6))
plt.title('Monthly Averages: Temperature, Precipitation & Snowfall')
plt.xlabel('Month')
plt.ylabel('Average Value')
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------- Extreme Weather Events -------------------
print("\nTop 5 Hottest Days:")
print(df[['DATE', 'TMAX_C']].sort_values(by='TMAX_C', ascending=False).head())

print("\nTop 5 Coldest Days:")
print(df[['DATE', 'TMIN_C']].sort_values(by='TMIN_C').head())

print("\nTop 5 Rainiest Days:")
print(df[['DATE', 'PRCP_mm']].sort_values(by='PRCP_mm', ascending=False).head())
