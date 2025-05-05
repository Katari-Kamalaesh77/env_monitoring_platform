import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Paths
DATA_PATH = 'D:/env_monitoring_platform/backend/weather_data_files/processed_weather_data.csv'
PLOT_PATH = 'D:/env_monitoring_platform/backend/plots/weather/weather_corr_heatmap.png'

# Create output directory
os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH, parse_dates=['date'])

# Drop non-numeric columns
non_numeric = ['station', 'name', 'date']
df_numeric = df.drop(columns=[col for col in non_numeric if col in df.columns])

# Compute correlation
corr_matrix = df_numeric.corr()

# Plot
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu', fmt=".2f", square=True, cbar_kws={"shrink": .8})
plt.title("Weather Variables - Correlation Heatmap", fontsize=14)
plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.close()

print(f"Weather heatmap saved to: {PLOT_PATH}")
