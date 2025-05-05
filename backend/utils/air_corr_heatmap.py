import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Paths
DATA_PATH = r"D:\env_monitoring_platform\backend\ml\combined_air_data.csv"
PLOT_PATH = 'D:/env_monitoring_platform/backend/plots/air/air_corr_heatmap.png'

# Create output directory
os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH, parse_dates=['date_local'])

# Resample to monthly averages
df.set_index('date_local', inplace=True)
monthly_df = df.resample('M').mean()

# Compute correlation
corr_matrix = monthly_df.corr()

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar_kws={"shrink": .8})
plt.title("Air Quality Pollutants - Correlation Heatmap", fontsize=14)
plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.close()

print(f"Air quality heatmap saved to: {PLOT_PATH}")
