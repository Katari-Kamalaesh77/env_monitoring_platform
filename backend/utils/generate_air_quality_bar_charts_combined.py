import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load combined metrics
df = pd.read_csv("air_quality_combined_metrics.csv")

# Output dir
output_dir = "../plots/air/bar_charts"
os.makedirs(output_dir, exist_ok=True)

# Seaborn style
sns.set(style="whitegrid")

# Plotting function
def plot_bar(metric):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Pollutant', y=metric, hue='Model', data=df, errorbar=None)
    plt.title(f'Air Quality Forecasting Comparison — {metric}')
    plt.xlabel("Pollutant")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"air_quality_comparison_{metric.lower()}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

# Plot all 3 metrics
for metric in ['RMSE', 'MAE', 'MAPE']:
    plot_bar(metric)
