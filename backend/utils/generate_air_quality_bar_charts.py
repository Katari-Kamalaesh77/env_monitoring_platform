import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load metrics CSV for air quality
csv_path = 'forecast_metrics_summary.csv'  # Update this path if needed
df = pd.read_csv(csv_path)

# Ensure output folder exists
output_dir = '../plots/air/air_quality/'
os.makedirs(output_dir, exist_ok=True)

# Metrics to plot
metrics = ['RMSE', 'MAE', 'MAPE']

for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Pollutant', y=metric, hue='Model', data=df, errorbar=None)
    plt.title(f'{metric} Comparison for Air Quality Models')
    plt.xlabel('Pollutant')
    plt.ylabel(metric)
    plt.legend(title='Model')
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(output_dir, f'air_quality_{metric.lower()}_comparison.png')
    plt.savefig(save_path)
    plt.close()

print(" Air quality metric bar charts generated and saved successfully.")
