import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.utils.pollutants import POLLUTANTS

# Paths
DATA_DIR = 'D:/env_monitoring_platform/backend/epa_data_by_state'
PLOT_DIR = 'D:/env_monitoring_platform/backend/plots/air/sarima/'

def load_data(pollutant_file):
    path = os.path.join(DATA_DIR, pollutant_file)
    df = pd.read_csv(path, parse_dates=['date_local'], low_memory=False)
    df = df.groupby('date_local')['arithmetic_mean'].mean().reset_index()
    df.set_index('date_local', inplace=True)
    df = df.resample('M').mean().dropna()
    return df

def run_sarima(df, pollutant):
    print(f"Running SARIMA for {pollutant}...")

    try:
        model = SARIMAX(df['arithmetic_mean'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        results = model.fit(disp=False)

        forecast = results.get_forecast(steps=5)
        pred = forecast.predicted_mean
        conf_int = forecast.conf_int()

        # Take last 5 actual values for comparison
        y_true = df['arithmetic_mean'][-5:]
        y_pred = pred[:len(y_true)]

        rmse = round(np.sqrt(mean_squared_error(y_true, y_pred)), 2)
        mae = round(mean_absolute_error(y_true, y_pred), 2)
        mape = round(mean_absolute_percentage_error(y_true, y_pred) * 100, 2)

        # Plotting
        plt.figure(figsize=(10, 4))
        df['arithmetic_mean'].plot(label='Observed', color='blue')
        pred.plot(label='Forecast', color='orange')
        plt.fill_between(pred.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
        plt.title(f"{pollutant} - SARIMA Forecast")
        plt.legend()
        plt.grid(True)

        os.makedirs(PLOT_DIR, exist_ok=True)
        safe_pollutant = pollutant.replace('.', '').replace(' ', '_').lower()
        plot_path = os.path.join(PLOT_DIR, f'sarima_{safe_pollutant}.png')
        plt.savefig(plot_path)
        plt.close()

        print(f"Saved plot: {plot_path}")

        return {'Pollutant': pollutant, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

    except Exception as e:
        print(f"Error running SARIMA for {pollutant}: {e}")
        return {'Pollutant': pollutant, 'RMSE': None, 'MAE': None, 'MAPE': None}

def run_sarima_forecast(return_metrics=False):
    results = []

    for pollutant, filename in POLLUTANTS.items():
        try:
            df = load_data(filename)
            metrics = run_sarima(df, pollutant)
            results.append(metrics)
        except Exception as e:
            print(f"Skipping {pollutant} due to error: {e}")
            results.append({'Pollutant': pollutant, 'RMSE': None, 'MAE': None, 'MAPE': None})

    metrics_df = pd.DataFrame(results)
    metrics_save_path = "D:/env_monitoring_platform/backend/ml/sarima_air_metrics.csv"
    metrics_df.to_csv(metrics_save_path, index=False)
    print(f"Metrics saved to: {metrics_save_path}")

    if return_metrics:
        return metrics_df

if __name__ == '__main__':
    run_sarima_forecast()
