import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.utils.pollutants import POLLUTANTS

# Paths
DATA_DIR = 'D:/env_monitoring_platform/backend/epa_data_by_state'
PLOT_DIR = 'D:/env_monitoring_platform/backend/plots/air/xgboost/'
METRICS_PATH = 'D:/env_monitoring_platform/backend/ml/xgboost_air_metrics.csv'

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

def load_data(pollutant_file):
    path = os.path.join(DATA_DIR, pollutant_file)
    df = pd.read_csv(path, parse_dates=['date_local'], low_memory=False)
    df = df.groupby('date_local')['arithmetic_mean'].mean().reset_index()
    df.set_index('date_local', inplace=True)
    df = df.resample('M').mean().dropna()
    print(f"Loaded {pollutant_file}: {len(df)} monthly records")
    return df

def create_features(series, n_lags=12):
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i - n_lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def run_xgboost(df, pollutant, n_lags=12, n_forecast=5):
    print(f"\nRunning XGBoost for {pollutant}...")

    try:
        series = df['arithmetic_mean'].values
        X, y = create_features(series, n_lags=n_lags)

        if len(X) <= n_forecast:
            raise ValueError("Not enough data after feature creation.")

        split = len(X) - n_forecast
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        model = XGBRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = round(np.sqrt(mean_squared_error(y_test, preds)), 2)
        mae = round(mean_absolute_error(y_test, preds), 2)
        mape = round(mean_absolute_percentage_error(y_test, preds) * 100, 2)

        # Plotting
        plt.figure(figsize=(10, 4))
        test_index = df.index[-n_forecast:]
        plt.plot(test_index, y_test, label='Observed', color='blue')
        plt.plot(test_index, preds, label='Forecast', color='orange')
        plt.title(f"{pollutant} - XGBoost Forecast")
        plt.legend()
        plt.grid(True)

        safe_pollutant = pollutant.replace('.', '').replace(' ', '_').lower()
        plot_path = os.path.join(PLOT_DIR, f'xgboost_{safe_pollutant}.png')
        plt.savefig(plot_path)
        plt.close()

        print(f"Saved plot: {plot_path}")

        return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

    except Exception as e:
        print(f"Error running XGBoost for {pollutant}: {e}")
        return {'RMSE': None, 'MAE': None, 'MAPE': None}

def run_xgboost_forecast(return_metrics=False):
    all_metrics = {}
    for pollutant, filename in POLLUTANTS.items():
        try:
            df = load_data(filename)
            metrics = run_xgboost(df, pollutant)
            all_metrics[pollutant] = metrics
        except Exception as e:
            print(f"Skipping {pollutant} due to error: {e}")
            all_metrics[pollutant] = {'RMSE': None, 'MAE': None, 'MAPE': None}

    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.to_csv(METRICS_PATH)
    print(f"\nSaved metrics to: {METRICS_PATH}")

    if return_metrics:
        return all_metrics

if __name__ == '__main__':
    run_xgboost_forecast()
