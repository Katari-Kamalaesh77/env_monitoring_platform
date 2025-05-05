import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Fix: Add root of the project to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.utils.pollutants import POLLUTANTS  # ✅ Now this will work


# === Paths ===
DATA_DIR = 'D:/env_monitoring_platform/backend/epa_data_by_state'
PLOT_DIR = 'D:/env_monitoring_platform/backend/plots/air/prophet'
METRIC_SAVE_PATH = 'D:/env_monitoring_platform/backend/ml/prophet_air_metrics.csv'

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(METRIC_SAVE_PATH), exist_ok=True)

def load_data(pollutant_file):
    """Load and preprocess air quality data for Prophet."""
    path = os.path.join(DATA_DIR, pollutant_file)
    df = pd.read_csv(path, parse_dates=['date_local'], low_memory=False)
    df = df.groupby('date_local')['arithmetic_mean'].mean().reset_index()
    df.columns = ['ds', 'y']  # Prophet expects these column names
    return df

def run_prophet(df, pollutant):
    """Train Prophet, forecast, plot, and return metrics."""
    model = Prophet()
    model.fit(df)

    # Forecast 12 months ahead
    future = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(future)

    # Save plot
    plt.figure()
    model.plot(forecast)
    plt.title(f'{pollutant} Forecast with Prophet')
    plot_path = os.path.join(PLOT_DIR, f'prophet_{pollutant}_forecast.png')
    plt.savefig(plot_path)
    plt.close()
    print(f" Saved forecast plot: {plot_path}")

    # Evaluation metrics on last 5 known data points
    y_true = df['y'].tail(5)
    y_pred = forecast['yhat'].iloc[-17:-12]  # Last 5 actual points (excluding future ones)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    return {
        'Pollutant': pollutant,
        'RMSE': round(rmse, 2),
        'MAE': round(mae, 2),
        'MAPE': round(mape * 100, 2)
    }

def run_prophet_forecast(return_metrics=False):
    """Run Prophet forecast for all pollutants and optionally save metrics."""
    all_metrics = []

    for pollutant, filename in POLLUTANTS.items():
        print(f' Running Prophet forecast for: {pollutant}')
        df = load_data(filename)
        metrics = run_prophet(df, pollutant)
        all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(METRIC_SAVE_PATH, index=False)
    print(f" Metrics saved to: {METRIC_SAVE_PATH}")

    return metrics_df if return_metrics else None

if __name__ == '__main__':
    run_prophet_forecast(return_metrics=False)
