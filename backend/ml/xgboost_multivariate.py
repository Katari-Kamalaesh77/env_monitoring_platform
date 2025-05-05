import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# ---- Config ----
POLLUTANT_CODES = {
    'PM2.5': '88101',
    'PM10': '81102',
    'CO': '42101',
    'SO2': '42401',
    'NO2': '42602',
    'O3': '44201'
}
DATA_DIR = "../epa_data_by_state"
PLOT_DIR = "plots/xgboost_multivariate"
METRICS_CSV = "xgb_multivariate_metrics.csv"
LAGS = 3
ALL_POLLUTANTS = list(POLLUTANT_CODES.keys())

os.makedirs(PLOT_DIR, exist_ok=True)


# ---- Load and Merge All Pollutants into One DataFrame ----
def load_and_merge_pollutants():
    dfs = []
    for pollutant, code in POLLUTANT_CODES.items():
        file_path = os.path.join(DATA_DIR, f"{code}_all.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, low_memory=False)
            df['date_local'] = pd.to_datetime(df['date_local'])
            daily_df = df.groupby('date_local')['arithmetic_mean'].mean().reset_index()
            daily_df = daily_df.rename(columns={'arithmetic_mean': pollutant})
            dfs.append(daily_df)
        else:
            print(f"File not found for {pollutant}: {file_path}")

    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='date_local', how='outer')

    merged_df = merged_df.sort_values('date_local')
    merged_df = merged_df.fillna(method='ffill').fillna(method='bfill').dropna()
    return merged_df


# ---- Create Lag Features ----
def create_lag_features(df, lags):
    for col in ALL_POLLUTANTS:
        for lag in range(1, lags + 1):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df.dropna().reset_index(drop=True)


# ---- Train and Evaluate Model ----
def train_xgboost_model(X_train, y_train):
    model = XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        objective="reg:squarederror", random_state=42
    )
    model.fit(X_train, y_train)
    return model


# ---- Evaluate Predictions ----
def evaluate_forecast(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return round(rmse, 4), round(mae, 4), round(mape, 2)


# ---- Plot Forecast ----
def plot_forecast(dates_test, y_test, y_pred, target):
    plt.figure(figsize=(12, 6))
    plt.plot(dates_test, y_test, label="Actual", color="blue")
    plt.plot(dates_test, y_pred, label="Predicted", color="orange")
    plt.title(f"XGBoost Multivariate Forecast — {target}")
    plt.xlabel("Date")
    plt.ylabel(f"{target} Value")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fname = f"xgb_multivariate_{target.replace('.', '').lower()}.png"
    plt.savefig(os.path.join(PLOT_DIR, fname))
    plt.close()


# ---- Main Forecasting Function ----
def run_multivariate_forecasting():
    merged_df = load_and_merge_pollutants()
    print(f"Merged data shape: {merged_df.shape}")
    metrics_summary = []

    for target in ALL_POLLUTANTS:
        print(f"\n--- Forecasting {target} ---")
        df = merged_df.copy()
        df = create_lag_features(df, LAGS)
        dates = df['date_local']

        feature_cols = [col for col in df.columns if "lag" in col]
        X = df[feature_cols]
        y = df[target]

        X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
            X, y, dates, test_size=0.2, shuffle=False
        )

        model = train_xgboost_model(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse, mae, mape = evaluate_forecast(y_test, y_pred)
        metrics_summary.append({'Pollutant': target, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape})

        plot_forecast(dates_test, y_test.values, y_pred, target)

    metrics_df = pd.DataFrame(metrics_summary)
    metrics_df.to_csv(METRICS_CSV, index=False)
    print("\nSaved metrics to:", METRICS_CSV)
    print(metrics_df)


# ---- Run Forecasts ----
if __name__ == "__main__":
    run_multivariate_forecasting()
