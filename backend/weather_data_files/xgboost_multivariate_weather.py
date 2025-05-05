import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# === Load cleaned weather data ===
csv_path = "D:/env_monitoring_platform/backend/weather_data_files/cleaned_weather_data.csv"
df = pd.read_csv(csv_path, parse_dates=["DATE"], low_memory=False)

print("CSV loaded. Columns available:", df.columns.tolist())

# === Define targets and features ===
target_columns = ["Max_Temperature", "Min_Temperature", "Average_Temperature"]
feature_columns = [
    col for col in df.columns
    if col not in target_columns + ["DATE"] and df[col].dtype in [np.float64, np.int64]
]

# === Output directories ===
plot_dir = "D:/env_monitoring_platform/backend/plots/weather/xgboost"
os.makedirs(plot_dir, exist_ok=True)

# === Initialize metrics list ===
metrics_summary = []

# === Iterate over each target variable ===
for target in target_columns:
    print(f"\nProcessing target: {target}")

    if target not in df.columns:
        print(f"Column {target} not found. Skipping.")
        continue

    data = df[df[target].notnull()]
    print(f"Rows available for modeling: {len(data)}")

    if len(data) < 100:
        print("Not enough data. Skipping this target.")
        continue

    X = data[feature_columns].fillna(0)
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # === Model training ===
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # === Predictions and metrics ===
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"   RMSE: {rmse:.2f}")
    print(f"   MAE : {mae:.2f}")
    print(f"   R²  : {r2:.3f}")

    # === Save predictions CSV ===
    results_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
    results_df.to_csv(os.path.join(plot_dir, f"xgb_multivariate_predictions_{target}.csv"), index=False)

    # === Append metrics ===
    metrics_summary.append({
        "Variable": target,
        "Model": "XGBoost-Multivariate",
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R²": round(r2, 3)
    })

# === Save metrics to model-specific CSV ===
metrics_df = pd.DataFrame(metrics_summary)
metrics_path = os.path.join(plot_dir, "xgboost_weather_multivariate_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)
print(f"\nMultivariate XGBoost metrics saved to: {metrics_path}")

# === Append to global summary ===
summary_path = "D:/env_monitoring_platform/backend/weather_data_files/xgboost_weather_multivariate_metrics.csv"
if os.path.exists(summary_path):
    existing = pd.read_csv(summary_path)
    combined = pd.concat([existing, metrics_df], ignore_index=True)
else:
    combined = metrics_df
combined.to_csv(summary_path, index=False)
print(f"Updated summary saved to: {summary_path}")
