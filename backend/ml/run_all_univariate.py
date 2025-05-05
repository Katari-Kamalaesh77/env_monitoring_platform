from prophet_univariate import run_prophet_forecast
from sarima_univariate import run_sarima_forecast
from xgboost_univariate import run_xgboost_forecast

def main():
    print("\n--- Running Prophet Forecasts ---")
    prophet_metrics = run_prophet_forecast()  # Return metrics to log them if needed
    print(f"Prophet Metrics: {prophet_metrics}")

    print("\n--- Running SARIMA Forecasts ---")
    sarima_metrics = run_sarima_forecast()  # Return metrics to log them if needed
    print(f"SARIMA Metrics: {sarima_metrics}")

    print("\n--- Running XGBoost Forecasts ---")
    xgboost_metrics = run_xgboost_forecast()  # Return metrics to log them if needed
    print(f"XGBoost Metrics: {xgboost_metrics}")

    print("\nAll univariate forecasts completed successfully!")

if __name__ == "__main__":
    main()
