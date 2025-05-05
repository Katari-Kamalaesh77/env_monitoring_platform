import pandas as pd
from prophet import Prophet
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def forecast_pm25(data):
    try:
        df = pd.DataFrame(data)

        # Filter PM2.5
        df = df[df["parameter"] == "PM2.5 - Local Conditions"]

        # Select date and PM2.5 value
        df = df[["date_local", "arithmetic_mean"]]
        df.columns = ["ds", "y"]

        df["ds"] = pd.to_datetime(df["ds"])
        df = df.dropna(subset=["y"])

        if len(df) < 2:
            return {"error": "Not enough PM2.5 data for forecasting."}

        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=7)
        forecast = model.predict(future)

        return forecast[["ds", "yhat"]].tail(7).to_dict(orient="records")

    except Exception as e:
        logger.exception("Exception in forecast_pm25")
        return {"error": str(e)}
