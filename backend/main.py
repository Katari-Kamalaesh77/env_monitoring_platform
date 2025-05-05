from fastapi import FastAPI
from fastapi import APIRouter
from fastapi.middleware.cors import CORSMiddleware
from utils.airnow_api import get_air_quality_data
from utils.airnow_api import get_historical_air_quality_data
from utils.noaa_api import get_weather_data
from ml.forecast import forecast_pm25
from ml.data_preparation import fetch_pm25_data, clean_pm25_data
from ml.data_exploration import explore_data
import pandas as pd
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

router = APIRouter()

# CORS Middleware
origins = [
    "http://localhost:3000",  # React front-end
    "http://127.0.0.1:3000",  # React front-end
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows the frontend URLs to make requests
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

@app.get("/api/airquality")
def get_air_quality():
    try:
        # Get raw air quality data from AirNow API
        raw_data = get_air_quality_data()

        if not raw_data:
            return {"message": "No data available. Try again later."}

        # Extract PM2.5 and O3 data
        pm25_data = next((item for item in raw_data if item["ParameterName"] == "PM2.5"), None)
        o3_data = next((item for item in raw_data if item["ParameterName"] == "O3"), None)

        # Get weather data (temperature, humidity) from NOAA API
        weather_data = get_weather_data()

        # Logging the data for debugging
        # logging.debug(f"Air quality data: {raw_data}")
        # logging.debug(f"Weather data: {weather_data}")

        # Returning the data
        return {
            "pm25": pm25_data["AQI"] if pm25_data else None,
            "pm25_category": pm25_data["Category"]["Name"] if pm25_data else "Unavailable",
            "o3": o3_data["AQI"] if o3_data else None,
            "o3_category": o3_data["Category"]["Name"] if o3_data else "Unavailable",
            "temperature": weather_data["temperature"] if weather_data else "Unavailable",
            "humidity": weather_data["humidity"] if weather_data else "Unavailable"
        }
    except Exception as e:
        # logging.error(f"Error fetching air quality data: {str(e)}")
        return {"error": "Failed to fetch air quality data"}

@app.get("/api/forecast/pm25")
def get_pm25_forecast():
    try:
        # Define the date range for historical data
        start_date = "20230101"  # You can dynamically generate or accept from query params
        end_date = "20231231"    # You can dynamically generate or accept from query params

        # Get air quality data (historical data)
        airnow_data = get_historical_air_quality_data(start_date, end_date)

        # Log the fetched data to debug
        # logging.debug(f"Fetched air quality data: {airnow_data}")

        # Check if no data was returned
        if not airnow_data:
            return {"error": "No air quality data available"}

        # Add timestamp field to air quality data if necessary
        for item in airnow_data:
            if "DateObserved" in item:
                item["timestamp"] = item["DateObserved"]  # Add timestamp field

        # Debug: log the raw data being passed for forecasting
        # logging.debug(f"Air quality data passed to forecast: {airnow_data}")

        # Get PM2.5 forecast from machine learning model
        forecast = forecast_pm25(airnow_data)

        # Check if the forecast is valid, otherwise return an error
        if forecast is None:
            return {"error": "Failed to generate PM2.5 forecast"}

        # Debug: log the forecast results
        # logging.debug(f"Forecast PM2.5: {forecast}")

        # Return the forecasted PM2.5 data
        return {"forecast": forecast}

    except Exception as e:
        # Log any errors that occurred during the forecasting process
        logging.error(f"Error forecasting PM2.5: {str(e)}")
        return {"error": "Failed to generate PM2.5 forecast"}

@router.get("/api/test/pm25-data")
def test_pm25_data():
    # Fetch last 90 days of PM2.5 data
    raw_df = fetch_pm25_data("20240101", "20240401")
    cleaned_df = clean_pm25_data(raw_df)

    # Show first 5 records
    return cleaned_df.head(5).to_dict(orient="records")


app.include_router(router)


#data = [
#    {"ds": "2024-01-01", "y": 5.91},
#    {"ds": "2024-01-02", "y": 5.22},
#    {"ds": "2024-01-03", "y": 7.27},
#    {"ds": "2024-01-04", "y": 5.00},
#    {"ds": "2024-01-05", "y": 5.91},
#]

#df = pd.DataFrame(data)
#df["ds"] = pd.to_datetime(df["ds"])

#explore_data(df)

