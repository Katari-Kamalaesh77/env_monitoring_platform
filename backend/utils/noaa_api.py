import os
import requests
from dotenv import load_dotenv

load_dotenv()

NOAA_TOKEN = os.getenv("NOAA_API_TOKEN")

def get_weather_data(latitude=40.7128, longitude=-74.0060):
    headers = {
        "token": NOAA_TOKEN
    }
    url = f"https://api.weather.gov/points/{latitude},{longitude}"
    point_data = requests.get(url, headers=headers).json()
    
    forecast_url = point_data['properties']['forecastHourly']
    forecast_data = requests.get(forecast_url, headers=headers).json()
    
    # Get the first period’s temp/humidity
    first_hour = forecast_data['properties']['periods'][0]
    return {
        "temperature": first_hour['temperature'],
        "humidity": first_hour.get('relativeHumidity', {}).get('value', 'N/A')
    }
