import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging

# Load environment variables from the .env file
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Get API key from the .env file
API_KEY = os.getenv("AIRNOW_API_KEY")
api_key = os.getenv("EPA_API_KEY")

def get_air_quality_data(zip_code="10001"):
    try:
        # URL for the AirNow API's current air quality observation
        url = "https://www.airnowapi.org/aq/observation/zipCode/current"
        
        # Parameters for the API request
        params = {
            "format": "application/json",
            "API_KEY": API_KEY,
            "zipCode": zip_code,  # You can provide a zip code for a specific location
        }

        # Send the GET request to AirNow API
        response = requests.get(url, params=params)

        # Check if the response was successful
        if response.status_code == 200:
            # Parse the JSON response and return it
            return response.json()
        else:
            # Return an error if the API call fails
            print(f"Error fetching data: {response.status_code}, {response.text}")
            return []

    except Exception as e:
        # Handle any exceptions during the request
        print(f"Error fetching air quality data: {e}")
        return []

import logging
import requests

def get_historical_air_quality_data(start_date, end_date, state_code="36"):
    try:
        url = "https://aqs.epa.gov/data/api/dailyData/byState"
        params = {
            "email": "kamalaesh.katari03@gmail.com",
            "key": api_key,
            "param": "88101",  # PM2.5
            "bdate": start_date,  # Format: YYYYMMDD
            "edate": end_date,
            "state": state_code,
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            json_data = response.json()

            if "Data" in json_data and isinstance(json_data["Data"], list):
                return json_data["Data"]
            else:
                logging.warning("No valid 'Data' field in response.")
                return []  # Return empty list if key is missing or not a list
        else:
            logging.error(f"API error {response.status_code}: {response.text}")
            return []
    except Exception as e:
        logging.error(f"Exception fetching air quality data: {e}")
        return []
