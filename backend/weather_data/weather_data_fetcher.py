import requests
import pandas as pd
from datetime import datetime

# NOAA API token (replace with your actual token)
api_token = 'CfRzdxtXoKjMyJZWBApmtTocJgzIJcyj'

# Base URL for NOAA CDO API
base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/"

# Latitude and Longitude of Central NYC (near Times Square, NYC)
lat = 40.7128
lon = -74.0060

# Radius around Central NYC (in miles)
radius = 50

# Define headers with API token
headers = {
    'token': api_token
}

# Function to fetch station IDs within a radius of NYC
def get_stations(lat, lon, radius=50):
    url = base_url + 'stations'
    params = {
        'latitude': lat,
        'longitude': lon,
        'radius': radius,
        'limit': 50  # Adjust if needed
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json().get('results', [])
    else:
        print("Error fetching station data:", response.status_code)
        print(response.text)
        return []

# Function to fetch weather data for a specific station
def get_weather_data(station_id, start_date="1995-01-01", end_date="2024-12-31"):
    url = base_url + 'data'
    params = {
        'datasetid': 'GHCND',   # <-- this is the missing required parameter!
        'stationid': station_id,
        'startdate': start_date,
        'enddate': end_date,
        'datatypeid': ['TMAX', 'TMIN', 'AWND', 'PRCP'],
        'limit': 1000,
        'units': 'metric'
    }
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        if response.text.strip():
            try:
                return response.json().get('results', [])
            except Exception as e:
                print(f"JSON decoding failed for station {station_id}: {e}")
                print(response.text)
                return []
        else:
            print(f"Empty response for station {station_id}")
            return []
    else:
        print(f"Error fetching weather data for station {station_id}: {response.status_code}")
        print(response.text)
        return []

# Function to process and aggregate data from multiple stations
def aggregate_weather_data(stations):
    all_data = []

    for station in stations:
        station_id = station['id']
        station_name = station.get('name', 'Unknown')
        print(f"Fetching data for station {station_name} ({station_id})")
        
        weather_data = get_weather_data(station_id)

        if not weather_data:
            print(f"No data returned for station {station_name} ({station_id})")
            continue

        for entry in weather_data:
            record = {
                'date': entry['date'],
                'station_name': station_name,
                'tmax': None,
                'tmin': None,
                'awnd': None,
                'prcp': None
            }

            if entry['datatype'] == 'TMAX':
                record['tmax'] = entry['value']
            elif entry['datatype'] == 'TMIN':
                record['tmin'] = entry['value']
            elif entry['datatype'] == 'AWND':
                record['awnd'] = entry['value']
            elif entry['datatype'] == 'PRCP':
                record['prcp'] = entry['value']

            all_data.append(record)

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    if df.empty:
        print("No weather data collected from any station.")
        return pd.DataFrame()

    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Drop rows with invalid dates
    df = df.dropna(subset=['date'])

    # Aggregate by date (average across stations)
    df_daily_avg = df.groupby('date').agg({
        'tmax': 'mean',
        'tmin': 'mean',
        'awnd': 'mean',
        'prcp': 'mean'
    }).reset_index()

    return df_daily_avg

# ========== Main Script ==========

if __name__ == "__main__":
    print("Fetching station list around NYC...")
    stations = get_stations(lat, lon, radius)

    if not stations:
        print("No stations found. Exiting.")
    else:
        print(f"Found {len(stations)} stations. Fetching weather data...")
        aggregated_data = aggregate_weather_data(stations)

        if not aggregated_data.empty:
            aggregated_data.to_csv("nyc_weather_data.csv", index=False)
            print("Aggregated weather data saved to 'nyc_weather_data.csv'")
        else:
            print("No valid weather data available to save.")
