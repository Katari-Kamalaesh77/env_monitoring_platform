# backend/ml/data_fetching.py

import os
import time
import requests
import pandas as pd
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    filename="data_fetching_errors.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.ERROR
)

# Create data directory
SAVE_DIR = "epa_data_by_state"
os.makedirs(SAVE_DIR, exist_ok=True)

# Constants
EMAIL = "kamalaesh.katari03@gmail.com"
API_KEY = "greenmouse29"
BASE_URL = "https://aqs.epa.gov/data/api/dailyData/byState"

STATE_CODE = "36"  # New York
PARAMS = {
    "CO": "42101",
    "PM2.5": "88101",
    "O3": "44201",
    "NO2": "42602",
    "SO2": "42401",
    "PM10": "81102",
}
YEARS = list(range(1995, 2025))


def fetch_data(param_code, year):
    url = (
        f"{BASE_URL}?email={EMAIL}&key={API_KEY}"
        f"&param={param_code}&bdate={year}0101&edate={year}1231&state={STATE_CODE}"
    )
    try:
        response = requests.get(url)
        response.raise_for_status()
        json_data = response.json()
        if json_data["Data"]:
            return pd.DataFrame(json_data["Data"])
        else:
            print(f"No data for param={param_code} in {year}")
            return None
    except Exception as e:
        logging.error(f"Error fetching param={param_code} for year={year}: {e}")
        return None


def main():
    for gas_name, param_code in PARAMS.items():
        print(f"\nFetching data for {gas_name}...")
        for year in tqdm(YEARS):
            df = fetch_data(param_code, year)
            if df is not None and not df.empty:
                filename = f"{SAVE_DIR}/{gas_name}_{year}.csv"
                try:
                    df.to_csv(filename, index=False)
                except Exception as e:
                    logging.error(f"Failed to save CSV for {gas_name} {year}: {e}")
            time.sleep(1)  # To avoid hitting rate limits


if __name__ == "__main__":
    main()
