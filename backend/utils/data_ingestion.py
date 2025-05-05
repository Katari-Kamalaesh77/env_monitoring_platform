import requests
import pandas as pd

def fetch_data_from_api(api_url):
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            json_data = response.json()
            return json_data.get("Data", [])  # Return only the relevant list
        else:
            print(f"Failed to fetch data: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

def get_clean_pm25_dataframe(data):
    # Turn list of dictionaries into DataFrame
    if not data:
        print("No data returned from API.")
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Validate columns
    if 'date_local' not in df.columns or 'arithmetic_mean' not in df.columns:
        print("Required fields missing in data.")
        return pd.DataFrame()

    # Drop rows with missing values
    df = df[['date_local', 'arithmetic_mean']].dropna()

    # Rename columns for Prophet
    df.rename(columns={'date_local': 'ds', 'arithmetic_mean': 'y'}, inplace=True)

    # Convert data types
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')

    # Drop any row with invalid dates or values
    df = df.dropna().sort_values('ds')

    return df
