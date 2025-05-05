import pandas as pd
import requests

def fetch_pm25_data(start_date: str, end_date: str, state_code="36"):
    url = "https://aqs.epa.gov/data/api/dailyData/byState"
    params = {
        "email": "kamalaesh.katari03@gmail.com",
        "key": "greenmouse29",
        "param": "88101",
        "bdate": start_date,
        "edate": end_date,
        "state": state_code
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "Data" in data:
        df = pd.DataFrame(data["Data"])
        return df
    else:
        return pd.DataFrame()

def clean_pm25_data(df: pd.DataFrame):
    print("COLUMNS:", df.columns.tolist())
    df = df[["date_local", "arithmetic_mean"]]
    df = df.rename(columns={"arithmetic_mean": "pm25"})
    print(df.columns.tolist())
    df["date_local"] = pd.to_datetime(df["date_local"], format='%m/%d/%Y', errors='coerce')
    df = df.rename(columns={"date_local": "ds", "pm25": "y"})
    df = df.groupby("ds").mean().reset_index()
    return df
