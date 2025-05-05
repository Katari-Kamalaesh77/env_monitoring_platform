import os
import pandas as pd

# Define the directory containing the weather data files
data_dir = r'D:\env_monitoring_platform\backend\weather_data_files'

# Path to the text file containing station IDs
station_ids_file = os.path.join(data_dir, 'ny_station_ids.txt')

# Read the station IDs from the text file
with open(station_ids_file, 'r') as f:
    station_ids = [line.strip() for line in f if line.strip()]

# Initialize a list to store data from all stations
all_data = []

# Process each station ID
for station_id in station_ids:
    # Construct the filename without the 'GHCND:' prefix
    filename = f"{station_id}.csv"
    filepath = os.path.join(data_dir, filename)

    # Check if the file exists
    if not os.path.isfile(filepath):
        print(f"Warning: File {filepath} does not exist.")
        continue

    try:
        # Read the CSV file into a DataFrame with low_memory=False to handle mixed data types
        df = pd.read_csv(filepath, low_memory=False)

        # Add a column for the station ID
        df['station_id'] = station_id

        # Append the DataFrame to the list
        all_data.append(df)
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

# Combine all DataFrames into a single DataFrame
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)

    # Define the output file path
    output_file = os.path.join(data_dir, 'combined_weather_data.csv')

    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")
else:
    print("No data was processed.")
