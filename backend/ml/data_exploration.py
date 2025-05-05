import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

# Mapping parameter codes to pollutant names
PARAMETER_CODE_TO_NAME = {
    "42101": "CO",
    "42401": "SO2",
    "42602": "NO2",
    "44201": "O3",
    "81102": "PM10",
    "88101": "PM2.5"
}

def load_and_process_file(file_path):
    """Load CSV, process datetime and concentration, and resample monthly."""
    try:
        df = pd.read_csv(file_path)
        if "date_local" not in df.columns or "arithmetic_mean" not in df.columns:
            print(f" Skipping {file_path} — required columns missing.")
            return None, None

        df = df[['date_local', 'arithmetic_mean']].copy()
        df.columns = ['Date', 'Concentration']
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Displaying the raw data preview
        print(f"\nRaw Data Preview (First 5 rows) for {file_path}:")
        print(df.head())

        df = df.dropna()
        df = df[df['Concentration'] > -999]

        # Display cleaned data after dropping NAs and filtering
        print(f"\nCleaned Data (After dropping NaNs and invalid values) for {file_path}:")
        print(df.head())

        df.set_index('Date', inplace=True)
        df_monthly = df.resample('M').mean()

        # Display monthly resampled data
        print(f"\nMonthly Resampled Data for {file_path}:")
        print(df_monthly.head())

        return df_monthly, None
    except Exception as e:
        return None, str(e)

def plot_time_series(df_monthly, pollutant):
    """Plot monthly average time series for a pollutant."""
    plt.figure(figsize=(12, 6))
    plt.plot(df_monthly.index, df_monthly['Concentration'], label=pollutant)
    plt.title(f"{pollutant} Monthly Average Concentration")
    plt.xlabel("Date")
    plt.ylabel("Concentration")
    plt.grid(True)
    plt.tight_layout()
    print(f" Showing plot for {pollutant}...")
    plt.show()  # Display plot without saving it

def generate_correlation_heatmap(monthly_dataframes):
    """Create and display a correlation heatmap from all pollutant data."""
    combined_df = pd.concat(monthly_dataframes.values(), axis=1)
    combined_df.columns = list(monthly_dataframes.keys())
    combined_df = combined_df.dropna()

    if combined_df.empty:
        print(" Skipping heatmap — no overlapping data across pollutants.")
        return

    corr_matrix = combined_df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation between Pollutants (Monthly Avg)")
    plt.tight_layout()
    print(" Showing correlation heatmap...")
    plt.show()  # Display heatmap without saving it

def run_exploration(data_dir="D:/env_monitoring_platform/backend/epa_data_by_state"):
    """Main function to run EDA for EPA air quality data."""
    csv_files = glob(os.path.join(data_dir, "*_all.csv"))
    print(f"Found CSV files: {csv_files}")  # Check if files are found
    monthly_dataframes = {}

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        code = filename.split("_")[0]
        pollutant = PARAMETER_CODE_TO_NAME.get(code, code)

        print(f"\nProcessing: {filename} as {pollutant}")
        df_monthly, error = load_and_process_file(file_path)

        if error:
            print(f" Error loading {filename}: {error}")
            continue
        if df_monthly is None or df_monthly.empty:
            print(f" Skipping {pollutant} — no valid data.")
            continue

        monthly_dataframes[pollutant] = df_monthly
        plot_time_series(df_monthly, pollutant)

    if monthly_dataframes:
        generate_correlation_heatmap(monthly_dataframes)

    print("\nEDA complete for all pollutants.")

# Only run if executed as script
if __name__ == "__main__":
    run_exploration()
