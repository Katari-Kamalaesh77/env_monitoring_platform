import pandas as pd
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the cleaned weather data
df = pd.read_csv("cleaned_weather_data.csv", parse_dates=["DATE"], low_memory=False)

# Aggregate to daily average if multiple entries exist per day
daily_weather = df.groupby("DATE").agg({
    "Max_Temperature": "mean", 
    "Min_Temperature": "mean", 
    "Average_Temperature": "mean"
}).reset_index()

# Ensure output directory exists
os.makedirs("sarima_outputs", exist_ok=True)

# Loop over each weather variable to run SARIMA
for column in ["Max_Temperature", "Min_Temperature", "Average_Temperature"]:
    print(f"\nRunning SARIMA for: {column}")

    # Prepare the data
    data = daily_weather[["DATE", column]].dropna()
    data = data.rename(columns={column: "y"})
    data = data.set_index("DATE")

    # Define the SARIMA model
    sarima_model = SARIMAX(data["y"], 
                           order=(1, 0, 0), 
                           seasonal_order=(1, 0, 1, 12),
                           enforce_stationarity=False, 
                           enforce_invertibility=False)
    
    # Fit the model
    results = sarima_model.fit()

    # Print the summary of the model
    print(results.summary())

    # Save the SARIMA model results
    results.save(f"sarima_outputs/{column}_sarima_results.pkl")

    # Plot the actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data["y"], label="Actual")
    plt.plot(data.index, results.fittedvalues, label="Fitted", color='red')
    plt.title(f"SARIMA Model for {column}")
    plt.xlabel("Date")
    plt.ylabel(f"{column}")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(f"sarima_outputs/{column}_sarima_plot.png")
    plt.close()

print("SARIMA modeling and plotting complete!")
