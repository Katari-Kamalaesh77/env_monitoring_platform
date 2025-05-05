import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load cleaned weather data
df = pd.read_csv("cleaned_weather_data.csv", parse_dates=["DATE"])

# Select relevant columns for Prophet (DATE and Max_Temperature)
df_tmax = df[["DATE", "Max_Temperature"]].dropna()

# Prepare the data for Prophet
df_tmax = df_tmax.rename(columns={"DATE": "ds", "Max_Temperature": "y"})

# Initialize and fit the model
model = Prophet()
model.fit(df_tmax)

# Create future dataframe for next 365 days
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Plot forecast
fig1 = model.plot(forecast)
plt.title("Max Temperature Forecast (Prophet)")
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.tight_layout()
plt.savefig("prophet_tmax_forecast.png")

# Optional: Save forecast
forecast.to_csv("tmax_prophet_forecast.csv", index=False)
print("Forecast completed and saved.")
