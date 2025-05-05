import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load cleaned weather data
df = pd.read_csv("D:/env_monitoring_platform/backend/weather_data_files/cleaned_weather_data.csv", parse_dates=["DATE"])

# Rename columns to expected short names
df = df.rename(columns={
    "Min_Temperature": "TMIN",
    "Max_Temperature": "TMAX",
    "Average_Temperature": "TAVG",
    "Precipitation": "PRCP",
    "Snowfall": "SNOW"
})

# Drop duplicated columns, if any
df = df.loc[:, ~df.columns.duplicated()]

# Define the features and target
features = ["TMIN", "TAVG", "PRCP", "SNOW", "AWND"]
target = "TMAX"

# Select relevant columns and drop missing values
df = df[["DATE"] + features + [target]].dropna()

# Create lag features (1-day lag)
for col in features + [target]:
    df[f"{col}_lag1"] = df[col].shift(1)

df = df.dropna()

# Split data into train and test sets (80-20 split)
X = df[features + [f"{col}_lag1" for col in features + [target]]]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train the XGBoost model
model = xgb.XGBRegressor(objective="reg:squarederror")
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Print results
print(f"Root Mean Squared Error (RMSE): {rmse}")
