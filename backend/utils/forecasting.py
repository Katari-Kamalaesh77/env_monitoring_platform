from fbprophet import Prophet
import pandas as pd

def train_forecasting_model(data):
    # Assuming 'data' is a DataFrame with 'date' and 'value' columns
    df = pd.DataFrame(data, columns=['date', 'value'])
    df['ds'] = pd.to_datetime(df['date'])
    df['y'] = df['value']
    
    model = Prophet()
    model.fit(df)
    
    return model

def predict_future(model, periods=5):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']]
