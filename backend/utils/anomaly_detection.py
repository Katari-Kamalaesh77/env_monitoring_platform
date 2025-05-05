from sklearn.ensemble import IsolationForest
import pandas as pd

def train_anomaly_detection_model(data):
    df = pd.DataFrame(data, columns=['value'])
    
    # Assuming data needs to be transformed to fit the model
    model = IsolationForest(n_estimators=100)
    model.fit(df[['value']])
    
    return model

def detect_anomalies(model, data):
    df = pd.DataFrame(data, columns=['value'])
    anomalies = model.predict(df[['value']])
    
    return anomalies
