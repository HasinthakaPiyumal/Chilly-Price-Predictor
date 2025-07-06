import pandas as pd
import os
from MLOps import logger
from sklearn.linear_model import ElasticNet
from MLOps.entity.config_entity import ModelTrainerConfig
from prophet import Prophet
import cloudpickle 
from datetime import datetime

class Predictor:
    def __init__(self, model,dollar_rate, last_date):
        self.model = model
        self.dollar_rate = dollar_rate
        self.last_date = last_date
        
    def predict(self, periods,freq='W',include_history=False,start_from=None) -> pd.Series:
        date_diff = (datetime.now() - pd.to_datetime(self.last_date)).days
        future = self.model.make_future_dataframe(periods=periods+date_diff, freq=freq,include_history=include_history)  # monthly
        predictions = self.model.predict(future)
        
        if not include_history:
            predictions = predictions[predictions['ds'] > datetime.now()]
        elif include_history and start_from:
            predictions = predictions[predictions['ds'] >= pd.to_datetime(start_from)]
        predictions['yhat'] = predictions['yhat'] * self.dollar_rate  # Adjusting predictions based on dollar rate
        predictions['yhat_lower'] = predictions['yhat_lower'] * self.dollar_rate
        predictions['yhat_upper'] = predictions['yhat_upper'] * self.dollar_rate
        return predictions

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        print(f"Training model with config: {self.config.train_data_path}")
        data = pd.read_csv(self.config.train_data_path)

        train_data = data.copy()
        train_data['date'] = pd.to_datetime(train_data['date'])
        train_data = train_data.rename(columns={'date': 'ds', 'today': 'y'})
        print(train_data.tail())
        model = Prophet(yearly_seasonality=True)
        model.fit(train_data)
        
        predictor = Predictor(model=model, dollar_rate=data.iloc[-1]['dollar_rate'],last_date = data.iloc[-1]['date'])
        
        model_file_path = os.path.join(self.config.root_dir, self.config.model_name)
        # joblib.dump(predictor, model_file_path)
        with open(model_file_path, 'wb') as f:
            cloudpickle.dump(predictor, f)