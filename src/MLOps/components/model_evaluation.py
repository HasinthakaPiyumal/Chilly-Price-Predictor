from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import numpy as np,pandas as pd
from MLOps.config.configuration import ModelEvaluationConfig
from MLOps.utils.common import save_json,create_directory

from prophet import Prophet
import cloudpickle

class Predictor:
    def __init__(self, model,dollar_rate):
        self.model = model
        self.dollar_rate = dollar_rate

    def predict(self, periods,freq) -> pd.Series:
        future = self.model.make_future_dataframe(periods=periods, freq=freq,include_history=True)  # monthly
        predictions = self.model.predict(future)
        predictions['yhat'] = predictions['yhat'] * self.dollar_rate  # Adjusting predictions based on dollar rate
        predictions['yhat_lower'] = predictions['yhat_lower'] * self.dollar_rate
        predictions['yhat_upper'] = predictions['yhat_upper'] * self.dollar_rate
        return predictions

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        # self.model = joblib.load(self.config.model_path)
        with open(self.config.model_path, 'rb') as f:
            self.model = cloudpickle.load(f)

        self.test_data = pd.read_csv(self.config.test_data_path)
        self.target_column = self.config.target_column
        
    def evaluate_model(self,y_act,y_pred):
        rmse = np.sqrt(mean_squared_error(y_act,y_pred))
        mae = mean_absolute_error(y_act,y_pred)
        r2 = r2_score(y_act,y_pred)
        return rmse,mae,r2

    def evaluate(self):
        X_test = self.test_data.drop(columns=[self.target_column])
        y_test = self.test_data[self.target_column]
        
        pred = self.model.predict(30, 'D')
        y_pred = pred['yhat'][-30:].values
        rmse, mae, r2 = self.evaluate_model(y_test, y_pred)
        print(y_pred)
        print(y_test)
        
        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        
        save_json(self.config.metric_file_path,metrics,)