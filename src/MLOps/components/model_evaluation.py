from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import numpy as np,pandas as pd
import joblib
from MLOps.config.configuration import ModelEvaluationConfig
from MLOps.utils.common import save_json,create_directory

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.model = joblib.load(self.config.model_path)
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
        
        y_pred = self.model.predict(X_test)
        
        rmse, mae, r2 = self.evaluate_model(y_test, y_pred)
        
        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        
        save_json(self.config.metric_file_path,metrics,)