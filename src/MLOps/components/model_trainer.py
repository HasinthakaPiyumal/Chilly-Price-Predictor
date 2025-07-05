import pandas as pd
import os
from MLOps import logger
from sklearn.linear_model import ElasticNet
import joblib
from MLOps.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        X_train = train_data.drop(columns=[self.config.target_column])
        y_train = train_data[self.config.target_column]
        X_test = test_data.drop(columns=[self.config.target_column])
        y_test = test_data[self.config.target_column]

        model = ElasticNet(
            alpha=self.config.alpha,
            l1_ratio=self.config.l1_ratio,
            random_state=self.config.random_state
        )
        
        model.fit(X_train, y_train)
        
        model_file_path = os.path.join(self.config.root_dir, self.config.model_name)
        joblib.dump(model, model_file_path)