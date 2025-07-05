import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from MLOps.config.configuration import ConfigurationManager

class PredictionPipeline:
    def __init__(self):
        config = ConfigurationManager()
        self.model = joblib.load(config.get_model_evaluation_config().model_path)

    def predict(self,data):        
        prediction = self.model.predict(data)
        
        return prediction