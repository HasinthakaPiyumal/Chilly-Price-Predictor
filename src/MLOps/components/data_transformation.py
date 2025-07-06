import os
from sklearn.model_selection import train_test_split
import pandas as pd
from MLOps.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_split(self):
        data = pd.read_csv(self.config.data_path)
        
        data.date = pd.to_datetime(data.date)
        train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
        # Save all data to Future accuracy improvement
        data['today'] = data['today'] / data['dollar_rate']
        
        last_30_days = data[:-30].copy()
        data = data.copy()
        data.to_csv(os.path.join(self.config.root_dir, 'full.csv'), index=False)
        data[:-30].to_csv(os.path.join(self.config.root_dir, 'train.csv'), index=False)
        test_set = data[-30:].copy()
        test_set['today'] = test_set['today'] * test_set['dollar_rate']
        test_set.to_csv(os.path.join(self.config.root_dir, 'test.csv'), index=False)
        
        return train_set, test_set