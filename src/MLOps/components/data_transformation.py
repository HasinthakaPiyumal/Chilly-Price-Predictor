import os
from sklearn.model_selection import train_test_split
import pandas as pd
from MLOps.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_split(self):
        data = pd.read_csv(self.config.data_path)
        train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
        
        train_set.to_csv(os.path.join(self.config.root_dir, 'train.csv'), index=False)
        test_set.to_csv(os.path.join(self.config.root_dir, 'test.csv'), index=False)
        
        return train_set, test_set