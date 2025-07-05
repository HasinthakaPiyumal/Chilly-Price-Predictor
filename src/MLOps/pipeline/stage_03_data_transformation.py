from MLOps.entity.config_entity import DataTransformationConfig
from MLOps.config.configuration import ConfigurationManager
from MLOps.components.data_transformation import DataTransformation
from pathlib import Path
from MLOps import logger


class DataTransformationPipeline:
    def __init__(self):
        self.config = ConfigurationManager()

    def run(self):
        try:
            with open(self.config.get_data_validation_config().status_file, 'r') as f:
                status = f.read().split(" ")[-1]
            
            if status == "True":
                config = self.config
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(config=data_transformation_config)
                data_transformation.train_test_split()
            
            else:
                raise Exception("Data Validation failed, cannot proceed with Data Transformation.")
        
        except Exception as e:
            print(e)
        