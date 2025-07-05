from MLOps.config.configuration import ConfigurationManager
from MLOps.components.data_validation import DataValidation

class DataValidationPipeline:
    def __init__(self):
        self.config = ConfigurationManager()
        self.data_validation_config = self.config.get_data_validation_config()
        self.data_validation = DataValidation(config=self.data_validation_config)

    def run(self):
        try:
            self.data_validation.validate_all_columns()
        except Exception as e:
            raise e