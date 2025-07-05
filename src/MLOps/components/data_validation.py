from MLOps.entity.config_entity import DataValidationConfig
import pandas as pd

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self)-> bool:
        try:
            validation_status = None
            
            data = pd.read_csv(self.config.unzip_data_dir)
            all_columns = list(data.columns)
            
            schema = self.config.all_schema.keys()
            
            for cols in all_columns:
                if cols not in schema:
                    validation_status = False
                    with open(self.config.status_file, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                    break
                else:
                    validation_status = True
                    with open(self.config.status_file, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                        
            return validation_status
        except Exception as e:
            raise e
        