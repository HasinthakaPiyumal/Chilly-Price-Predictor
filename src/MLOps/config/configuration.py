from MLOps.constants import *
from MLOps.utils.common import read_yaml, create_directories
from MLOps.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig,ModelTrainerConfig,ModelEvaluationConfig

class ConfigurationManager:
    def __init__(self, config_file_path=CONFIG_FILE_PATH, params_file_path=PARAMS_FILE_PATH, schema_file_path=SCHEMA_FILE_PATH):
        
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        self.schema = read_yaml(schema_file_path)
        create_directories(list_of_directories=[self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        data_ingestion_config = self.config.data_ingestion
        create_directories([data_ingestion_config.root_dir])
        return DataIngestionConfig(
            root_dir=data_ingestion_config.root_dir,
            source_URL=data_ingestion_config.source_URL,
            local_data_file=data_ingestion_config.local_data_file,
            unzip_dir=data_ingestion_config.unzip_dir
        )
        
    def get_data_validation_config(self) -> DataValidationConfig:
        data_validation_config = self.config.data_validation
        schema = self.schema.COLUMNS
        create_directories([data_validation_config.root_dir])
        return DataValidationConfig(
            root_dir=data_validation_config.root_dir,
            status_file=data_validation_config.STATUS_FILE,
            unzip_data_dir=data_validation_config.unzip_data_dir,
            all_schema=schema
        )
        
    def get_data_transformation_config(self) -> DataTransformationConfig:
        data_transformation_config = self.config.data_transformation
        schema = self.schema.COLUMNS
        create_directories([data_transformation_config.root_dir])
        return DataTransformationConfig(
            root_dir=data_transformation_config.root_dir,
            data_path=data_transformation_config.data_path,
        )
        
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        model_trainer_config = self.config.model_trainer
        schema = self.schema.TARGET_COLUMN
        params = self.params.ElasticNet
        create_directories([model_trainer_config.root_dir])
        return ModelTrainerConfig(
            root_dir=model_trainer_config.root_dir,
            train_data_path=model_trainer_config.train_data_path,
            test_data_path=model_trainer_config.test_data_path,
            model_name=model_trainer_config.model_name,
            alpha=params.alpha,
            l1_ratio=params.l1_ratio,
            random_state=params.random_state,
            target_column=schema.name
        )
        
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.ElasticNet
        
        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            model_path=config.model_path,
            test_data_path=config.test_data_path,
            all_param=params,
            metric_file_path=config.metric_file_path,
            target_column=self.schema.TARGET_COLUMN.name
        )
        print(model_evaluation_config)
        
        return model_evaluation_config