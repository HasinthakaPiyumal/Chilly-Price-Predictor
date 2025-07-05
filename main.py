from MLOps.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from MLOps.pipeline.stage_02_data_validation import DataValidationPipeline
from MLOps.pipeline.stage_03_data_transformation import DataTransformationPipeline
from MLOps.pipeline.stage_04_model_training import ModelTrainingPipeline
from MLOps.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline
from MLOps import logger

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<")
    data_ingestion_pipeline = DataIngestionPipeline()
    data_ingestion_pipeline.run()
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<")
    data_validation_pipeline = DataValidationPipeline()
    data_validation_pipeline.run()
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Transformation Stage"
try:
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<")
    data_transformation_pipeline = DataTransformationPipeline()
    data_transformation_pipeline.run()
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Training Stage"
try:
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<")
    model_training_pipeline = ModelTrainingPipeline()
    model_training_pipeline.run()
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Evaluating Stage"
try:
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<")
    model_training_pipeline = ModelEvaluationPipeline()
    model_training_pipeline.run()
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e