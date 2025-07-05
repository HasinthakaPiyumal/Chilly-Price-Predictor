from MLOps.config.configuration import ConfigurationManager
from MLOps.components.model_trainer import ModelTrainer
from MLOps import logger

class ModelTrainingPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_trainer_config()

    def run(self):
        try:
            model_trainer = ModelTrainer(config=self.config)
            model_trainer.train()
        except Exception as e:
            raise e