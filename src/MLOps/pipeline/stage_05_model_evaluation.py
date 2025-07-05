from MLOps.config.configuration import ConfigurationManager
from MLOps.components.model_evaluation import ModelEvaluation
from MLOps import logger

class ModelEvaluationPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def run(self):
        try:
            model_trainer = ModelEvaluation(config=self.config)
            model_trainer.evaluate()
        except Exception as e:
            raise e