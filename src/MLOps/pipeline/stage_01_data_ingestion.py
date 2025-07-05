from MLOps.config.configuration import ConfigurationManager
from MLOps.components.data_ingestion import DataIngestion

class DataIngestionPipeline:
    def __init__(self):
        self.config = ConfigurationManager()
        self.data_ingestion_config = self.config.get_data_ingestion_config()
        self.data_ingestion = DataIngestion(config=self.data_ingestion_config)

    def run(self):
        try:
            self.data_ingestion.download_file()
            self.data_ingestion.extract_zip_file()
        except Exception as e:
            raise e