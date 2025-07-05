import os
import urllib.request as request
import zipfile
from MLOps import logger
from MLOps.utils.common import get_size
from MLOps.entity.config_entity import DataIngestionConfig
from MLOps.utils.common import create_directories,get_size

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        create_directories([self.config.root_dir])
    
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            logger.info(f"Downloading file from {self.config.source_URL} to {self.config.local_data_file}")
            request.urlretrieve(self.config.source_URL, self.config.local_data_file)
            logger.info(f"Downloaded {get_size(self.config.local_data_file)} bytes")
        else:
            logger.info(f"File already exists at {self.config.local_data_file}")
    
    def extract_zip_file(self):
        logger.info(f"Extracting zip file to {self.config.unzip_dir}")
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(self.config.unzip_dir)
        logger.info("Extraction completed")
        