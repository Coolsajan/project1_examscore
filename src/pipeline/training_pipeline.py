from utils.exception import CustomException
from utils.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation

from src.entity.config_entity import DataIngestionConfig,DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact

import os, sys

class ModelTrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method of TrainPipeline class is responsible for starting data ingestion component
        """
        try:
            logging.info("Starting Data Ingestion.")
            dataingestionobj = DataIngestion(dataingestionconfig=self.data_ingestion_config)
            dataingestionartifact = dataingestionobj.initiate_data_ingestion()
            logging.info("Data Ingestion Completed. Sucessfully")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            return dataingestionartifact

        except Exception as e:
            raise CustomException(e,sys)
        
    def start_data_validation(self,dataingestionartifact:DataIngestionArtifact)->DataValidationArtifact:
        """
        This method of TrainPipeline class is responsible for starting data validation component
        """
        try:
            logging.info("Starting data validation component in training pipeline.")
            datavalidationobj = DataValidation(data_validation_config=DataValidationConfig(),
                                               data_ingestion_artifact=dataingestionartifact)
            datavalidationartifact = datavalidationobj.initiate_data_validation()
            logging.info("Data Validation Completed. Sucessfully")
            logging.info("Exited the start_data_validation method of TrainPipeline class")
            return datavalidationartifact
        except Exception as e:
            raise CustomException(e,sys)


    def run_pipeline(self, ) -> None:
        """
        This method of TrainPipeline class is responsible for running complete pipeline
        """
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(dataingestionartifact=data_ingestion_artifact)
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e,sys)