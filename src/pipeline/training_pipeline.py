from utils.exception import CustomException
from utils.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining
from src.components.model_eval import ModelEvaluation

from src.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransfromationConfig,ModelTrainingConfig,ModelEvaluationConfig
from src.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact,ModelTrainigArtifact,ModelEvaluationArtifact

import os, sys

class ModelTrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransfromationConfig()
        self.model_trainer_config = ModelTrainingConfig()
        self.model_evalutation_config = ModelEvaluationConfig()

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
            datavalidationobj = DataValidation(data_validation_config=self.data_validation_config,
                                               data_ingestion_artifact=dataingestionartifact)
            datavalidationartifact = datavalidationobj.initiate_data_validation()
            logging.info("Data Validation Completed. Sucessfully")
            logging.info("Exited the start_data_validation method of TrainPipeline class")
            return datavalidationartifact
        except Exception as e:
            raise CustomException(e,sys)
        
    def start_data_transformation(self,dataingestionartifact:DataIngestionArtifact,datavalidationartifact:DataValidationArtifact) -> DataTransformationArtifact:
        """
        This method of TrainPipeLine class is responsible for stating data transformation component.
        """
        try:
            logging.info("Starting the data transformation.")
            datatransformationobj = DataTransformation(data_ingestion_artifact=dataingestionartifact,
                                                       data_transformation_config=self.data_transformation_config,
                                                       data_validation_artifact=datavalidationartifact)
            datatransformationartifact = datatransformationobj.initiat_data_transformation()
            logging.info("Data Transformation Completed. Sucessfully")
            logging.info("Exited the start_data_transformation method of TrainPipeline class")
            return datatransformationartifact
        except Exception as e:
            raise CustomException(e,sys)
        
    def start_model_trainner(self ,data_transformation_artifact : DataTransformationArtifact)-> ModelTrainigArtifact:
        """
        This method of TrainPipeLine class is responsible for stating model trainer component.
        """
        try:
            modeltrainerobj = ModelTraining(modeltrainingconfig=self.model_trainer_config,
                                              datatransformationartifact=data_transformation_artifact)
            
            modeltrainerartifact = modeltrainerobj.initiate_model_train()
            logging.info("Model Trainner Completed. Sucessfully")
            logging.info("Exited the start_model_trainner method of TrainPipeline class")
            return modeltrainerartifact
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def start_model_evaluation(self,model_traininer_artifact : ModelTrainigArtifact , data_transformation_artifact:DataTransformationArtifact) -> ModelEvaluationArtifact:
        """
        This method of TrainPipeLine class is responsible for stating model evaluation component.
        """
        try:
            logging.info("Model evalutaion started in taining pipeline")
            modelevalobj = ModelEvaluation(modelevaluationconfig=self.model_evalutation_config,\
                                           modeltrainingartifact=model_traininer_artifact,
                                           datatransformationartifact=data_transformation_artifact)
            modelevalartifact = modelevalobj.initiate_model_eval()
            logging.info("Model TraEvaluation inner Completed. Sucessfully")
            logging.info("Exited the start_model_evaluation method of TrainPipeline class")
            return modelevalartifact

        except Exception as e:
            raise CustomException(e,sys)


    def run_pipeline(self, ) -> None:
        """
        This method of TrainPipeline class is responsible for running complete pipeline
        """
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(dataingestionartifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(dataingestionartifact=data_ingestion_artifact,
                                                                          datavalidationartifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainner(data_transformation_artifact=data_transformation_artifact)
            model_eval_artifact = self.start_model_evaluation(model_traininer_artifact=model_trainer_artifact,
                                                              data_transformation_artifact=data_transformation_artifact)
            print(f"Best performing model path is {model_eval_artifact}")
            return model_eval_artifact
        except Exception as e:
            raise CustomException(e,sys)