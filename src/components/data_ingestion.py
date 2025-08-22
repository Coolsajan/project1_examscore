from utils.logger import logging
from utils.exception import CustomException

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import kaggle
import sys
import os
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
 
load_dotenv(".env")
USER_NAME = os.getenv("USER_NAME")
DATASET_NAME = os.getenv("DATASET_NAME")

class DataIngestion:
    """
    This class provides method to ingest the data from DataSourse (Kaggle)
    """

    def __init__(self ,dataingestionconfig :DataIngestionConfig):
        
        self.data_path = dataingestionconfig.DATA_INGESTION_PATH
        self.train_data_path = dataingestionconfig.TRAIN_PATH
        self.test_data_path = dataingestionconfig.TEST_PATH

        self.status = None

    def get_data_from_kaggle(self ,data_user_name :str = USER_NAME , dataset_name :str = DATASET_NAME ,
                            kaggle_username:str = "KAGGLE_USERNAME",kaggle_key : str = "KAGGLE_KEY" ) ->str:
        """
        This method will create the connection with the kaggle and get the dataset 
        """
        try:
            DataIngestion.auth_kaggle(username=kaggle_username,kaggle_key=kaggle_key)
            dataset_link = data_user_name+"/"+dataset_name
            os.makedirs(self.data_path, exist_ok=True)
            dataset_path = self.data_path

            kaggle.api.dataset_download_files(dataset_link ,path=dataset_path,unzip= True)
            logging.info(f"DataSet saved at : {dataset_path}")

            return dataset_path
        except Exception as e:
            raise CustomException(e,sys)
        
    @staticmethod
    def auth_kaggle(username:str , kaggle_key : str):
        """This will setup the kaggle."""
        try:
            load_dotenv(".env")
            os.environ['KAGGLE_USERNAME'] = os.getenv(username)
            os.environ['KAGGLE_KEY'] = os.getenv(kaggle_key)
            kaggle.api.authenticate()
            logging.info("Kaggle connection established")
        except Exception as e:
            raise CustomException(e,sys)
        
    def split_test_train(self ,filepath:str,train_size : int = 0.8)-> pd.DataFrame:
        """
        This will split the data into train and test data and store it.
        """
        try:

            csv_path = os.path.join(filepath, "StudentPerformanceFactors.csv")
            df = pd.read_csv(csv_path)
            
            logging.info(f"DataFrame with shape : {df.shape} obtained")
            train_df , test_df = train_test_split(df , train_size= train_size)

            os.makedirs(self.train_data_path,exist_ok=True)
            os.makedirs(self.test_data_path,exist_ok=True)
            
            train_path = os.path.join(self.train_data_path , "train.csv")
            test_path = os.path.join(self.test_data_path , "test.csv")

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            logging.info(f"Train and Test data saved to {train_path}, {test_path} respectively.")
            
            # Return the file paths (assuming DataIngestionArtifact expects file paths, not directories)
            return train_path, test_path
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        """
        This method will initaite the process.
        """
        try:
            logging.info("DataIngestion started.")
            data_path = self.get_data_from_kaggle()
            
            logging.info("Dataset obtained.")
            train_path , test_path = self.split_test_train(filepath=data_path)
            dataingestionartifact = DataIngestionArtifact(train_path,test_path)
            logging.info("DATA INGESTION COMPLETED.")
            return dataingestionartifact

        except Exception as e:
            raise CustomException(e,sys)