from utils.logger import logging
from utils.exception import CustomException

import pandas as pd
import kaggle
import sys
import os
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
 
load_dotenv()
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")
USER_NAME = os.getenv("USER_NAME")
DATASET_NAME = os.getenv("DATASET_NAME")

class DataIngestion:
    """
    This class provides method to ingest the data from DataSourse (Kaggle)
    """

    def __init__(self ,dataingestionconfig :DataIngestionConfig = DataIngestionConfig()):
        
        self.data_path = dataingestionconfig.DATASET_PATH
        self.train_data_path = dataingestionconfig.TRAIN_PATH
        self.test_data_path = dataingestionconfig.TEST_PATH

        self.status = None

    def get_data_from_kaggle(self ,user_name :str = USER_NAME , dataset_name :str = DATASET_NAME ) ->str:
        """
        This method will create the connection with the kaggle and get the dataset 
        """
        try:
            os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
            os.environ['KAGGLE_KEY'] = KAGGLE_KEY
            kaggle.api.authenticate()
            logging.info("Kaggle connection established")

            dataset_link = user_name+"/"+dataset_name
            os.makedirs(self.data_path, exist_ok=True)
            dataset_path = self.data_path

            kaggle.api.dataset_download_files(dataset_link ,path=dataset_path,unzip= True)
            logging.info(f"DataSet saved at : {dataset_path}")

            return dataset_path
        except Exception as e:
            raise CustomException(e,sys)
        
    def split_test_train(self ,filepath:str,train_size : int = 0.8)-> pd.DataFrame:
        """
        This will split the data into train and test data and store it.
        """
        try:
            df = pd.read_csv(filepath)
            logging.info(f"DataFrame with shape : {df.shape} obtained")
            train_df , test_df = train_test_split(df , train_size= train_size)

            os.makedirs(self.train_data_path,exist_ok=True)
            train_path = self.train_data_path

            os.makedirs(self.test_data_path,exist_ok=True)
            test_path = self.test_data_path

            train_df.to_csv(train_path,index=False)
            test_df.to_csv(test_path,index=False)

            logging.info(f"Train and Test data saved to {train_path} , {test_path} respectively.")
            return train_path , test_path
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
            dataingestionartifact = dataingestionartifact(train_path,test_path)
            logging.info("DATA INGESTION COMPLETED.")
            return dataingestionartifact

        except Exception as e:
            raise CustomException(e,sys)