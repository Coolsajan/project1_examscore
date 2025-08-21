from utils.exception import CustomException
from utils.logger import logging
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact ,DataValidationArtifact
from utils.common_utils import read_yaml_file , write_yaml_file
import json
from evidently import Report
from evidently.metrics import *
from evidently.presets import *

import pandas as pd
import os, sys


class DataValidation():
    """
    This class will validate the data using EvidentlyAI
    """

    def __init__(self , data_validation_config :DataValidationConfig = DataValidationConfig() ):
        self.data_val_report_path = data_validation_config.DATA_VALIDATION_REPORT
        self._config_schema = read_yaml_file("schema.yaml")
        self.train_data = None
        self.test_data = None

        
    def check_columns_len(self,df : pd.DataFrame)->bool:
        """
        This method will check the number of columns.
        """
        try:
            status =len(df.columns) == len(self._config_schema['columns'])
            logging.info(f"Is required column present: [{status}]")
            return status
        except Exception as e:
            raise CustomException(e,sys)
        
    def check_req_columns(self,df:pd.DataFrame)->bool:
        """
        This method will check if the required columns are present in the df or not.
        """
        try:
            df_columns = df.columns
            missing_cate_columns = []
            missing_num_columns = []

            for column in self._config_schema['numerical_columns']:
                if column not in df_columns:
                    missing_num_columns.append(column)

            if len(missing_num_columns) > 0 :
                logging.info(f"Missing numerical column {missing_num_columns}")

            for column in self._config_schema['categorical_columns']:
                if column not in df_columns:
                    missing_cate_columns.append(column)

            if len(missing_cate_columns)>0:
                logging.info(f"Missing categorical coloum {missing_cate_columns}")

            return False if len(missing_num_columns) > 0 or len(missing_cate_columns) > 0 else True
        except Exception as e:
            raise CustomException(e,sys)
        
    @staticmethod
    def read_data(file_path)-> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise CustomException(e,sys)
        
    def detect_data_drift(self,refrence_df:pd.DataFrame , current_df : pd.DataFrame) -> bool:
        """
        This method wil examine and detect the data drift and provide the report.
        """
        try:
            data_drift_profile = Report([DataDriftPreset() , DataSummaryPreset()])
            logging.info("Report Object created.")

            data_drift_profile.run(reference_data=refrence_df , current_data=current_df)

            report = data_drift_profile.json()
            json_report = json.loads(report)

            write_yaml_file(file_path=self.data_val_report_path , content=json_report) 

            n_features = json_report["data_drift"]["data"]["metrics"]["n_features"]
            n_drifted_features = json_report["data_drift"]["data"]["metrics"]["n_drifted_features"]

            logging.info(f"{n_drifted_features}/{n_features} drift detected.")
            drift_status = json_report["data_drift"]["data"]["metrics"]["dataset_drift"]
            return drift_status 
                  
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_validation(self,) -> DataValidationArtifact:
        """
        This method will initiatie the data validation.
        """
        try:
            logging.info("Data Validiation initiated.")
            train_data,test_data = (DataValidation.read_data(file_path=DataIngestionArtifact().train_path) 
                                    ,DataValidation.read_data(file_path=DataIngestionArtifact().test_path))
            
            validation_error_msg =""

            status = self.check_columns_len(train_data)
            logging.info(f"All required columns present in training dataframe: {status}")
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe."
            status = self.check_columns_len(test_data)

            logging.info(f"All required columns present in testing dataframe: {status}")
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe."

            status = self.check_req_columns(df=train_data)

            if not status:
                validation_error_msg += f"Columns are missing in training dataframe."
            status = self.check_req_columns(df=test_data)

            if not status:
                validation_error_msg += f"columns are missing in test dataframe."

            validation_status = len(validation_error_msg) == 0

            if validation_status:
                drift_status = self.detect_data_drift(refrence_df=train_data , current_df=test_data)
                if drift_status:
                    logging.info(f"Data Drift Detected.")
                    validation_error_msg = "Data Drift Detected."

                else :
                    validation_error_msg = "Data Drift Not Detected"

            data_validation_artifact = DataValidationArtifact(status=validation_status , data_drift_report=DataValidationConfig().DATA_VALIDATION_REPORT)
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            CustomException(e,sys)
    
        
    
        
    