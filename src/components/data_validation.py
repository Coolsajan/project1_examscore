from utils.exception import CustomException
from utils.logger import logging
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact ,DataValidationArtifact
from utils.common_utils import read_yaml_file ,save_object
from scipy.stats import ks_2samp, chi2_contingency

import pandas as pd
import os, sys


class DataValidation():
    """
    This class will validate the data using EvidentlyAI
    """

    def __init__(self , data_validation_config :DataValidationConfig,data_ingestion_artifact:DataIngestionArtifact):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config
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
        
    
    def detect_drift(self,train_df, test_df, alpha=0.05):
        """
        This method will the detec is data drift is present or not.
        """
        try:
            drift_report = {}
            num_drifted_feat = 0
            drift_detected = False
            for col in train_df.columns:
                if pd.api.types.is_numeric_dtype(train_df[col]):
                    stat, p = ks_2samp(train_df[col], test_df[col])
                else:
                    contingency = pd.crosstab(train_df[col], test_df[col])
                    stat, p, _, _ = chi2_contingency(contingency)
                
                drift_report[col] = {
                    "p_value": p,
                    "drift_detected": p < alpha
                }
                if p < alpha :
                    num_drifted_feat += 1

            if num_drifted_feat > len(train_df.columns)*0.25:
                drift_detected = True

            return drift_report , drift_detected

                  
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_validation(self,) -> DataValidationArtifact:
        """
        This method will initiatie the data validation.
        """
        try:
            logging.info("Data Validiation initiated.")
            train_data,test_data = (DataValidation.read_data(file_path=self.data_ingestion_artifact.train_path) 
                                    ,DataValidation.read_data(file_path=self.data_ingestion_artifact.test_path))
            
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
                drift_report ,drift_status = self.detect_drift(train_df=train_data , test_df=test_data)
                os.makedirs(os.path.dirname(self.data_validation_config.DATA_VALIDATION_REPORT), exist_ok=True)
                save_object(file_path=self.data_validation_config.DATA_VALIDATION_REPORT, obj=drift_report)

                
                if drift_status:
                    logging.info(f"Data Drift Detected.")
                    validation_error_msg = "Data Drift Detected."

                else :
                    validation_error_msg = "Data Drift Not Detected"

            data_validation_artifact = DataValidationArtifact(status=validation_status , data_drift_report=self.data_validation_config.DATA_VALIDATION_REPORT)
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e,sys)