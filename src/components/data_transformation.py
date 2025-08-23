from utils.exception import CustomException
from utils.logger import logging
from src.entity.config_entity import DataTransfromationConfig

from src.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact
import os,sys
from utils.common_utils import read_yaml_file ,save_numpy_array_data ,save_object

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder , OrdinalEncoder ,StandardScaler
from sklearn.pipeline import Pipeline

import pandas as pd
import numpy as np



class DataTransformation:
    def __init__(self , data_ingestion_artifact:DataIngestionArtifact,data_transformation_config:DataTransfromationConfig,data_validation_artifact:DataValidationArtifact)->DataTransformationArtifact:
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_transformation_config = data_transformation_config
        self.data_validation_artifact = data_validation_artifact
        self._schema = read_yaml_file("schema.yaml")

    def get_transformation_obj(self):
        """
        This method will return a data transfromatio obj.
        """
        try:
            logging.info("Data transfromation obj method started.")
            OHE = OneHotEncoder()
            OE = OrdinalEncoder()
            SS = StandardScaler()
            SI = SimpleImputer()

            ohe_columns = self._schema['oh_columns']
            oe_columns = self._schema['oe_columns']
            ss_column = self._schema['ss_columns']
            pl_columns = self._schema['si_oe_columns']
            logging.info("Columns extracted.")

            SI_OE_pipeline = Pipeline([("SimpleImputer",SimpleImputer(strategy="most_frequent")),
                           ("OrdinalEncoder",OrdinalEncoder())])
            
            preprocessor = ColumnTransformer([
                ("SI_OE_Pipeline",SI_OE_pipeline,pl_columns),
                ("OneHotEncoding",OneHotEncoder(),ohe_columns),
                ("OrdinalENcoder",OrdinalEncoder(),oe_columns),
                ("StandardScaler",StandardScaler(),ss_column)
            ],remainder = "passthrough"
            )
            logging.info("Column Transfromer created sucessfully.")

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    @staticmethod
    def read_data(filepath):
        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiat_data_transformation(self):
        """
        This method will initiate the data transformation.
        """
        try:
            logging.info("Data Transformation Initiated.")
            train_data = DataTransformation.read_data(self.data_ingestion_artifact.train_path)
            test_data = DataTransformation.read_data(self.data_ingestion_artifact.test_path)

            logging.info(f"train and test data loaded , train : {train_data.shape} test : {test_data.shape}")

            preproocessor = self.get_transformation_obj()

            input_train_feature = train_data.drop(columns=[self._schema['target_column']]+[self._schema['drop_columns']])
            target_train_feature = train_data[self._schema['target_column']]
            logging.info("Trained input and output feature seperated.")

            input_test_feature = test_data.drop(columns=[self._schema['target_column']]+[self._schema['drop_columns']])
            target_test_feature = test_data[self._schema['target_column']]
            logging.info("Trained input and output feature seperated.")

            transformed_input_train_feature = preproocessor.fit_transform(input_train_feature)
            transformed_input_test_feature = preproocessor.transform(input_test_feature)

            transformed_train_data = np.c_[
                transformed_input_train_feature , np.array(target_train_feature)
            ]

            transformed_test_data = np.c_[
                transformed_input_test_feature , np.array(target_test_feature)
            ]

            #saving obj and data,
            save_object(file_path=self.data_transformation_config.DATA_TRANSFOMATION_OBJ)
            save_numpy_array_data(file_path = self.data_transformation_config.TRANSFORMED_TRAIN_DATA_PATH , array = transformed_train_data)
            save_numpy_array_data(file_path = self.data_transformation_config.TRANSFORMED_TEST_DATA_PATH , array = transformed_test_data )
            

            data_transformation_artifact = DataTransformationArtifact(
                transform_obj_path = self.data_transformation_config.DATA_TRANSFOMATION_OBJ,
                transformed_train = self.data_transformation_config.TRANSFORMED_TRAIN_DATA_PATH,
                transformed_test = self.data_transformation_config.TRANSFORMED_TEST_DATA_PATH 
            )

            return data_transformation_artifact            

        except Exception as e:
            raise CustomException(e,sys)

        
