from dataclasses import dataclass
import os
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    ARTIFACT_DIR:str = "Artifact"
    timestamp: str = TIMESTAMP
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    
#DATA INGESTION CONFIG
trainingpieplineartifact = TrainingPipelineConfig().artifact_dir

@dataclass
class DataIngestionConfig:
    DATA_INGESTION_PATH = os.path.join(trainingpieplineartifact,"DATA_INGESTION")
    TEST_PATH = os.path.join(DATA_INGESTION_PATH,"Splited_data")
    TRAIN_PATH = os.path.join(DATA_INGESTION_PATH,"Splited_data")

@dataclass
class DataValidationConfig:
    DATA_VALIDATION_PATH = os.path.join(trainingpieplineartifact,"DATA_VALIDATION")
    DATA_VALIDATION_REPORT = os.path.join(DATA_VALIDATION_PATH,"Report")

@dataclass
class DataTransfromationConfig:
    DATA_TRANSFOMATION_PATH = os.path.join(trainingpieplineartifact,"DATA_TRANSFORMATION")
    DATA_TRANSFOMATION_OBJ = os.path.join(DATA_TRANSFOMATION_PATH,"transfomation.pkl")
    TRANSFORMED_TRAIN_DATA_PATH = os.path.join(DATA_TRANSFOMATION_PATH,"TRANSFORMED","transformed_train.csv")
    TRANSFORMED_TEST_DATA_PATH = os.path.join(DATA_TRANSFOMATION_PATH,"TRANSFORMED","transformed_test.csv")
