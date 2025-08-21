from dataclasses import dataclass
import os
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    ARTIFACT_DIR:str = "Artifact"
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP
#DATA INGESTION CONFIG
trainingpieplineartifact = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    DATA_INGESTION_PATH = os.path.join(trainingpieplineartifact,"DATA_INGESTION")
    TEST_PATH = os.path.join(DATA_INGESTION_PATH,"Splited_data","test.csv")
    TRAIN_PATH = os.path.join(DATA_INGESTION_PATH,"Splited_data","train.csv")

@dataclass
class DataValidationConfig:
    DATA_VALIDATION_PATH = os.path.join(trainingpieplineartifact,"DATA_VALIDATION")
    DATA_VALIDATION_REPORT = os.pardir.join(DATA_VALIDATION_PATH,"Report")
