from dataclasses import dataclass
import os

#DATA INGESTION CONFIG
@dataclass
class DataIngestionConfig:
    DATASET_PATH = os.path.join("Artifact","Dataset")
    TEST_PATH = os.path.join("ARTIFACT","test.csv")
    TRAIN_PATH = os.path.join("ARTIFACT","train.csv")