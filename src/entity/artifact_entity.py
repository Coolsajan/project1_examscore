from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_path : str
    test_path : str

@dataclass
class DataValidationArtifact:
    status : bool
    data_drift_report : str

@dataclass
class DataTransformationArtifact:
    transform_obj_path :str
    transformed_train :str
    transformed_test : str

@dataclass
class ModelTrainigArtifact:
    trained_model_path : str
    trained_model_score : str

@dataclass
class ModelEvaluationArtifact:
    best_model_path : str