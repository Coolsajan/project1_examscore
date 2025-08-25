from utils.logger import logging
from utils.exception import CustomException
import os, sys
from utils.common_utils import load_object, load_numpy_array_data
import pandas as pd

from src.entity.artifact_entity import ModelTrainigArtifact, DataTransformationArtifact, ModelEvaluationArtifact
from src.entity.config_entity import ModelEvaluationConfig
from sklearn.metrics import r2_score


class ModelEvaluation:
    def __init__(self, modeltrainingartifact: ModelTrainigArtifact,
                 datatransformationartifact: DataTransformationArtifact,
                 modelevaluationconfig: ModelEvaluationConfig) -> ModelEvaluationArtifact:
        self.modelevalationconfig = modelevaluationconfig
        self.modeltrainingartifact = modeltrainingartifact
        self.datatransformationartifact = datatransformationartifact

    def model_eval_test(self, X_test, y_test, model_path):
        """
        Load models from directory and evaluate on test set.
        """
        try:
            test_report = []

            model_files = os.listdir(path=model_path)
            logging.info(f"Found {len(model_files)} models.")

            for file_name in model_files:
                full_model_path = os.path.join(model_path, file_name)
                model = load_object(full_model_path)
                model_name = os.path.splitext(file_name)[0]

                pred = model.predict(X_test)
                score = r2_score(y_test, pred)

                test_report.append({
                    "model_name": model_name,
                    "test_score": score
                })
                logging.info(f"Model {model_name} | R2 Score: {score:.4f}")

            return pd.DataFrame(test_report)

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_eval(self):
        """
        Evaluate models on test data and select the best one.
        """
        try:
            logging.info("Starting model evaluation...")

            test_array = load_numpy_array_data(file_path=self.datatransformationartifact.transformed_test)
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            logging.info(f"Test data split -> X_test: {X_test.shape}, y_test: {y_test.shape}")

            # Test report
            model_report = self.model_eval_test(X_test, y_test, model_path=self.modeltrainingartifact.trained_model_path)

            # Train report
            #train_report = load_object(file_path=self.modeltrainingartifact.trained_model_score)
            #train_report_df = pd.DataFrame(list(train_report.items()), columns=["model_name", "train_score"])

            # Merge reports
            #model_report = pd.merge(train_report_df, test_report_df, on="model_name")
            #model_report.sort_values(by="test_score", ascending=False, inplace=True)

            # Select best model
            best_model_name = model_report.iloc[0]["model_name"]
            best_model_path = os.path.join(self.modeltrainingartifact.trained_model_path, f"{best_model_name}.pkl")

            logging.info(f"Best model selected: {best_model_name} | Path: {best_model_path}")

            model_eval_artifact = ModelEvaluationArtifact(best_model_path=best_model_path)
            
            return model_eval_artifact

        except Exception as e:
            raise CustomException(e, sys)
