from utils.logger import logging
from utils.exception import CustomException

import os ,sys
from src.entity.config_entity import ModelTrainingConfig
from src.entity.artifact_entity import DataTransformationArtifact , ModelTrainigArtifact
from utils.common_utils import *

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score




class ModelTraining:
    def __init__(self, modeltrainingconfig: ModelTrainingConfig, datatransformationartifact: DataTransformationArtifact) -> ModelTrainigArtifact:
        self.modeltrainingconfig = modeltrainingconfig
        self.datatransformationartifact = datatransformationartifact

    def get_model(self):
        try:
            models = {
                "LinearRegression": (LinearRegression(), {"fit_intercept": [True, False]}),
                "DecisionTree": (DecisionTreeRegressor(), {"max_depth": [3, 5, 10, None]}),
                "KNN": (KNeighborsRegressor(), {"n_neighbors": [3, 5, 7, 9]}),
                "LinearSVR": (LinearSVR(), {"C": [0.1, 1, 10]})
            }
            return models
        except Exception as e:
            raise CustomException(e, sys)

    def get_best_model_parms(self, train_X, train_y):
        try:
            models = self.get_model()
            trained_models = {}
            train_scores = {}

            for model_name, (model, params) in models.items():
                logging.info(f"Training with GridSearchCV on {model_name}...")

                grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring="r2", n_jobs=-1)
                grid_search.fit(train_X, train_y)

                trained_models[model_name] = grid_search.best_estimator_
                train_scores[model_name] = grid_search.best_score_

                logging.info(f"{model_name} | Best Params: {grid_search.best_params_} | CV R2: {grid_search.best_score_}")

            return trained_models, train_scores
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_train(self):
        try:
            logging.info("Loading the transformed data...")

            train_data = load_numpy_array_data(file_path=self.datatransformationartifact.transformed_train)

            train_X, train_y = train_data[:, :-1], train_data[:, -1]

            trained_models, model_scores = self.get_best_model_parms(train_X=train_X, train_y=train_y)

            # Ensure directory exists
            os.makedirs(os.path.dirname(self.modeltrainingconfig.CANDIDATE_MODELS_PATH), exist_ok=True)

            # Save all candidate models
            for model_name, model in trained_models.items():
                model_path = os.path.join(self.modeltrainingconfig.CANDIDATE_MODELS_PATH, f"{model_name}.pkl")
                save_object(file_path=model_path, obj=model)
                logging.info(f"Saved {model_name} at {model_path}")

            # Save scores in JSON
            os.makedirs(os.path.dirname(self.modeltrainingconfig.CANDIDATE_MODELS_SCORE_PATH),exist_ok=True)
            save_object(self.modeltrainingconfig.CANDIDATE_MODELS_SCORE_PATH, model_scores)
            logging.info(f"Saved candidate model scores at {self.modeltrainingconfig.CANDIDATE_MODELS_SCORE_PATH}")

            return ModelTrainigArtifact(
                trained_model_path=self.modeltrainingconfig.CANDIDATE_MODELS_PATH,
                trained_model_score=model_scores
            )

        except Exception as e:
            raise CustomException(e, sys)
