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
    def __init__(self , modeltrainingconfig : ModelTrainingConfig , datatransformationartifact : DataTransformationArtifact) -> ModelTrainigArtifact:
        self.modetrainingconfig = modeltrainingconfig
        self.datatransformationartifact = datatransformationartifact

    def get_model(self):
        """
        Define candidate models and their hyperparameter grids.
        """
        try:
            model = {
                "LinearRegression": (LinearRegression(), {"fit_intercept": [True, False]}),
                "DecisionTree": (DecisionTreeRegressor(), {"max_depth": [3, 5, 10, None]}),
                "KNN": (KNeighborsRegressor(), {"n_neighbors": [3, 5, 7, 9]}),
                "LinearSVR": (LinearSVR(), {"C": [0.1, 1, 10]})
            }

            return model
        except Exception as e:
            raise CustomException(e,sys)
        
    def get_best_model_parms(self,train_X ,train_y , test_X , test_y):
        """
        This method will get the best model and best parms.
        """
        try:
            models = self.get_model()
            best_model = None
            best_score = -float("inf")
            best_model_name = None

            for model_name , (model , parms) in models.items():
                logging.info(f"Model training runing on {model_name}")

                grid_search = GridSearchCV(estimator=model , param_grid=parms ,cv=5 ,scoring="r2",n_jobs=-1)
                grid_search.fit(train_X,train_y)

                model = grid_search.best_estimator_

                pred = model.predict(test_X)
                score = r2_score(y_true= test_y , y_pred= pred)
                logging.info(f"{model_name} | Best Params: {grid_search.best_params_} | Test R2: {score}")
                if score > best_score:
                    best_score = score 
                    best_model = model
                    best_model_name = model_name

            return best_model_name , best_model , best_score

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_model_train(self):
        """
        Initiate the model training
        """
        try:
            logging.info("Loading the transformed data.")

            train_data = load_numpy_array_data(file_path=self.datatransformationartifact.transformed_train)
            test_data = load_numpy_array_data(file_path=self.datatransformationartifact.transformed_test)

            train_X , train_y = train_data[:,:-1] ,train_data[:,-1]
            test_X , test_y = test_data[:,:-1] ,test_data[:,-1]

            best_model_name , best_model , best_score = self.get_best_model_parms(train_X=train_X,
                                                                                  train_y=train_y,
                                                                                  test_X=test_X,
                                                                                  test_y=test_y)

            model_save_path = os.makedirs(self.modetrainingconfig.TRAINED_MODEL_PATH)
            save_object(file_path=model_save_path , obj= best_model)
            logging.info(f"Model save in path {model_save_path}")
            
            save_score = {best_model_name : best_score}
            model_detial_path = os.makedirs(self.modetrainingconfig.TRAINED_MODEL_SCORE)
            save_object(file_path=model_detial_path , obj=save_score)
            logging.info(f"The best model is {best_model_name} with the r2 score of {best_score}")

            modeltrainerartifact = ModelTrainigArtifact(trained_model_path=model_save_path , trained_model_score=save_score)
            logging.info("Exiting the model training class.")
            return modeltrainerartifact
        except Exception as e:
            raise CustomException(e,sys)

