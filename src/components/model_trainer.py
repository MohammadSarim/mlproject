import os
import sys
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

import mlflow
import mlflow.sklearn
import dagshub

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

        # Initialize Dagshub integration for MLflow
        dagshub.init(repo_owner='mohdsarim8', repo_name='mlproject', mlflow=True)

    @staticmethod
    def eval_metrics(actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']},
                "Random Forest": {'n_estimators': [8, 16, 32, 64, 128, 256]},
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {'learning_rate': [0.1, 0.01, 0.05, 0.001], 'n_estimators': [8, 16, 32, 64, 128, 256]},
                "CatBoosting Regressor": {'depth': [6, 8, 10], 'learning_rate': [0.01, 0.05, 0.1], 'iterations': [30, 50, 100]},
                "AdaBoost Regressor": {'learning_rate': [0.1, 0.01, 0.5, 0.001], 'n_estimators': [8, 16, 32, 64, 128, 256]}
            }

            model_report, trained_models = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_name = max(model_report, key=lambda x: model_report[x]["test_score"])
            best_model_score = model_report[best_model_name]["test_score"]
            best_params = model_report[best_model_name]["best_params"]
            best_model = trained_models[best_model_name]
            
            print(f"Best Model Name: {best_model_name}")
            print(f"Best Parameters: {best_params}")

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best Model: {best_model_name} with R2 score: {best_model_score}")

            ### MLflow Tracking ###
            with mlflow.start_run():
                mlflow.log_param("model_name", best_model_name)
                
                # Ensure we always have at least one parameter to log
                if not best_params:
                    best_params = {'default_param': 1}  # Dummy parameter
                
                mlflow.log_params(best_params)

                predicted = best_model.predict(X_test)
                rmse, mae, r2 = self.eval_metrics(y_test, predicted)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                mlflow.sklearn.log_model(best_model, "model")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return r2_score(y_test, predicted)

        except Exception as e:
            raise CustomException(e, sys)