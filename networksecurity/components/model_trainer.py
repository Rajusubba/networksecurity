
import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object, load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data, evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import mlflow

import dagshub
dagshub.init(repo_owner='Rajusubba', repo_name='networksecurity', mlflow=True)


class ModelTrainer:
    def __init__(self,
                 model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def track_mlflow(self, best_model, train_metric, test_metric):
        """
        Log train & test metrics to MLflow (via DagsHub).
        We intentionally do NOT log the model with mlflow.sklearn.log_model
        because DagsHub's MLflow server does not support the new logged-model
        endpoint used by MLflow 3.x.
        """
        with mlflow.start_run():
            # Train metrics
            mlflow.log_metric("train_f1_score", train_metric.f1_score)
            mlflow.log_metric("train_precision_score", train_metric.precision_score)
            mlflow.log_metric("train_recall_score", train_metric.recall_score)

            # Test metrics
            mlflow.log_metric("test_f1_score", test_metric.f1_score)
            mlflow.log_metric("test_precision_score", test_metric.precision_score)
            mlflow.log_metric("test_recall_score", test_metric.recall_score)

            # ❌ DO NOT DO THIS with DagsHub + MLflow 3.x
            # mlflow.sklearn.log_model(best_model, "model")

            logging.info("Train & test metrics logged to MLflow (model not logged).")

    def train_model(self, x_train, y_train, x_test, y_test) -> ModelTrainerArtifact:
        """
        Train multiple models, select the best one, wrap it with NetworkModel,
        save it, and return a ModelTrainerArtifact.
        """
        try:
            logging.info("Starting model training and model selection.")

            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1, max_iter=1000),
                "AdaBoost": AdaBoostClassifier(),
            }

            params = {
                "Decision Tree": {
                    "criterion": ["gini", "entropy", "log_loss"],
                },
                "Random Forest": {
                    # You can add hyperparameters here if you want
                    # e.g. "n_estimators": [50, 100, 200]
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Logistic Regression": {
                    # Leave empty for now or add C, penalty, etc.
                },
                "AdaBoost": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            model_report: dict = evaluate_models(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            # best score
            best_model_score = max(model_report.values())

            # best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            # classification metrics
            y_train_pred = best_model.predict(x_train)
            
            classification_train_metric = get_classification_score(
                y_true=y_train, y_pred=y_train_pred
            )
            ##Tracking test metrics on mlflow
    
            
            
            
            y_test_pred = best_model.predict(x_test)
            classification_test_metric = get_classification_score(
                y_true=y_test, y_pred=y_test_pred
            )
            self.track_mlflow(best_model, classification_train_metric, classification_test_metric)
            # load preprocessor
            preprocessor = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )

            # ensure model dir exists
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            # wrap model + preprocessor
            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)

            # save the combined model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=network_model,
            )
            #save best model separately for deployment purpose
            save_object("final_models/model.pkl", best_model)

            # create and return artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_model_file_path=classification_train_metric,
                test_model_file_path=classification_test_metric,
            )

            logging.info(f"Model trainer artifact created: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Loading transformed train and test arrays for model training.")

            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # loading train and test array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact = self.train_model(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)