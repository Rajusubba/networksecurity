import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models,
)
from networksecurity.utils.ml_utils.metric.classification_metric import (
    get_classification_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)

import mlflow


# ------------------------------------------------------------------
# OPTIONAL: Enable DagsHub ONLY when explicitly requested
# ------------------------------------------------------------------
def init_dagshub_if_enabled():
    """
    Enable DagsHub + MLflow tracking ONLY when ENABLE_DAGSHUB=true.
    Prevents OAuth crashes in Docker/EC2.
    """
    if os.getenv("ENABLE_DAGSHUB", "false").lower() == "true":
        import dagshub
        dagshub.init(
            repo_owner="Rajusubba",
            repo_name="networksecurity",
            mlflow=True,
        )
        logging.info("DagsHub MLflow tracking ENABLED")
    else:
        logging.info("DagsHub tracking DISABLED")


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, train_metric, test_metric):
        """
        Logs metrics only (no model logging – compatible with DagsHub).
        """
        with mlflow.start_run():
            mlflow.log_metric("train_f1", train_metric.f1_score)
            mlflow.log_metric("train_precision", train_metric.precision_score)
            mlflow.log_metric("train_recall", train_metric.recall_score)

            mlflow.log_metric("test_f1", test_metric.f1_score)
            mlflow.log_metric("test_precision", test_metric.precision_score)
            mlflow.log_metric("test_recall", test_metric.recall_score)

            logging.info("Metrics logged to MLflow")

    def train_model(self, x_train, y_train, x_test, y_test) -> ModelTrainerArtifact:
        try:
            # ✅ Enable MLflow/DagsHub only if requested
            init_dagshub_if_enabled()

            logging.info("Starting model training")

            models = {
                "RandomForest": RandomForestClassifier(verbose=1),
                "DecisionTree": DecisionTreeClassifier(),
                "GradientBoosting": GradientBoostingClassifier(verbose=1),
                "LogisticRegression": LogisticRegression(max_iter=1000),
                "AdaBoost": AdaBoostClassifier(),
            }

            params = {
                "DecisionTree": {
                    "criterion": ["gini", "entropy"],
                },
                "GradientBoosting": {
                    "learning_rate": [0.1, 0.01],
                    "n_estimators": [50, 100],
                },
                "AdaBoost": {
                    "learning_rate": [0.1, 0.01],
                    "n_estimators": [50, 100],
                },
            }

            model_report = evaluate_models(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            logging.info(
                f"Best model: {best_model_name} "
                f"Score: {model_report[best_model_name]}"
            )

            # Metrics
            train_pred = best_model.predict(x_train)
            test_pred = best_model.predict(x_test)

            train_metric = get_classification_score(y_train, train_pred)
            test_metric = get_classification_score(y_test, test_pred)

            self.track_mlflow(train_metric, test_metric)

            # Load preprocessor
            preprocessor = load_object(
                self.data_transformation_artifact.transformed_object_file_path
            )

            # Save wrapped model
            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_file_path),
                exist_ok=True,
            )

            network_model = NetworkModel(
                preprocessor=preprocessor,
                model=best_model,
            )

            save_object(
                self.model_trainer_config.trained_model_file_path,
                network_model,
            )

            # ✅ Deployment artifacts
            os.makedirs("final_models", exist_ok=True)
            save_object("final_models/model.pkl", best_model)
            save_object("final_models/preprocessor.pkl", preprocessor)

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_model_file_path=train_metric,
                test_model_file_path=test_metric,
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            return self.train_model(
                x_train=train_arr[:, :-1],
                y_train=train_arr[:, -1],
                x_test=test_arr[:, :-1],
                y_test=test_arr[:, -1],
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys)