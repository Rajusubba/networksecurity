from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file

from scipy.stats import ks_2samp
import pandas as pd
import os
import sys


class DataValidation:
    def __init__(
        self,
        data_validation_config: DataValidationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
    ):
        """
        data_validation_config: paths where validated files & drift report should go
        data_ingestion_artifact: has train/test CSV paths from data ingestion step
        """
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact

            # load schema
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

            logging.info("Initialized DataValidation")
            logging.info(f"  Train file: {self.data_ingestion_artifact.trained_file_path}")
            logging.info(f"  Test file:  {self.data_ingestion_artifact.test_file_path}")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # ----------- helpers -----------

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            logging.info(f"Reading data from: {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Compare number of columns in df with schema.
        Assumes schema YAML has a 'columns' section.
        """
        try:
            # adjust this if your schema file structure is different
            expected_columns = self._schema_config.get("columns", {})
            number_of_columns = len(expected_columns)

            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Dataframe has columns: {len(dataframe.columns)}")

            return len(dataframe.columns) == number_of_columns
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> bool:
        """
        Returns True if drift is found in ANY column, False otherwise.
        Writes a YAML drift report.
        """
        try:
            drift_found_any = False
            report = {}

            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]

                ks_result = ks_2samp(d1, d2)
                p_value = ks_result.pvalue

                # drift if p_value < threshold
                drift_status = p_value < threshold
                if drift_status:
                    drift_found_any = True

                report[column] = {
                    "pvalue": float(p_value),
                    "drift_status": drift_status,
                }

            drift_report_file_path = self.data_validation_config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)

            write_yaml_file(file_path=drift_report_file_path, content=report)
            logging.info(f"Drift report written to: {drift_report_file_path}")

            return drift_found_any
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # ----------- main entrypoint -----------

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            # 1. Get train/test paths from ingestion artifact
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # 2. Read data
            train_dataframe = self.read_data(train_file_path)
            test_dataframe = self.read_data(test_file_path)

            # 3. Validate number of columns in train & test
            train_ok = self.validate_number_of_columns(train_dataframe)
            test_ok = self.validate_number_of_columns(test_dataframe)

            if not train_ok:
                logging.info("Train dataframe does not contain all required columns.")
            if not test_ok:
                logging.info("Test dataframe does not contain all required columns.")

            # If either fails, we still proceed but mark validation_status accordingly
            structure_ok = train_ok and test_ok

            # 4. Check data drift between train and test
            drift_found = self.detect_dataset_drift(
                base_df=train_dataframe, current_df=test_dataframe
            )

            # 5. Save valid train/test copies to validation folders
            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_validation_config.valid_test_file_path), exist_ok=True)

            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path, index=False, header=True
            )
            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path, index=False, header=True
            )

            # 6. Build artifact
            data_validation_artifact = DataValidationArtifact(
                validation_status=structure_ok and not drift_found,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

            logging.info(f"Data validation completed. Artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)