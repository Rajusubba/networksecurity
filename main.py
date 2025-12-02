from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    TrainingPipelineConfig,
)
from networksecurity.logging.logger import logging
import sys

if __name__ == '__main__':
    try:
        trainingpipelineconfig = TrainingPipelineConfig()

        # Data Ingestion
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        dataingestionartifact = data_ingestion.initiate_data_ingestion()
        logging.info("Initiate the data ingestion")
        print(dataingestionartifact)

        # Data Validation
        data_validation_config = DataValidationConfig(trainingpipelineconfig)
        data_validation = DataValidation(data_validation_config, dataingestionartifact)
        logging.info("Initiate the data validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation completed")

        print(data_validation_artifact)

    except Exception as e:
        raise NetworkSecurityException(e, sys)