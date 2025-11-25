
import os
import sys
import numpy as np
import pandas as pd
import pymongo
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import certifi

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact

load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
ca = certifi.where()


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            logging.info(
                f"Initialized DataIngestion with DB: "
                f"{self.data_ingestion_config.database_name}, "
                f"collection: {self.data_ingestion_config.collection_name}"
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """
        Read data from MongoDB collection into a pandas DataFrame.
        """
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            logging.info(f"Using DB: {database_name}")
            logging.info(f"Using collection: {collection_name}")

            client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
            collection = client[database_name][collection_name]

            docs = list(collection.find())
            logging.info(f"Number of docs fetched from Mongo: {len(docs)}")

            if len(docs) == 0:
                # Return empty DataFrame so we can handle this gracefully later
                return pd.DataFrame()

            df = pd.DataFrame(docs)

            if "_id" in df.columns:
                df = df.drop(columns=["_id"])

            df.replace({"na": np.nan}, inplace=True)
            return df

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_data_intofeature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Save the full dataframe into the feature-store CSV.
        """
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Saving feature-store file to: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        """
        Split dataframe into train/test CSV files.
        """
        try:
            if dataframe.shape[0] == 0:
                raise ValueError(
                    "No rows in dataframe – cannot perform train/test split."
                )

            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42,
            )

            logging.info("Performed train/test split on the dataframe")

            train_dir = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(train_dir, exist_ok=True)

            logging.info(
                f"Exporting train file to: {self.data_ingestion_config.training_file_path}"
            )
            logging.info(
                f"Exporting test file to:  {self.data_ingestion_config.testing_file_path}"
            )

            # NOTE: train goes to training_file_path, test to testing_file_path
            train_set.to_csv(
                self.data_ingestion_config.training_file_path,
                index=False,
                header=True,
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path,
                index=False,
                header=True,
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion process")

            dataframe = self.export_collection_as_dataframe()

            if dataframe.shape[0] == 0:
                raise ValueError(
                    "MongoDB collection returned 0 documents. "
                    "Check database/collection names and that data is actually present."
                )

            dataframe = self.export_data_intofeature_store(dataframe)
            self.split_data_as_train_test(dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )

            logging.info(f"Data ingestion completed: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)