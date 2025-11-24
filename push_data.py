
import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
print(f"MONGO_DB_URL: {MONGO_DB_URL}")

import certifi
ca = certifi.where()

import pandas as pd
import numpy as np
import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

class NetworkDataExtract():
    def __init__(self):
        try:
            # create Mongo client once
            self.mongo_client = pymongo.MongoClient(
                MONGO_DB_URL,
                tlsCAFile=ca   # needed for MongoDB Atlas
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def csv_to_json_convertor(self,file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def insert_data_to_mongodb(self, records, database_name,collection_name):
        try:
            db = self.mongo_client[database_name]
            # select collection
            collection = db[collection_name]

            # insert
            result = collection.insert_many(records)
            return len(result.inserted_ids)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
            
            
if __name__ == "__main__":
    FILE_PATH = "/Users/rajusubba/Documents/MLOpsProject/Network Security/Network_Data/phisingData.csv"
    DATABASE = "NetworkSecurity"
    Collection = "PhishingData"
    networkobj = NetworkDataExtract()
    records = networkobj.csv_to_json_convertor(file_path=FILE_PATH)
    print(records)
    no_of_records = networkobj.insert_data_to_mongodb(records, DATABASE, Collection)
    print(no_of_records)
