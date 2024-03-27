import sys, os 
import numpy as np 
import pandas as pd 
from zipfile import Path

from src.constant import *
from src.exception import CustomException
from src.logger import logging 
from src.utils.utils import MainUtils
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    raw_data_dir:str = os.path.join(artifact_folder,"raw.csv")
    #data_ingestion_dir: str = os.path.join(artifact_folder)
    #raw_data_path: str = os.path.join(data_ingestion_dir,"card_data.csv")


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion Method starts')

        try:
            df = pd.read_csv(os.path.join("notebook/data",'raw_data.csv'))
            logging.info('dataset read as pandas dataframe')

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_dir),exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_dir , index=False)
            logging.info('Raw Data is created')
            
            return (self.data_ingestion_config.raw_data_dir)
            
            

        except Exception as e:
            raise CustomException(e,sys)