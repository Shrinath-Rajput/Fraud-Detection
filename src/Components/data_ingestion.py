import os
import sys

from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# from src.Components.data_transformation import DataTrasformation
# from src.Components.data_transformation import DataTrasformationConfig
# from src.Components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts',"train.csv")
    test_data_path:str=os.path.join('artifacts',"test.csv")
    Row_data_path:str=os.path.join('artifacts',"Row.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config_data=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Enter the datasets and method that points")
        try:
            df = pd.read_csv('.\\Dataset\\creditcard.csv')


            logging.info("Read the dataset here")

            os.makedirs(os.path.dirname(self.ingestion_config_data.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config_data,index=False,header=True)
            logging.info("Data read completely")

            #train and test model

            Train_data,test_data=train_test_split(df,random_state=42,test_size=0.2)
            Train_data.to_csv(self.ingestion_config_data)
            test_data.to_csv(self.ingestion_config_data)


            return (

                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as ex:
            raise CustomException(ex,sys)

        


