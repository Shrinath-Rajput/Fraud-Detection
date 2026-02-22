import sys
import os

from src.exception import CustomException
from src.logger import logging
from src.utlis import save_object

from src.Components.data_ingestion import DataIngestion
from src.Components.data_transformation import DataTrasformation
from src.Components.model_trainer  import ModelTrainer

def main():
    try:
        logging.info("================Training Pipeline is Started =====================")

        #Data Ingstion
        data_ingestion=DataIngestion()
        train_data_path,Test_data_path=data_ingestion.initiate_data_ingestion()

        #DataTrasformation

        data_trasformation=DataTrasformation()
        X_train, y_train, X_test, y_test = data_trasformation.initiate_data_transformation(train_data_path,
        Test_data_path)

        #model training
        model_trainer1 = ModelTrainer()
        model_trainer=model_trainer1.initiate_model_trainer(X_train,y_train,X_test,y_test)
    except Exception as e:
        raise CustomException(e,sys)

if __name__=="__main__":
    main()        


