import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder ,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utlis import save_object

@dataclass
class Data_trasformation_config():
    preprocessor_obj_file_path= os.path.join('artifacts',"preprocessor.pkl")

class DataTrasformation:
    def __init__(self):
        self.preprocessing_config_file=Data_trasformation_config()

    def get_data_trasformation(self):
        try:
            numrical_columns=["Amount"]

            num_pipeline=Pipeline(
                steps=[
                    
                       ("imputer",SimpleImputer(strategy="median")),
                        ("scaler",StandardScaler(with_mean=False))
                    
                ]
            )

            # cat_piprline=Pipeline(
            #     steps=[
            #         ("onehotencoding",OneHotEncoder)
            #     ]
            # )

            logging.info("Numerical and categorical pipelines created")

            preprocessing=ColumnTransformer(
                [
                    ("num_pipelines",num_pipeline,numrical_columns)
                ],
                remainder="passthrough"
            )

            return preprocessing
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_data,test_data):
        try:
            train_data_df=pd.read_csv(train_data)
            test_data_df=pd.read_csv(test_data)
            logging.info("Read train and test data completed ")

            preprocessing_obj=self.get_data_trasformation()

            target_column_name="Class"

            #spliting data

            X_train=train_data_df.drop(columns=[target_column_name])
            y_train= train_data_df[target_column_name]

            X_test = test_data_df.drop(columns=[target_column_name])
            y_test = test_data_df[target_column_name]

           
            logging.info("Applying preprocessing on training and testing dataset")

            model_fit_train_df=preprocessing_obj.fit_transform(X_train)
            model_fit_tst_df=preprocessing_obj.transform(X_test)

           
            save_object(
                file_path=self.preprocessing_config_file.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("preprocessing saved successfully")

            return (
            model_fit_train_df,
            y_train.values.ravel(),
            model_fit_tst_df,
            y_test.values.ravel()
                )


            
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    from src.Components.data_ingestion import DataIngestion

    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    data_transformation = DataTrasformation()
    X_train, y_train, X_test, y_test = data_transformation.initiate_data_transformation(
        train_path,
        test_path
    )

    print("Data Transformation Completed")
    print("X_train shape:", X_train.shape)
