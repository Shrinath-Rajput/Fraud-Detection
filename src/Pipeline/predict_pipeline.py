import os
import sys
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utlis import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            logging.info("Model and preprocessor loaded successfully")

            # Transform input
            data_scaled = preprocessor.transform(features)

            # Predict
            prediction = model.predict(data_scaled)

            return prediction

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    This class converts user input into dataframe
    """

    def __init__(
        self,
        Time,   # 👈 ADD THIS
        V1, V2, V3, V4, V5, V6, V7, V8, V9, V10,
        V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
        V21, V22, V23, V24, V25, V26, V27, V28,
        Amount
    ):
        self.Time = Time   # 👈 ADD THIS
        self.V1 = V1
        self.V2 = V2
        self.V3 = V3
        self.V4 = V4
        self.V5 = V5
        self.V6 = V6
        self.V7 = V7
        self.V8 = V8
        self.V9 = V9
        self.V10 = V10
        self.V11 = V11
        self.V12 = V12
        self.V13 = V13
        self.V14 = V14
        self.V15 = V15
        self.V16 = V16
        self.V17 = V17
        self.V18 = V18
        self.V19 = V19
        self.V20 = V20
        self.V21 = V21
        self.V22 = V22
        self.V23 = V23
        self.V24 = V24
        self.V25 = V25
        self.V26 = V26
        self.V27 = V27
        self.V28 = V28
        self.Amount = Amount

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "Time": [self.Time],   # 👈 ADD THIS
                "V1": [self.V1],
                "V2": [self.V2],
                "V3": [self.V3],
                "V4": [self.V4],
                "V5": [self.V5],
                "V6": [self.V6],
                "V7": [self.V7],
                "V8": [self.V8],
                "V9": [self.V9],
                "V10": [self.V10],
                "V11": [self.V11],
                "V12": [self.V12],
                "V13": [self.V13],
                "V14": [self.V14],
                "V15": [self.V15],
                "V16": [self.V16],
                "V17": [self.V17],
                "V18": [self.V18],
                "V19": [self.V19],
                "V20": [self.V20],
                "V21": [self.V21],
                "V22": [self.V22],
                "V23": [self.V23],
                "V24": [self.V24],
                "V25": [self.V25],
                "V26": [self.V26],
                "V27": [self.V27],
                "V28": [self.V28],
                "Amount": [self.Amount],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    data = CustomData(
        10000,   # 👈 Time add kar
        0.1, -1.2, 0.3, 0.5, -0.4, 0.2, -0.1, 0.3, 0.6, -0.7,
        0.4, -0.2, 0.1, -0.5, 0.6, -0.8, 0.2, 0.3, -0.1, 0.4,
        -0.6, 0.7, -0.2, 0.1, -0.3, 0.5, 0.2, -0.4,
        2000
    )

    df = data.get_data_as_dataframe()

    predict_pipeline = PredictPipeline()
    result = predict_pipeline.predict(df)

    print("Prediction:", result)

    if result[0] == 1:
        print("⚠️ Fraud Transaction")
    else:
        print("✅ Normal Transaction")