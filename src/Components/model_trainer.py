import sys
import os
from dataclasses import dataclass

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.exception import CustomException
from src.logger import logging
from src.utlis import save_object


@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_model(self, y_true, y_pred, y_prob):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred),
            "ROC_AUC": roc_auc_score(y_true, y_prob),
        }

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=100)
            }

            best_model = None
            best_recall = 0
            best_model_name = ""

            mlflow.set_experiment("Fraud Detection Experiment")

            for model_name, model in models.items():

                with mlflow.start_run(run_name=model_name):

                    logging.info(f"Training Model: {model_name}")

                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]

                    metrics = self.evaluate_model(y_test, y_pred, y_prob)

                    # Log parameters
                    mlflow.log_param("model_name", model_name)

                    # Log metrics
                    for key, value in metrics.items():
                        mlflow.log_metric(key, value)

                    # Log model
                    mlflow.sklearn.log_model(model, model_name)

                    print(f"\nModel: {model_name}")
                    for key, value in metrics.items():
                        print(f"{key}: {round(value, 4)}")

                    if metrics["Recall"] > best_recall:
                        best_recall = metrics["Recall"]
                        best_model = model
                        best_model_name = model_name

            print(f"\nBest Model Selected: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.model_path,
                obj=best_model
            )

            logging.info("Best model saved successfully")

            return best_model_name

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.Components.data_ingestion import DataIngestion
    from src.Components.data_transformation import DataTrasformation

    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    transformation = DataTrasformation()
    X_train, y_train, X_test, y_test = transformation.initiate_data_transformation(
        train_path, test_path
    )

    trainer = ModelTrainer()
    best_model = trainer.initiate_model_trainer(
        X_train, y_train, X_test, y_test
    )

    print("\nTraining Completed Successfully")
