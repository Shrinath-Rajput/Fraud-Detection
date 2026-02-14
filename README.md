Credit Card Fraud Detection System
* Project Overview

This project aims to build a Machine Learning model that detects fraudulent credit card transactions.

The dataset contains anonymized transaction features and is highly imbalanced, making this a real-world classification challenge.

The goal is to accurately identify fraudulent transactions while minimizing false negatives.

* Dataset

Source: Kaggle
Dataset Name: Credit Card Fraud Detection

Dataset Features
4

284,807 transactions

30 input features (V1–V28, Amount)

Target column: Class

0 → Normal

1 → Fraud

# The dataset is highly imbalanced (fraud cases are very rare).

* Project Workflow
1 Data Ingestion

Load dataset using pandas

Analyze dataset shape and class distribution

2️ Exploratory Data Analysis (EDA)

Fraud vs Normal visualization

Correlation heatmap

Transaction amount analysis

3️ Data Preprocessing

Feature scaling using StandardScaler

Train-test split

Handling class imbalance using SMOTE

4️ Model Training

Logistic Regression

Random Forest

XGBoost

5️ Model Evaluation

Precision

Recall

F1 Score

ROC-AUC Score

Confusion Matrix

* Focus is on Recall, as missing fraud transactions is costly.

* Technologies Used

Python

NumPy

Pandas

Scikit-learn

XGBoost

Matplotlib / Seaborn

MLflow (for experiment tracking)

* Project Structure
Credit-Card-Fraud-Detection/
│
├── data/
│   └── creditcard.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│
├── models/
│   └── best_model.pkl
│
├── requirements.txt
└── README.md

# Results

Successfully handled imbalanced dataset using SMOTE

Achieved high recall for fraud detection

Compared multiple models and selected the best-performing one

# Future Improvements

Deploy model using Flask/FastAPI

Real-time fraud detection API

Dashboard visualization

Model monitoring

* Author

Shrinath Rajput
Machine Learning Enthusiast