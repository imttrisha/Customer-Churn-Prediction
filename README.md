# Customer Churn Prediction

This project aims to predict customer churn using machine learning algorithms. Customer churn refers to when customers stop doing business with a company. The dataset used in this project includes various customer features such as demographics, account information, and services they are subscribed to.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)

## Project Overview
The goal of this project is to predict whether a customer will churn (stop using the service) based on the available features in the dataset. We utilize machine learning algorithms to achieve this, and also perform exploratory data analysis, feature scaling, and hyperparameter tuning.

## Technologies Used
- Python 3.12
- Libraries:
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn
  - Scikit-learn
  - RandomForestClassifier

## Dataset
The dataset contains customer information such as:
- `customerID`: Unique ID for each customer
- `gender`: Gender of the customer
- `SeniorCitizen`: Whether the customer is a senior citizen or not
- `tenure`: Number of months the customer has been with the company
- `PhoneService`: Whether the customer has a phone service
- `Churn`: Whether the customer has churned or not (Target variable)

## Exploratory Data Analysis (EDA)
We performed the following steps during EDA:
- Checked for class imbalance in the `Churn` column.
- Visualized the correlation between the features using a heatmap.
- Identified any missing or incorrect data and cleaned it.
- Plotted the distribution of key features.

## Feature Engineering
We encoded categorical variables using `LabelEncoder` and ensured that all features were numerical so they could be used in machine learning models.

## Model Building
We used the `RandomForestClassifier` as the primary model. The data was split into training (80%) and testing (20%) sets, and the feature values were standardized using `StandardScaler`.

Model building steps:
1. Split the data into train and test sets using `train_test_split`.
2. Standardized the feature values using `StandardScaler`.
3. Trained the `RandomForestClassifier` model with `n_estimators=100`.

## Model Evaluation
The model's performance was evaluated using:
- Accuracy score
- Classification report (precision, recall, F1-score)
- Confusion matrix
- Feature importance plot

