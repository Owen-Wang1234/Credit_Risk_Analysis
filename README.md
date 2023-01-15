# Credit_Risk_Analysis

## Project Overview
The client is seeking a reliable method to quickly predict the credit risk of potential loan candidates based on some prior historical data about them. The plan is to develop supervised machine learning models to evaluate credit risk based on credit card data from LendingClub. The ideal model should take a large group of candidates and determine their credit risk in an accurate and precise manner.

## Resources

### Data Sources

- LoanStats_2019Q1.csv
- credit_risk_resampling.ipynb
- credit_risk_ensemble.ipynb

### Software
The supervised machine learning models are developed in Python within the machine learning environment (mlenv):

- Python 3.7.15
- Jupyter Notebook 6.5.2
- Pandas 1.3.5
- Numpy 1.21.5
- scikit-learn 1.0.2
- imbalanced-learn 0.10.1

## Developing Machine Learning Models

### ETL - LendingClub Credit Data
Prior to loading the data, the column names are established, and the target value is identified. The DataFrame is loaded with the input data, and null columns and rows are dropped. The focus is the credit risk of potential loans, so those that already have the loans issued are filtered out. The interest rate is converted into a floating numeric type, and the loan status (the target parameter) is recategorized between two values: "low risk" (from "Current") and "high risk" (status is not "Current"). The DataFrame then has the index reset.

Before the data can be split between training and testing groups, any other non-target columns with categorical non-numeric types must be converted to numeric types, so those are identified. Pandas can then handle the conversion by using the `get_dummies` method for binary encoding. The result moves the target column to `y` and leaves the rest to `X`. A quick check shows that the data contains **68,470 low risk candidates and 347 high risk candidates** prior to splitting between training and testing. The splitting is done with the `random_state` seed set to `1` to ensure reproducible results. This same `random_state` seed of `1` is also used in every sampling and Logistic Regression modeling methods.

### Resampling Methods
The data in use is heavily unbalanced; approximately 0.5% of the data has one outcome, and the remaining 99.5% has the other outcome. In order to properly train the Logistic Regression model, the training data must be resampled towards more balanced proportions.

#### 1 - Naive Random Oversampling


#### 2 - Synthetic Minority Over-sampling TEchnique (SMOTE)


#### 3 - Cluster Centroids Undersampling


#### 4 - Combination (Over and Under) Sampling


### Ensemble Learning Methods