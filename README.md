# Credit_Risk_Analysis

## Project Overview
The client is seeking a reliable method to quickly predict the credit risk of potential loan candidates based on some prior historical data about them. The plan is to develop supervised machine learning models to evaluate credit risk based on credit card data from LendingClub. The ideal model should take a large group of candidates and quickly determine their credit risk in an accurate and precise manner.

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

Before the data can be split between training and testing groups, any other feature (non-target) columns with categorical non-numeric types must be converted to numeric types, so those are identified. Pandas can then handle the conversion by using the `get_dummies` method for binary encoding. Afterwards, the target column goes to `y` and the rest go to `X`. A quick check shows that the data set to be run through the machine learning models contains **68,470 low risk candidates and 347 high risk candidates** prior to splitting between training and testing. The splitting is done with the `random_state` seed set to `1` to ensure reproducible results if that seed is used. This same `random_state` seed of `1` is also used in every sampling and Logistic Regression modeling methods.

### Resampling Methods
The data in use is heavily unbalanced; approximately 0.5% of the data has one outcome (the minority class - in this project, the high risk candidates), and the remaining 99.5% has the other outcome (the majority class - the low risk candidates). In order to properly train the Logistic Regression model, the training data must be resampled towards more balanced proportions.

#### 1 - Naive Random Oversampling
Random Oversampling takes the minority set and randomly selects instances from the set to add to the training data until there is a balance between the majority and minority classes. After the training data is resampled accordingly, the Logistic Regression model is trained with the resampled data. The fitted model produces a set of predictions after feeding in the test data.

The actual test results and the predictions are compared to produce the balanced accuracy score, the confusion matrix, and the imbalanced classification report for evaluation of this resampling method.

#### 2 - Synthetic Minority Over-sampling TEchnique (SMOTE)
Synthetic Minority Over-sampling TEchnique (SMOTE) creates more instances artificially for the minority set by random acts of interpolation to add to the training data until there is a balance between the majority and minority classes. After the training data is resampled accordingly, the Logistic Regression model is trained with the resampled data. The fitted model produces a set of predictions after feeding in the test data.

The actual test results and the predictions are compared to produce the balanced accuracy score, the confusion matrix, and the imbalanced classification report for evaluation of this resampling method.

#### 3 - Cluster Centroids Undersampling
Cluster Centroids Undersampling works similarly to SMOTE in a different direction. This method identifies clusters of the majority class and generates synthetic data points (the centroids) to represent them. Then the majority class is undersampled (remove randomly selected instances) to match the size of the minority class. The amount of computing involved to do this makes this take minutes compared to the seconds required for the previous two methods. After the training data is resampled accordingly, the Logistic Regression model is trained with the resampled data. The fitted model produces a set of predictions after feeding in the test data.

The actual test results and the predictions are compared to produce the balanced accuracy score, the confusion matrix, and the imbalanced classification report for evaluation of this resampling method.

#### 4 - Combination (Over and Under) Sampling
SMOTE can be combined with Edited Nearest Neighbors (ENN) to form the SMOTEENN method that uses both oversampling and undersampling to rebalance the training data. SMOTE starts by oversampling the minority class. Then ENN undersamples to clean the results - if a data point has two nearest neighbors from different classes, that point is removed in order to provide a more clean separation between the two classes. The amount of computing involved to do this means this method also requires minutes rather than seconds as well. After the training data is resampled accordingly, the Logistic Regression model is trained with the resampled data. The fitted model produces a set of predictions after feeding in the test data.

The actual test results and the predictions are compared to produce the balanced accuracy score, the confusion matrix, and the imbalanced classification report for evaluation of this resampling method.

### Ensemble Learning Methods
When designing a machine learning model, one way to get a good performance is to combine multiple models to help improve accuracy and robustness while mitigating the variance (the ensemble learning method).

#### 1 - Balanced Random Forest Classifier
The Random Forest type of algorithm samples the data and runs multiple smaller and simpler decision trees. Each tree is based on a small random sample group used to train that tree; these trees would not be accurate individually, but the whole culmination should yield a better performance. Additionally, this product should be more robust against outliers and overfitting. The Balanced Random Forest Classifier algorithm, used in this project, is a variation that randomly undersamples each sample group for balancing purposes.

The Balanced Random Forest Classifier model is instantiated with mostly default settings; the estimator count (number of random forests) of one hundred and a random state of one are the only added inputs. Rather than resampling the training data, they can be directly used to fit the model. The fitted model produces a set of predictions after feeding in the test data.

The actual test results and the predictions are compared to produce the balanced accuracy score, the confusion matrix, and the imbalanced classification report for evaluation of this ensemble learning method.

Random Forest algorithms have one additional feature - they also calculate the feature importance, the amount of impact a feature has on the decision process. Thus, the list of features and the list of the calculated importance values are zipped together and then sorted by importance in descending order to see which features have the most impact on the credit risk evaluation process.

#### 2 - Easy Ensemble AdaBoost Classifier
The Boosting type of algorithm samples the data into multiple training sets which are run through their own learning model in sequence. Each model learns from the errors of the prior model, allowing them to develop a slightly better accuracy. Each individual prediction is combined to produce the ensemble prediction. The Adaptive Boosting (AdaBoost) method adds an extra step when training the next model; after the errors of the prior model are evaluated, those errors are given extra weight to reduce and minimize similar errors in subsequent models. The Easy Ensemble AdaBoost Classifier algorithm, used in this project, is a variation that "is an ensemble of AdaBoost learners trained on different balanced bootstrap samples. The balancing is achieved by random under-sampling" (taken from the Easy Ensemble documentation).

The Easy Ensemble model is instantiated with mostly default settings; the estimator count of one hundred and a random state of one are the only added inputs. Rather than resampling the training data, they can be directly used to fit the model. The fitted model produces a set of predictions after feeding in the test data.

The actual test results and the predictions are compared to produce the balanced accuracy score, the confusion matrix, and the imbalanced classification report for evaluation of this ensemble learning method.

### Results
After running several different types of supervised machine learning models for imbalanced data sets, the balanced accuracy score, the confusion matrix, and the imbalanced classification report are all collected for each model and displayed here:

1. NAIVE RANDOM OVERSAMPLING
![Performance Metrics of the Naive Random Oversampling Model](https://github.com/Owen-Wang1234/Credit_Risk_Analysis/blob/main/Images/RandomOverSampleResults.png)

2. SMOTE
![Performance Metrics of the SMOTE Model](https://github.com/Owen-Wang1234/Credit_Risk_Analysis/blob/main/Images/SMOTEResults.png)

3. CLUSTER CENTROIDS UNDERSAMPLING
![Performance Metrics of the Custer Centroids Model](https://github.com/Owen-Wang1234/Credit_Risk_Analysis/blob/main/Images/ClusterCentroidResults.png)

4. SMOTEENN
![Performance Metrics of the SMOTEENN Model](https://github.com/Owen-Wang1234/Credit_Risk_Analysis/blob/main/Images/SMOTEENNResults.png)

5. BALANCED RANDOM FOREST CLASSIFIER
![Performance Metrics of the Balanced Random Forest Model](https://github.com/Owen-Wang1234/Credit_Risk_Analysis/blob/main/Images/BalancedRandomForestResults.png)

An additional output available from random forest classifiers is the calculated importance of the features in the data set. The top 30 features based on their importance in determing the credit risk of the candidates are listed below:
![Top 30 Features According to the Random Forest](https://github.com/Owen-Wang1234/Credit_Risk_Analysis/blob/main/Images/Top30Features.png)

6. EASY ENSEMBLE ADABOOST CLASSIFIER
![Performance Metrics of the Easy Ensemble Model](https://github.com/Owen-Wang1234/Credit_Risk_Analysis/blob/main/Images/EasyEnsembleResults.png)

- The precision scores in the classification report focus on the columns of the confusion matrix. The values reflect the percentage of predictions that were accurate (how many candidates who are labeled as high/low risk truly are high/low risk). The precision scores are as follows:

| Learning Model | Recall (High Risk) | Recall (Low Risk) |
| --- | ---: | ---: |
| Random Oversampling | 0.01 | 1.00 |
| SMOTE | 0.01 | 1.00 |
| Cluster Centroids | 0.01 | 1.00 |
| SMOTEENN | 0.01 | 1.00 |
| Balanced Random Forest | 0.03 | 1.00 |
| Easy Ensemble AdaBoost | 0.09 | 1.00 |

- The recall scores in the classification report focus on the rows of the confusion matrix. The values reflect the percentage of the class that were correctly labeled (how many candidates who are high/low risk are correctly labeled as high/low risk). The recall scores are as follows:

| Learning Model | Precision (High Risk) | Precision (Low Risk) |
| --- | ---: | ---: |
| Random Oversampling | 0.71 | 0.60 |
| SMOTE | 0.63 | 0.69 |
| Cluster Centroids | 0.69 | 0.40 |
| SMOTEENN | 0.70 | 0.58 |
| Balanced Random Forest | 0.70 | 0.87 |
| Easy Ensemble AdaBoost | 0.92 | 0.94 |

- The balanced accuracy scores of these models are the averages of the calculated recall scores for the two classes. This helps to evaluate the overall performance of a model while mitigating the effects of working with imbalanced data sets. The balanced accuracy scores are as follows:

| Learning Model | Balanced Accuracy |
| --- | ---: |
| Random Oversampling | 0.6573 |
| SMOTE | 0.6622 |
| Cluster Centroids | 0.5447 |
| SMOTEENN | 0.6392 |
| Balanced Random Forest | 0.7885 |
| Easy Ensemble AdaBoost | 0.9317 |

### Summary