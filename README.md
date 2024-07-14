# Loan Approval Prediction Analysis

## Overview
This project analyzes the Loan Approval Prediction dataset from Kaggle to build a K-Nearest Neighbors (KNN) model for predicting loan approval status. The analysis includes data cleaning, exploratory data analysis (EDA), outlier detection and treatment, and model training and evaluation.

## About the Dataset
  - The Loan Approval Prediction Dataset, sourced from Kaggle, contains information about loan applicants and their loan approval status. Key features include:
  -	Demographic information: Number of dependents, education level, self-employment status
  -	Financial information: Annual income, loan amount requested, loan term
  -	Credit information: CIBIL score
  -	Asset information: Residential, commercial, and luxury asset values, bank asset value
  -	Target variable: Loan status (Approved/Rejected)
  -	The dataset consists of 4,269 entries with 13 features, providing a comprehensive view of factors potentially influencing loan approval decisions. This dataset is suitable for binary classification tasks, where the goal is to predict whether a loan application will be approved or rejected based on the given features.

## Dataset
The dataset used is the [Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset) from Kaggle.

## Project Structure
  - `loan_approval_analysis.ipynb`: Jupyter notebook containing the full analysis
  - `loan_approval_summary.docx`: Word document summarizing the results
  - `README.md`: This file, providing an overview of the project

## Key Steps
1. Data Cleaning
   - Checked for and handled missing values
   - Removed unnecessary spaces in column names and values
   - Dropped irrelevant columns

2. Exploratory Data Analysis (EDA)
   - Generated descriptive statistics
   - Analyzed correlations between features
   - Identified key patterns in the data

3. Outlier Detection and Treatment
   - Used IQR method to detect outliers
   - Applied capping to treat outliers

4. Data Splitting and Model Training
   - Manually split data into 80% training and 20% testing sets
   - Performed feature scaling
   - Trained KNN model and implemented cross-validation

5. Model Evaluation
   - Evaluated model using accuracy, precision, recall, F1-score, and ROC-AUC
   - Compared results from manual split with cross-validation

## Results
- KNN model performance:
  - Accuracy: 90.16%
  - Precision: 93.30%
  - Recall: 90.86%
  - F1-score: 0.9206
  - ROC-AUC: 0.8993
- Confusion Matrix:
  - True Positives: 283
  - False Positives: 35
  - False Negatives: 49
  - True Negatives: 487
- Cross-validation mean score: 0.9078

- Key factors influencing loan approval: CIBIL score, income, loan amount
-	Consistent results between manual split and cross-validation indicate model stability
-	CIBIL score is identified as a crucial factor in loan approval decisions
-	The model shows good balance between precision and recall, as evidenced by the high F1-score
-	The K-Nearest Neighbors (KNN) model developed for loan approval prediction demonstrates strong performance and reliability:
-	With an accuracy of 90.16%, the model shows excellent overall predictive capability.
-	High precision (93.30%) and recall (90.86%) indicate the model's effectiveness in both approving worthy candidates and identifying potential defaults.
-	An ROC-AUC score of 0.8993 suggests the model's strong ability to distinguish between approved and rejected loan applications.
-	Cross-validation scores (mean 0.9078) demonstrate the model's stability across different data subsets, indicating good generalizability.
-	The analysis highlighted CIBIL score, income, and loan amount as crucial factors in loan approval decisions.
-	The model's performance suggests it could be a valuable tool in assisting loan approval decisions in real-world scenarios.

## Tools Used
- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## How to Run
1. Clone this repository
2. Install required packages: `pip install -r requirements.txt`
3. Open and run `loan_approval_analysis.ipynb` in Jupyter Notebook or JupyterLab

## Author
[Sarah Rafiq Shaikh]

