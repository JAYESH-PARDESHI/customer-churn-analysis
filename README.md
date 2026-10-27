# Customer Churn Analysis & Prediction

## Project Overview

Customer churn is a major business challenge for subscription-based companies. Identifying customers who are likely to discontinue a service can help businesses develop better customer-retention strategies.

This project analyzes customer demographics, service subscriptions, contract types, payment methods, and billing information to identify patterns associated with customer churn. It also develops and evaluates machine learning classification models to predict whether a customer is likely to leave.

## Objectives

- Understand customer churn patterns through exploratory data analysis (EDA).
- Identify customer characteristics associated with higher churn rates.
- Clean and preprocess numerical and categorical data.
- Build a reproducible machine learning preprocessing pipeline.
- Train and evaluate Logistic Regression and XGBoost classifiers.
- Compare models using accuracy, precision, recall, F1-score, and ROC-AUC.

## Dataset

The project uses the Telco Customer Churn dataset, which contains **7,043 customer records and 21 columns**, including customer information, subscribed services, billing details, tenure, and churn status.

- **Target variable:** `Churn`
- **Classification task:** Binary classification (`Yes` / `No`)
- **Records:** 7,043
- **Original columns:** 21

The `customerID` column is excluded from the model features because it serves as a customer identifier rather than a predictive business feature.

Dataset files are not included in this repository by default. See `data/README.md` for dataset access and setup instructions.

## Project Workflow

### 1. Data Understanding and Cleaning
- Inspected dataset dimensions, column types, and missing values.
- Converted `TotalCharges` from text to a numeric data type.
- Identified 11 missing `TotalCharges` values associated with zero-tenure customers.
- Investigated missing data before applying imputation.

### 2. Exploratory Data Analysis
- Analyzed the distribution of churn and non-churn customers.
- Examined customer tenure, monthly charges, and total charges.
- Compared churn rates across contract types and payment methods.
- Used visualizations and grouped statistics to identify important patterns.

### 3. Key Findings
- Approximately **26.54%** of customers in the dataset churned.
- Customers who churned had a median tenure of **10 months**, compared with **38 months** for customers who did not churn.
- Month-to-month contracts had a churn rate of approximately **42.71%**.
- Two-year contracts had a churn rate of approximately **2.83%**.
- Customers paying by electronic check had a churn rate of approximately **45.29%**.

These findings describe associations in the dataset; they do not establish that a particular service or payment method causes churn.

### 4. Data Preprocessing
- Separated features and target variable.
- Excluded the customer identifier from model inputs.
- Split data into training and test sets using an 80:20 ratio with stratification.
- Applied median imputation and standard scaling to numerical features.
- Applied most-frequent imputation and one-hot encoding to categorical features.
- Used Scikit-learn pipelines to keep preprocessing and modelling steps reproducible and reduce data leakage.

### 5. Model Development and Evaluation
Models evaluated so far:

- Logistic Regression — baseline model and selected candidate.
- XGBoost Classifier — tuned using RandomizedSearchCV with five-fold stratified cross-validation.

Performance was assessed using accuracy, precision, recall, F1-score, ROC-AUC, classification reports, and confusion matrices.

## Baseline Model Results

Results below are from the held-out test set.

| Metric | Logistic Regression | Tuned XGBoost |
|---|---:|---:|
| Accuracy | 80.55% | 80.91% |
| Precision (Churn) | 65.72% | 68.04% |
| Recall (Churn) | 55.88% | 52.94% |
| F1-score (Churn) | 60.40% | 59.55% |
| ROC-AUC | 84.19% | 84.66% |

### Model Selection

Logistic Regression was selected as the preferred model candidate because it achieved better churn recall and F1-score than the tuned XGBoost model in the current evaluation. It also provides a relatively interpretable baseline.

The results highlight a trade-off between identifying more customers likely to churn and reducing false-positive predictions. Further threshold analysis and validation may improve the balance depending on business priorities.

**Note:** The Logistic Regression figures above represent its baseline configuration. Any subsequent hyperparameter-tuning results should be reported separately after evaluation.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook
- Git and GitHub

## Project Structure

```text
customer-churn-analysis/
├── data/
│   └── README.md
├── notebooks/
│   └── 01_customer_churn_analysis.ipynb
├── reports/
│   └── figures/
├── src/
│   └── README.md
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Run the Project

1. Clone the repository.
2. Install the required dependencies.
3. Obtain the dataset and place it in the location specified in `data/README.md`.
4. Open the notebook in Jupyter.
5. Run the notebook cells in order.

Example setup:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd customer-churn-analysis

python -m venv .venv
```

Activate the environment, then run:

```bash
pip install -r requirements.txt
jupyter notebook
```

## Future Improvements

- Tune Logistic Regression hyperparameters and evaluate the tuned model.
- Analyze the precision-recall trade-off using different classification thresholds.
- Compare models using cross-validation and appropriate churn-focused metrics.
- Interpret feature contributions and model coefficients.
- Develop a simple interface for predicting churn risk on new customer records.

## Author

**Your Name**

Aspiring Machine Learning / Data Analytics Professional

GitHub: [Your GitHub Profile URL](https://github.com/)

LinkedIn: [Your LinkedIn Profile URL](https://www.linkedin.com/)
