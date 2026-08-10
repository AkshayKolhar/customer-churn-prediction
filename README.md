# Customer Churn Prediction

## Overview

Customer Churn Prediction is an end-to-end machine learning project that predicts whether a telecom customer is likely to churn.

The project uses customer demographic, service, contract, and billing information to calculate the probability of churn and generate a final `Yes` or `No` prediction.

The project demonstrates the complete machine learning workflow:

**Data → Preprocessing → Training → Evaluation → Validation → Threshold Tuning → Model Saving → Prediction → Streamlit Application**

---

## Dataset

The project uses the **Telco Customer Churn** dataset.

The target variable is:

- `Churn = Yes` → Customer churned
- `Churn = No` → Customer did not churn

The dataset contains customer information related to demographics, services, contracts, and billing.

---

## Features Used

The model uses customer information including:

- Customer ID
- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

---

## Machine Learning Workflow
            
            Telco Customer Churn Dataset
                        ↓
                  Data Cleaning
                        ↓
                 Feature Selection
                        ↓
                  Train-Test Split
                        ↓
               Feature Preprocessing
                        ↓
               ┌─────────────────────┐
               │ Numerical Features  │
               │    StandardScaler   │
               └─────────────────────┘
                        +
               ┌─────────────────────┐
               │ Categorical Features│
               │    OneHotEncoder    │
               └─────────────────────┘
                        ↓
                 ColumnTransformer
                        ↓
                Logistic Regression
                        ↓
                  Model Evaluation
                        ↓
                  Cross-Validation
                        ↓
                 Threshold Tuning
                        ↓
                 Final ML Pipeline
                        ↓
                  Save with Joblib
                        ↓
                New Customer Input
                        ↓
                  Churn Probability
                        ↓
                 Churn Prediction
                        ↓
                Streamlit Web App

## Preprocessing Workflow 

            Numerical Features
                    ↓
            StandardScaler
                    ↓
            Scaled Numerical Features
            
            Categorical Features
                    ↓
            OneHotEncoder
                    ↓
            Encoded Categorical Features
            
            Both
                    ↓
            ColumnTransformer

## Machine Learning Pipeline
            
            Raw Customer Data
                    ↓
            ColumnTransformer
                    ↓
             ┌──────────────────────┐
             │ Numerical Features   │
             │     StandardScaler   │
             └──────────────────────┘
                    +
             ┌──────────────────────┐
             │ Categorical Features │
             │     OneHotEncoder    │
             └──────────────────────┘
                    ↓
            Logistic Regression
                    ↓
            Churn Probability
                    ↓
            Final Prediction

## Model

The final model used in the project is Logistic Regression.

Logistic Regression is suitable for this binary classification problem because the target contains two possible outcomes:

Yes → Customer churn
No  → Customer does not churn

The model produces a probability of churn, which can then be converted into a final classification using a selected threshold.

## Model Evaluation

The model is evaluated using several classification metrics:

Accuracy
Precision
Recall
F1 Score
Confusion Matrix
Classification Report

## Classification Threshold

            Churn Probability
                    ↓
               Compare with
                Threshold
                    ↓
             ┌───────────────┐
             │ Probability   │
             │ >= Threshold  │
             └───────────────┘
                    ↓
               Prediction: Yes
            
            
             ┌───────────────┐
             │ Probability   │
             │ < Threshold   │
             └───────────────┘
                    ↓
               Prediction: No

## Model Serialization

After training, the complete machine learning pipeline is saved using Joblib.

The trained pipeline is stored at:

model/churn_prediction.pkl

## Streamlit Application

The project includes an interactive Streamlit web application.

The application allows users to enter customer information through a graphical interface and receive a churn prediction.

The application collects information such as:

            Customer ID
            Gender
            Senior Citizen
            Partner
            Dependents
            Tenure
            Phone Service
            Multiple Lines
            Internet Service
            Online Security
            Online Backup
            Device Protection
            Tech Support
            Streaming TV
            Streaming Movies
            Contract
            Paperless Billing
            Payment Method
            Monthly Charges
            Total Charges

After entering the customer information, the user can click Predict Churn.

The application then displays:
Churn Probability
Final Churn Prediction

Example:

            Churn Probability: 0.3034
            Prediction: Yes

## Project Structure 

            customer-churn-prediction/
            │
            ├── data/
            │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
            │
            ├── experiments_old/
            │   ├── decision_tree.py
            │   ├── k_nearest.py
            │   ├── log_reg.py
            │   ├── naive.py
            │   ├── random_forest.py
            │   └── SVM.py
            │
            ├── model/
            │   └── churn_prediction.pkl
            │
            ├── notebooks/
            │
            ├── app.py
            ├── predict.py
            ├── train_model.py
            ├── requirements.txt
            ├── .gitignore
            └── README.md

## 🛠️ Technologies Used

### Programming
- **Python** — Core programming language used to build the project.

### Data Processing
- **Pandas** — Used for loading, cleaning, transforming, and managing the dataset.
- **NumPy** — Used for numerical operations and data manipulation.

### Machine Learning
- **Scikit-learn** — Used for preprocessing, pipeline creation, Logistic Regression, model evaluation, cross-validation, and prediction.

### Model Serialization
- **Joblib** — Used to save and load the trained machine learning pipeline.

### Web Application
- **Streamlit** — Used to build the interactive customer churn prediction web application.

### Development Tools
- **Visual Studio Code** — Used as the development environment.
- **Git** — Used for version control.
- **GitHub** — Used to store and manage the project repository.

## 🧪 Example Prediction

### Example Customer Information

```text
Tenure: 12 months
Monthly Charges: 75.50
Total Charges: 906.00
Contract: Month-to-month
Internet Service: Fiber optic
Payment Method: Electronic check
```
##Example Output 
```text
Churn Probability: 0.3034
Prediction: Yes
```

## Machine Learning Concepts Demonstrated

- Binary Classification
- Logistic Regression
- Data Cleaning
- Feature Engineering
- Train-Test Split
- Standardization
- One-Hot Encoding
- ColumnTransformer
- Scikit-learn Pipeline
- Model Evaluation
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report
- Cross-Validation
- Probability Prediction
- Classification Threshold
- Model Serialization
- Joblib
- Streamlit Application Development
- Git and GitHub Project Management

## Author 
```bash
Akshay
AI/ML Engineering Student
```

## Project Status 
```text
Completed end-to-end customer churn prediction project with data preprocessing, model evaluation, cross-validation, threshold tuning, model persistence, customer prediction, and Streamlit deployment.
```



