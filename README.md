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

