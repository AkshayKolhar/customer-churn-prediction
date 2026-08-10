# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecom customer is likely to churn.

## 📌 Project Overview

Customer churn prediction helps companies identify customers who are likely to leave so that appropriate retention strategies can be applied.

This project builds a complete machine learning pipeline that:

- Loads and cleans customer data
- Handles missing and inconsistent values
- Separates features and target
- Performs train-test splitting
- Identifies numerical and categorical features
- Scales numerical features
- Applies one-hot encoding to categorical features
- Trains a Logistic Regression model
- Evaluates model performance
- Uses cross-validation
- Tunes the classification threshold
- Saves the complete preprocessing and model pipeline
- Predicts churn probability for new customers
- Provides a Streamlit web application

---

## 📊 Dataset

This project uses the **Telco Customer Churn dataset**.

The target variable is:

- `Yes` → Customer churned
- `No` → Customer did not churn

The dataset contains customer demographic, service, contract, and billing information.

### Main Features

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

## 🔄 Machine Learning Workflow

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Train / Test Split
     ↓
Feature Identification
     ↓
Numerical Feature Scaling
     ↓
Categorical One-Hot Encoding
     ↓
Logistic Regression
     ↓
Model Evaluation
     ↓
Cross-Validation
     ↓
Threshold Tuning
     ↓
Final Pipeline
     ↓
Save Pipeline
     ↓
New Customer Prediction
     ↓
Streamlit Application


