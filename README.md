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

Data Preprocessing

The dataset contains both numerical and categorical features.

Categorical features are converted into numerical features using One-Hot Encoding.

Numerical features are scaled using StandardScaler.

The preprocessing steps and machine learning model are combined into a Scikit-learn Pipeline so that the same preprocessing is automatically applied during prediction.

This prevents manual preprocessing of new customer data and ensures consistency between training and prediction.

Models Evaluated

Multiple classification algorithms were evaluated during the project:

Logistic Regression
Decision Tree
K-Nearest Neighbors
Naive Bayes
Random Forest
Support Vector Machine

The models were compared using:

Accuracy
Precision
Recall
F1 Score
Confusion Matrix
ROC-AUC
PR-AUC
Cross-validation
Model Selection

The final model was selected based on the business requirement of maintaining reasonable precision while identifying as many potential churners as possible.

Accuracy alone was not considered sufficient because the dataset contains more non-churners than churners.

Recall was given particular importance because missing a potential churner can result in losing a customer.

Precision was also considered because a very large number of false-positive churn predictions can result in unnecessary customer-retention efforts.

F1 Score was used to understand the balance between precision and recall.

ROC-AUC and PR-AUC were also considered during model comparison.

Final Model

The final application uses Logistic Regression inside a Scikit-learn Pipeline.

The pipeline performs preprocessing and prediction together.

The complete trained pipeline is saved as:

model/churn_prediction.pkl

The saved pipeline contains the preprocessing steps and trained model, allowing the application to load the model and directly predict new customers without retraining.

Classification Threshold

The classification threshold was changed from 0.50 to 0.30.

The purpose of lowering the threshold is to identify more potential churners.

The prediction rule is:

Probability >= 0.30 → Churn = Yes

Probability < 0.30 → Churn = No

For example, if the model produces a churn probability of 0.6422, the customer is classified as Yes because the probability is greater than the selected threshold of 0.30.

Lowering the threshold can increase recall, but it can also increase false positives and reduce precision.

Therefore, threshold selection depends on the company's business requirements and the precision-recall trade-off.

Model Performance

The final test-set accuracy was approximately 77.64%.

For the No class:

Precision: 0.91
Recall: 0.77
F1 Score: 0.84

For the Yes class:

Precision: 0.55
Recall: 0.79
F1 Score: 0.65

The confusion matrix was:

[[798 238]
 [ 77 296]]

The churn class is the most important class from a business perspective.

The recall of 0.79 indicates that the model successfully identified a large proportion of the actual churners in the test dataset.

Cross-Validation

Five-fold cross-validation was used to evaluate model stability.

The F1 scores obtained from the five folds were approximately:

0.6169, 0.6218, 0.6308, 0.5552, and 0.5756.

The mean F1 score was approximately 0.6001.

The standard deviation was approximately 0.0294.

The relatively small standard deviation indicates that the model performance does not vary dramatically across the validation folds.

Prediction

The trained pipeline generates a probability for the churn class.

For example, a customer may receive a churn probability of 0.6897.

Since this probability is greater than the selected threshold of 0.30, the final prediction is Yes.

The probability is an estimate generated by the model and does not guarantee that the customer will actually churn.

Streamlit Application

A Streamlit web application was developed to provide a simple interface for customer churn prediction.

The user can enter customer information into the application.

The application sends the information through the saved machine learning pipeline and displays:

Churn probability
Churn prediction

The application uses the already-trained model and does not retrain the model when making predictions.

Project Structure
Customer-churn-predictor/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
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
├── README.md
└── .gitignore
Installation

Clone the repository:

git clone <your-github-repository-url>

Move into the project directory:

cd Customer-churn-predictor

Install the required dependencies:

pip install -r requirements.txt
Requirements

The project uses the following Python libraries:

pandas
numpy
scikit-learn
joblib
streamlit
matplotlib
Train the Model

Run the training script:

python train_model.py

The trained pipeline will be saved as:

model/churn_prediction.pkl

Run Prediction

Run the prediction script:

python predict.py

The prediction script loads the saved pipeline and generates the churn probability and final prediction for a customer.

Run the Streamlit Application

Start the application using:

streamlit run app.py

The application will open in the web browser.

Evaluation Metrics
Accuracy

Accuracy measures the overall percentage of correctly classified customers.

Precision

Precision measures how many customers predicted as churners actually churned.

Recall

Recall measures how many actual churners were successfully identified by the model.

Recall is important because failing to identify a potential churner can result in losing that customer.

F1 Score

F1 Score combines precision and recall and provides a balance between both metrics.

ROC-AUC

ROC-AUC measures the ability of the model to distinguish between churners and non-churners across different classification thresholds.

PR-AUC

PR-AUC evaluates the relationship between precision and recall and is particularly useful for imbalanced classification problems.

Business Perspective

The purpose of this project is not simply to maximize accuracy.

A model can achieve high accuracy by predicting most customers as non-churners when the dataset contains more non-churners than churners.

Such a model may still fail to identify a significant number of customers who are actually going to churn.

Therefore, the project focuses on the trade-off between precision and recall.

A lower classification threshold can help identify more potential churners, but it can also produce more false-positive predictions.

The final threshold should therefore be selected according to the company's business requirements and the relative cost of false positives and false negatives.

Technologies Used
Python
Pandas
NumPy
Scikit-learn
Joblib
Streamlit
Matplotlib
Machine Learning Concepts Demonstrated

This project demonstrates:

Data cleaning
Train-test splitting
Numerical feature scaling
Categorical feature encoding
One-Hot Encoding
ColumnTransformer
Scikit-learn Pipeline
Logistic Regression
Classification
Confusion Matrix
Accuracy
Precision
Recall
F1 Score
ROC-AUC
PR-AUC
Cross-validation
Classification threshold tuning
Probability prediction
Model persistence
Streamlit deployment
Future Improvements

Possible future improvements include:

Hyperparameter tuning
Improved handling of class imbalance
Feature selection
Probability calibration
SHAP-based model explainability
Model monitoring
Automated retraining
REST API deployment
Cloud deployment
Customer retention recommendation system
Production model monitoring
Author

Akshay

AI/ML Engineering Student

Project Status

Completed end-to-end customer churn prediction project with data preprocessing, model evaluation, cross-validation, threshold tuning, model persistence, customer prediction, and Streamlit deployment.


