import streamlit as st
import joblib as jb
import pandas as pd 
from pridict import prediction_churn

pipe=jb.load("model/churn_pridiction.pkl")

st.title("customer churn predictor ")
st.write("Enter customer details to predict churn probability: ")
customer_id=st.text_input("enter the customer iD : ")
gender=st.selectbox("Gender",['male','gender'])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure=st.number_input("Tenure (months)",min_value=0,max_value=100,value=12)
phone_service=st.selectbox("phone Service",['Yes','No'])
multiple_lines=st.selectbox(
    "Multiple lines",
    ["No","Yes","No phone service"]
)

Internet_service=st.selectbox(
    "Internet Service ",
    ['DSL','Fiber optic','No']
)
online_security=st.selectbox("Online Security",["Yes","No"])
online_backup=st.selectbox("Online Backup",["Yes","No"])
Device_protection=st.selectbox("Device_protection",["Yes","No"])
Tech_support=st.selectbox("Tech support",["Yes","No"])
Streaming_tv=st.selectbox("Streaming TV",["Yes","No"])
Streaming_movies=st.selectbox("Streaming Movies",["Yes","No"])

contract=st.selectbox("Contract",["Month-to-month","One year","two year"])

paperless_billing=st.selectbox(
    "Paperless Billling",
    ["Yes","No"]
)
payment_method=st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)
monthly_charges=st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=75.00
    )

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=906.00
)

if st.button("Predict churn "):

    customer = pd.DataFrame({
        "customerID": [customer_id],
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [Internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [Device_protection],
        "TechSupport": [Tech_support],
        "StreamingTV": [Streaming_tv],
        "StreamingMovies": [Streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    probability,churn=prediction_churn(customer)

    st.write("Churn Probability :", probability)
    st.write("Prediction: ", churn)

