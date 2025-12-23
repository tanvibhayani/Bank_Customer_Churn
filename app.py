# =========================================
# BANK CUSTOMER CHURN PREDICTION WEB APP
# =========================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -----------------------------------------
# Load trained model and scaler
# -----------------------------------------
model = joblib.load("model.pkl")   # Replace with your model path
scaler = joblib.load("scaler.pkl") # Replace with your scaler path

# -----------------------------------------
# Page Configuration
# -----------------------------------------
st.set_page_config(
    page_title="Bank Churn Prediction App",
    page_icon="🏦",
    layout="centered"
)

# -----------------------------------------
# Title
# -----------------------------------------
st.title("🏦 Bank Customer Churn Prediction")
st.write("Enter customer details to predict churn")
st.markdown("---")

# -----------------------------------------
# User Inputs
# -----------------------------------------
credit_score = st.number_input("Credit Score", 300, 900, 600)
age = st.number_input("Age", 18, 100, 30)
balance = st.number_input("Account Balance", 0.0, value=50000.0)
salary = st.number_input("Estimated Salary", 0.0, value=60000.0)
tenure = st.number_input("Tenure (Years)", 0, 10, 3)

products_number = st.number_input("Number of Products", 1, 10, 1)
has_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])

gender = st.selectbox("Gender", ["Male", "Female"])
country = st.selectbox("Country", ["France", "Germany", "Spain"])

st.markdown("---")

# -----------------------------------------
# Prediction
# -----------------------------------------
if st.button("🔮 Predict Churn"):

    # ----- Manual One-Hot Encoding -----
    user_data = {
        "credit_score": credit_score,
        "age": age,
        "balance": balance,
        "estimated_salary": salary,
        "tenure": tenure,
        "products_number": products_number,
        "credit_card": has_card,
        "active_member": is_active,
        "gender_Female": 1 if gender == "Female" else 0,
        "gender_Male": 1 if gender == "Male" else 0,
        "country_France": 1 if country == "France" else 0,
        "country_Germany": 1 if country == "Germany" else 0,
        "country_Spain": 1 if country == "Spain" else 0
    }

    # Convert to DataFrame
    user_df = pd.DataFrame([user_data])

    # Add missing columns expected by scaler
    for col in scaler.feature_names_in_:
        if col not in user_df.columns:
            user_df[col] = 0

    # Ensure column order matches scaler
    user_df = user_df[scaler.feature_names_in_]

    # Scaling
    user_scaled = scaler.transform(user_df)

    # Prediction
    prediction = model.predict(user_scaled)

    # Result display
    if prediction[0] == 1:
        st.error("❌ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is NOT likely to churn")

st.markdown("---")
st.caption("Machine Learning Project | Bank Customer Churn Prediction")
