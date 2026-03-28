import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------
# Page Style
# -----------------------
st.markdown("""
    <style>
        .main {
            background-color: #f7f9fc;
        }
        .title {
            text-align: center;
            font-size: 42px !important;
            color: #1a3c5e;
            font-weight: 700;
        }
        .subheader {
            color: #444;
            font-size: 20px;
            margin-top: -10px;
        }
        .bold-label {
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# Load model
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "notebooks", "churn_model.pkl")
pipe = joblib.load(model_path)

st.markdown("<h1 class='title'>🏦 Bank Customer Churn Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Select customer details to predict whether they will churn</p>", unsafe_allow_html=True)

# -----------------------
# UI Layout
# -----------------------
st.markdown("### 👤 Customer Details")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.slider("Credit Score", 300, 850, 650)
    age          = st.slider("Age", 18, 92, 40)
    tenure       = st.slider("Tenure (years)", 0, 10, 5)
    geography    = st.selectbox("Geography", ["France", "Germany", "Spain"])

with col2:
    gender       = st.selectbox("Gender", ["Male", "Female"])
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_cr_card  = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active    = st.selectbox("Is Active Member?", ["Yes", "No"])

st.markdown("---")

st.markdown("### 💰 Financial Details")
col3, col4 = st.columns(2)

with col3:
    balance = st.number_input("Account Balance ($)", min_value=0.0,
                               max_value=300000.0, value=50000.0, step=1000.0)
with col4:
    estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0,
                                        max_value=200000.0, value=80000.0, step=1000.0)

# -----------------------
# Prediction Logic
# -----------------------
st.markdown("---")
if st.button("🔍 Predict Churn", use_container_width=True):

    if   age <= 25: age_cat = "Adult"
    elif age <= 40: age_cat = "Mid Age"
    elif age <= 60: age_cat = "Senior"
    else:           age_cat = "Old Age"

    query = pd.DataFrame([{
        "CreditScore":     credit_score,
        "Geography":       geography,
        "Gender":          gender,
        "Age":             float(age),
        "Tenure":          tenure,
        "Balance":         balance,
        "NumOfProducts":   num_products,
        "HasCrCard":       1.0 if has_cr_card == "Yes" else 0.0,
        "IsActiveMember":  1.0 if is_active   == "Yes" else 0.0,
        "EstimatedSalary": estimated_salary,
        "AgeCategory":     age_cat,
    }])

    prediction  = pipe.predict(query)[0]
    probability = pipe.predict_proba(query)[0][1]

    st.write('The values you selected for prediction:')
    st.write(query)

    if prediction == 1:
        st.error(f"⚠️ **This customer is likely to CHURN** — Churn Probability: {probability:.1%}")
    else:
        st.success(f"✅ **This customer is likely to STAY** — Churn Probability: {probability:.1%}")