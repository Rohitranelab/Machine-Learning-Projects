import streamlit as st
import pickle
import numpy as np
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------
# Load Model & Scaler
# -----------------------------
with open('artifact/heart_disease_model.pkl', "rb") as file:
    model = pickle.load(file)

with open('artifact/scaler.pkl', "rb") as file:
    scaler = pickle.load(file)

# -----------------------------
# Main Title
# -----------------------------
st.title("❤️ Heart Disease Prediction")

st.write(
    """
Enter the patient's clinical information below and click
**Predict Heart Disease**.
"""
)


st.markdown("---")

# -----------------------------
# Input Fields
# -----------------------------
col1, col2 = st.columns(2)

with col1:

    sex = st.selectbox(
        "Sex",
        ["Female", "Male"]
    )

    age = st.slider(
        "Age",
        18,
        100,
        40
    )

    cigsPerDay = st.slider(
        "Cigarettes Per Day",
        0,
        60,
        0
    )

    totChol = st.slider(
        "Total Cholesterol (mg/dL)",
        100,
        600,
        200
    )

with col2:

    sysBP = st.slider(
        "Systolic Blood Pressure",
        80,
        300,
        120
    )

    heartRate = st.slider(
        "Heart Rate",
        40,
        200,
        75
    )

    glucose = st.slider(
        "Glucose Level",
        40,
        400,
        80
    )

# Convert categorical value
sex = 1 if sex == "Male" else 0

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Predict Heart Disease", use_container_width=True):

    input_data = np.array([
        [
            sex,
            age,
            cigsPerDay,
            totChol,
            sysBP,
            heartRate,
            glucose
        ]
    ])

    # Scale data
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Probability
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.metric(
        label="Probability of Heart Disease",
        value=f"{probability:.2%}"
    )

    st.progress(float(probability))

    st.markdown("---")

    st.write("### Patient Information")

    st.write(f"**Sex:** {'Male' if sex else 'Female'}")
    st.write(f"**Age:** {age}")
    st.write(f"**Cigarettes Per Day:** {cigsPerDay}")
    st.write(f"**Total Cholesterol:** {totChol} mg/dL")
    st.write(f"**Systolic Blood Pressure:** {sysBP} mmHg")
    st.write(f"**Heart Rate:** {heartRate} bpm")
    st.write(f"**Glucose Level:** {glucose} mg/dL")