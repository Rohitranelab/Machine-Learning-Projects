import streamlit as st
import numpy as np
import pickle

with open('artifact/salary.pkl', 'rb') as file:
    model = pickle.load(file)

st.set_page_config(page_title="Salary Prediction")

st.title("Salary Prediction")
st.write("Enter your years of experience to predict the estimated salary.")

input_experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    step=0.5,
    placeholder="Type years of experience"
)

if st.button("Predict Salary", use_container_width=True):
    output = model.predict(np.array([[input_experience]]))

    st.success(f"💵 Predicted Salary: ₹ {output[0]:,.2f}")