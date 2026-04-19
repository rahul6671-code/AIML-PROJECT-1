import streamlit as st
import joblib
import pandas as pd

st.title("Attrition Predictor")

model = joblib.load("model.pkl")

age = st.number_input("Age")
income = st.number_input("Income")

if st.button("Predict"):
    df = pd.DataFrame({
        "Age": [age],
        "MonthlyIncome": [income]
    })

    pred = model.predict(df)[0]

    if pred == 1:
        st.error("Will Leave")
    else:
        st.success("Will Stay")
