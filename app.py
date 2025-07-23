# app.py
import streamlit as st
import numpy as np
import joblib

model = joblib.load("glass_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

st.title("🔍 Glass Type Prediction")
st.markdown("Enter the chemical composition values to predict the glass type.")

features = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
inputs = []

for feature in features:
    val = st.number_input(f"{feature}", format="%.4f")
    inputs.append(val)

if st.button("Predict"):
    input_scaled = scaler.transform([inputs])
    prediction = model.predict(input_scaled)
    proba = model.predict_proba(input_scaled)

    pred_label = le.inverse_transform(prediction)[0]
    st.success(f"🧪 Predicted Glass Type: {pred_label}")

    st.subheader("🔢 Prediction Probabilities:")
    for i, prob in enumerate(proba[0]):
        st.write(f"Type {le.inverse_transform([i])[0]}: {prob:.2%}")

st.sidebar.markdown("Made by Your Name | [GitHub](https://github.com/YOUR_USERNAME/glass-type-prediction-app)")