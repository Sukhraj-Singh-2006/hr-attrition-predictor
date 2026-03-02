import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 16px;
        height: 3em;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="HR Attrition Predictor", layout="centered")

st.title("HR Attrition Risk Assessment Tool")
st.write("This tool predicts employee attrition risk using a machine learning model.")

# Load model artifacts
model = joblib.load("clean_model.pkl")
scaler = joblib.load("clean_scaler.pkl")
features = joblib.load("clean_features.pkl")

st.header("Employee Profile")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 60, 30)
    monthly_income = st.number_input("Monthly Income", min_value=0)
    total_years = st.slider("Total Working Years", 0, 40, 5)
    years_company = st.slider("Years At Company", 0, 40, 5)
    years_role = st.slider("Years In Current Role", 0, 20, 3)
    overtime = st.selectbox("OverTime", ["No", "Yes"])

with col2:
    years_promo = st.slider("Years Since Last Promotion", 0, 20, 2)
    years_manager = st.slider("Years With Current Manager", 0, 20, 3)
    job_level = st.slider("Job Level (1–5)", 1, 5, 2)
    job_involvement = st.slider("Job Involvement (1–4)", 1, 4, 3)
    work_life = st.slider("Work Life Balance (1–4)", 1, 4, 3)
    env_sat = st.slider("Environment Satisfaction (1–4)", 1, 4, 3)

# Convert categorical
overtime = 1 if overtime == "Yes" else 0

# Prepare input
input_data = pd.DataFrame([[
    age,
    monthly_income,
    overtime,
    total_years,
    years_company,
    years_role,
    years_promo,
    years_manager,
    job_level,
    job_involvement,
    work_life,
    env_sat
]], columns=features)

if st.button("Assess Attrition Risk"):

    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.markdown("### Risk Assessment Result")

    st.progress(float(probability))

    if probability < 0.3:
        st.success(f"Low Attrition Risk ({probability:.2f})")
    elif probability < 0.6:
        st.warning(f"Moderate Attrition Risk ({probability:.2f})")
    else:
        st.error(f"High Attrition Risk ({probability:.2f})")

    if prediction == 1:
        st.error("Model Prediction: Employee Likely to Leave")
    else:
        st.success("Model Prediction: Employee Likely to Stay")