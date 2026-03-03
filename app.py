import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
# Load real evaluation data
fpr, tpr = joblib.load("roc_data.pkl")
cm = joblib.load("conf_matrix.pkl")
feature_importance = joblib.load("feature_importance.pkl")
# Load Logistic Regression artifacts
log_model = joblib.load("logistic_model.pkl")
log_fpr, log_tpr = joblib.load("logistic_roc.pkl")
log_cm = joblib.load("logistic_conf_matrix.pkl")

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

model_choice = st.selectbox(
    "Select Model",
    ["Random Forest", "Logistic Regression"]
)
if st.button("Assess Attrition Risk"):

    scaled_input = scaler.transform(input_data)

    if model_choice == "Random Forest":
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]
    else:
        prediction = log_model.predict(scaled_input)[0]
        probability = log_model.predict_proba(scaled_input)[0][1]

    # 👇 ADD HERE
    st.write("Model Used:", model_choice)
    st.write("Raw Probability:", probability)

    st.markdown("## Risk Assessment Result")

    if probability < 0.3:
        st.success(f"Low Attrition Risk ({probability:.4f})")
    elif probability < 0.6:
        st.warning(f"Moderate Attrition Risk ({probability:.2f})")
    else:
        st.error(f"High Attrition Risk ({probability:.2f})")

    if prediction == 1:
        st.error("Model Prediction: Employee Likely to Leave")
    else:
        st.success("Model Prediction: Employee Likely to Stay")

    st.markdown("---")
    # -------------------------------------------------
# 📊 PROFESSIONAL MODEL EVALUATION DASHBOARD
# -------------------------------------------------

st.markdown("---")
st.header("📊 Model Evaluation Dashboard")

tab1, tab2, tab3 = st.tabs([
    "Feature Importance",
    "ROC Curve",
    "Confusion Matrix"
])

# -------------------------------
# 1️⃣ Feature Importance
# -------------------------------
with tab1:
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": feature_importance
    }).sort_values(by="Importance", ascending=True)

    fig1, ax1 = plt.subplots()
    ax1.barh(importance_df["Feature"], importance_df["Importance"])
    ax1.set_xlabel("Importance Score")
    st.pyplot(fig1)

# -------------------------------
# 2️⃣ REAL ROC Curve
# -------------------------------
with tab2:
    fig2, ax2 = plt.subplots()

    if model_choice == "Random Forest":
        ax2.plot(fpr, tpr, label="Random Forest")
    else:
        ax2.plot(log_fpr, log_tpr, label="Logistic Regression")

    ax2.plot([0, 1], [0, 1], '--')
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()

    st.pyplot(fig2)

# -------------------------------
# 3️⃣ REAL Confusion Matrix
# -------------------------------
with tab3:
    fig3, ax3 = plt.subplots()

    if model_choice == "Random Forest":
        display_cm = cm
    else:
        display_cm = log_cm

    ax3.matshow(display_cm)

    for (i, j), val in np.ndenumerate(display_cm):
        ax3.text(j, i, f"{val}", ha='center', va='center')

    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")

    st.pyplot(fig3)