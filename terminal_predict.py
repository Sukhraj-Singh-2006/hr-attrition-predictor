import joblib
import pandas as pd

# Load model
model = joblib.load("clean_model.pkl")
scaler = joblib.load("clean_scaler.pkl")
features = joblib.load("clean_features.pkl")

print("HR Attrition Prediction (Terminal Version)")
print("-------------------------------------------")

# Take user input
age = int(input("Age: "))
monthly_income = float(input("Monthly Income: "))
overtime = input("OverTime (Yes/No): ")
total_years = int(input("Total Working Years: "))
years_company = int(input("Years At Company: "))
years_role = int(input("Years In Current Role: "))
years_promo = int(input("Years Since Last Promotion: "))
years_manager = int(input("Years With Current Manager: "))
job_level = int(input("Job Level (1-5): "))
job_involvement = int(input("Job Involvement (1-4): "))
work_life = int(input("Work Life Balance (1-4): "))
env_sat = int(input("Environment Satisfaction (1-4): "))

# Convert overtime
overtime = 1 if overtime.lower() == "yes" else 0

# Create dataframe
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

# Scale
scaled_input = scaler.transform(input_data)

# Predict
prediction = model.predict(scaled_input)[0]
probability = model.predict_proba(scaled_input)[0][1]

print("\n--- RESULT ---")

if probability < 0.3:
    print(f"Low Attrition Risk ({probability:.2f})")
elif probability < 0.6:
    print(f"Moderate Attrition Risk ({probability:.2f})")
else:
    print(f"High Attrition Risk ({probability:.2f})")

if prediction == 1:
    print("Model Prediction: Employee Likely to Leave")
else:
    print("Model Prediction: Employee Likely to Stay")