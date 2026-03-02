import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# -----------------------
# Load & Preprocess Data
# -----------------------
df = pd.read_csv("dataset.csv")

df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
df = df.drop(["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"], axis=1)

le = LabelEncoder()
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = le.fit_transform(df[col])

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

feature_names = X.columns  # Save feature names

# -----------------------
# Scaling
# -----------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for deployment
joblib.dump(scaler, "scaler.pkl")

# -----------------------
# SMOTE Balancing
# -----------------------
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# -----------------------
# Train-Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# -----------------------
# Models
# -----------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

results = []
best_model = None
best_accuracy = 0

# -----------------------
# Train & Evaluate
# -----------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results.append([name, accuracy, auc])

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# -----------------------
# Save Best Model
# -----------------------
joblib.dump(best_model, "best_model.pkl")
joblib.dump(feature_names, "feature_names.pkl")

print("\n=== MODEL COMPARISON ===")
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "AUC Score"])
print(results_df)

print(f"\nBest Model: {best_model_name}")
print(f"Best Accuracy: {best_accuracy:.4f}")
print("Best model saved as best_model.pkl")
print("Scaler saved as scaler.pkl")

# -----------------------
# Accuracy Comparison Chart
# -----------------------
plt.figure()
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()