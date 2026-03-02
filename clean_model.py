import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE

# -----------------------
# Load Dataset
# -----------------------
df = pd.read_csv("dataset.csv")

df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

# -----------------------
# Select Important Features Only
# -----------------------
selected_features = [
    "Age",
    "MonthlyIncome",
    "OverTime",
    "TotalWorkingYears",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "JobLevel",
    "JobInvolvement",
    "WorkLifeBalance",
    "EnvironmentSatisfaction"
]

df = df[selected_features + ["Attrition"]]

# Encode OverTime
df["OverTime"] = df["OverTime"].map({"No": 0, "Yes": 1})

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# -----------------------
# Scaling
# -----------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------
# SMOTE
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
# Train Random Forest
# -----------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# -----------------------
# Evaluation
# -----------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("Accuracy:", accuracy)
print("AUC Score:", auc)

# ROC Curve Data
fpr, tpr, _ = roc_curve(y_test, y_prob)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Feature Importance
feature_importance = model.feature_importances_

# -----------------------
# Save Artifacts
# -----------------------
joblib.dump(model, "clean_model.pkl")
joblib.dump(scaler, "clean_scaler.pkl")
joblib.dump(selected_features, "clean_features.pkl")

# Save evaluation data
joblib.dump((fpr, tpr), "roc_data.pkl")
joblib.dump(cm, "conf_matrix.pkl")
joblib.dump(feature_importance, "feature_importance.pkl")

print("Clean production model & evaluation data saved.")