import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import class_weight
import joblib

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("dataset.csv")

# Convert target variable
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

# Drop unnecessary columns
df = df.drop(["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"], axis=1)

# -------------------------------
# 2. Encode Categorical Variables
# -------------------------------
le = LabelEncoder()
categorical_columns = df.select_dtypes(include=["object"]).columns

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# -------------------------------
# 3. Split Features & Target
# -------------------------------
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# -------------------------------
# 4. Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 5. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------
# 6. Handle Class Imbalance
# -------------------------------
weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {0: weights[0], 1: weights[1]}

# -------------------------------
# 7. Train Logistic Regression
# -------------------------------
model = LogisticRegression(max_iter=2000, class_weight=class_weights)
model.fit(X_train, y_train)

# -------------------------------
# 8. Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 9. Evaluation
# -------------------------------
print("\n===== MODEL PERFORMANCE =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# 10. Confusion Matrix Plot
# -------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------------
# 11. Feature Importance
# -------------------------------
importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print("\nTop 10 Important Features:")
print(importance.head(10))

# -------------------------------
# 12. Save Model
# -------------------------------
joblib.dump(model, "attrition_model.pkl")
print("\nModel saved as attrition_model.pkl")