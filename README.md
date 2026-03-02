# Employee Attrition Prediction Using Logistic Regression

## 📌 Project Overview

This project predicts employee attrition (whether an employee will leave the company or not) using Logistic Regression.

The goal is to help HR departments identify employees who are at risk of leaving and take preventive action.

---

## 📊 Dataset

Dataset used: IBM HR Analytics Employee Attrition & Performance  
Source: Kaggle

Total Records: 1470  
Total Features: 35 (reduced to 31 after preprocessing)

Target Variable:

- Attrition (Yes = 1, No = 0)

---

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 🔄 Project Workflow

1. Data Loading
2. Data Cleaning
3. Encoding Categorical Variables
4. Feature Scaling
5. Train-Test Split (80/20)
6. Logistic Regression Model
7. Handling Class Imbalance
8. Model Evaluation
9. Confusion Matrix Visualization
10. Model Saving

---

## 📈 Model Performance

Accuracy: ~71% (after class imbalance handling)

The recall for employees who left improved significantly using balanced class weights.

---

## 📦 Files in Project

- dataset.csv → HR dataset
- attrition_model.py → Main ML code
- attrition_model.pkl → Saved trained model
- README.md → Project documentation
- requirements.txt → Required Python libraries

---

## 🎯 Conclusion

The Logistic Regression model successfully predicts employee attrition.  
Using class balancing improved the model’s ability to detect employees who are likely to leave.

This project demonstrates a complete machine learning pipeline from data preprocessing to deployment-ready model saving.

---

## 👨‍💻 Author

Sukhraj Singh
