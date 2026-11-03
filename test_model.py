
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load dataset and saved model
df = pd.read_csv("data/loan_data.csv")
model = joblib.load("models/best_loan_approval_model.pkl")

# Recreate the feature used during training
df["log_person_income"] = np.log1p(df["person_income"])

# Features
features = [
    "person_age",
    "log_person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
    "person_gender",
    "person_education",
    "person_home_ownership",
    "loan_intent",
    "previous_loan_defaults_on_file"
]

X = df[features]
y = df["loan_status"]

# Recreate the ORIGINAL train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Predict only on unseen test data
y_pred = model.predict(X_test)

print("\n========== UNSEEN TEST DATA EVALUATION ==========")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Test one actual approved applicant
approved_sample = X_test[y_test == 1].iloc[[0]]
actual_status = y_test.loc[approved_sample.index].iloc[0]

prediction = model.predict(approved_sample)[0]
probabilities = model.predict_proba(approved_sample)[0]

print("\n========== SINGLE APPLICANT TEST ==========")
print("Actual Status:", actual_status)
print("Predicted Status:", prediction)
print("Rejected Probability:", probabilities[0])
print("Approved Probability:", probabilities[1])

print("\nApplicant details:")
print(approved_sample.to_string(index=False))