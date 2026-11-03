
import joblib
import pandas as pd
import numpy as np

from pymongo import MongoClient
from datetime import datetime

# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

model = joblib.load("models/best_loan_approval_model.pkl")

print("\nRandom Forest model loaded successfully!")

# ==========================================
# 2. CONNECT TO MONGODB
# ==========================================

client = MongoClient("mongodb://localhost:27017/")
db = client["loan_approval_db"]
predictions_collection = db["loan_predictions"]

print("MongoDB connected successfully!")


# ==========================================
# 3. GET APPLICANT DETAILS
# ==========================================

def predict_loan():

    print("\n======================================")
    print("       LOAN APPROVAL PREDICTION")
    print("======================================")

    try:
        age = int(input("Enter age: "))

        income_usd = float(input("Enter annual income ($): "))

        employment = int(
            input("Enter employment experience (years): ")
        )

        loan_usd = float(input("Enter loan amount ($): "))

        interest = float(input("Enter loan interest rate (%): "))

        history = float(
            input("Enter credit history length (years): ")
        )

        score = int(input("Enter credit score: "))

        print("\nGender: male / female")
        gender = input("Enter gender: ").strip().lower()

        print("\nEducation: Associate / Bachelor / Doctorate / High School / Master")
        education = input("Enter education: ").strip()

        print("\nHome Ownership: RENT / MORTGAGE / OWN / OTHER")
        ownership = input("Enter home ownership: ").strip().upper()

        print("\nLoan Intent: DEBTCONSOLIDATION / EDUCATION / HOMEIMPROVEMENT / MEDICAL / PERSONAL / VENTURE")
        intent = input("Enter loan intent: ").strip().upper()

        defaults = input(
            "Previous loan defaults (Yes/No): "
        ).strip().capitalize()

        # ==========================================
        # 4. VALIDATE INPUTS
        # ==========================================

        if income_usd <= 0 or loan_usd <= 0:
            raise ValueError(
                "Income and loan amount must be greater than zero."
            )

        if age <= 0 or employment < 0 or history < 0:
            raise ValueError("Please enter valid age and experience values.")

        if not 0 < interest <= 100:
            raise ValueError("Enter a valid interest rate.")

        if not 300 <= score <= 850:
            raise ValueError("Enter a valid credit score.")

        if gender not in ["male", "female"]:
            raise ValueError("Gender must be male or female.")

        if education not in [
            "Associate", "Bachelor", "Doctorate",
            "High School", "Master"
        ]:
            raise ValueError("Invalid education category.")

        if ownership not in ["RENT", "MORTGAGE", "OWN", "OTHER"]:
            raise ValueError("Invalid home ownership category.")

        if intent not in [
            "DEBTCONSOLIDATION", "EDUCATION",
            "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"
        ]:
            raise ValueError("Invalid loan intent.")

        if defaults not in ["Yes", "No"]:
            raise ValueError("Previous defaults must be Yes or No.")

        # ==========================================
        # 5. PREPARE APPLICANT FEATURES
        # ==========================================

        applicant = {
            "person_age": age,
            "log_person_income": np.log1p(income_usd),
            "person_emp_exp": employment,
            "loan_amnt": loan_usd,
            "loan_int_rate": interest,
            "loan_percent_income": loan_usd / income_usd,
            "cb_person_cred_hist_length": history,
            "credit_score": score,
            "person_gender": gender,
            "person_education": education,
            "person_home_ownership": ownership,
            "loan_intent": intent,
            "previous_loan_defaults_on_file": defaults
        }

        input_data = pd.DataFrame([applicant])

        # ==========================================
        # 6. MAKE PREDICTION
        # ==========================================

        prediction = int(model.predict(input_data)[0])

        probabilities = model.predict_proba(input_data)[0]

        status = "Approved" if prediction == 1 else "Rejected"

        # ==========================================
        # 7. DISPLAY RESULT
        # ==========================================

        print("\n======================================")
        print("           PREDICTION RESULT")
        print("======================================")

        print("Applicant Age:", age)
        print("Annual Income: ${:,.2f}".format(income_usd))
        print("Loan Amount: ${:,.2f}".format(loan_usd))

        print("\nPredicted Loan Status:", status)

        print(
            "Approved Probability: {:.2f}%".format(
                probabilities[1] * 100
            )
        )

        print(
            "Rejected Probability: {:.2f}%".format(
                probabilities[0] * 100
            )
        )

        print("\nNote: This is an ML prediction, not an actual loan approval.")

        # ==========================================
        # 8. SAVE RESULT IN MONGODB
        # ==========================================

        record = {
            **applicant,
            "annual_income_usd": income_usd,
            "loan_amount_usd": loan_usd,
            "predicted_loan_status": prediction,
            "prediction_result": status,
            "approved_probability": float(probabilities[1]),
            "rejected_probability": float(probabilities[0]),
            "created_at": datetime.now()
        }

        predictions_collection.insert_one(record)

        print("\nPrediction successfully saved in MongoDB!")

    except ValueError as e:
        print("\nInvalid input:", e)

    except Exception as e:
        print("\nError:", e)


# ==========================================
# 9. RUN PROGRAM
# ==========================================

if __name__ == "__main__":
    predict_loan()
