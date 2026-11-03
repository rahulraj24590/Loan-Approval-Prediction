
from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from pymongo import MongoClient
from datetime import datetime

app = Flask(__name__)

# Load trained ML pipeline
model = joblib.load("models/best_loan_approval_model.pkl")

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["loan_approval_db"]
predictions_collection = db["loan_predictions"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        try:
            # Get applicant details directly in USD
            income_usd = float(request.form["person_income"])
            loan_usd = float(request.form["loan_amnt"])

            if income_usd <= 0 or loan_usd <= 0:
                raise ValueError(
                    "Income and loan amount must be greater than zero."
                )

            # Prepare applicant data
            applicant = {
                "person_age": int(request.form["person_age"]),

                "log_person_income": np.log1p(income_usd),

                "person_emp_exp": int(request.form["person_emp_exp"]),

                "loan_amnt": loan_usd,

                "loan_int_rate": float(request.form["loan_int_rate"]),

                "loan_percent_income": loan_usd / income_usd,

                "cb_person_cred_hist_length": float(
                    request.form["cb_person_cred_hist_length"]
                ),

                "credit_score": int(request.form["credit_score"]),

                "person_gender": request.form["person_gender"].strip().lower(),

                "person_education": request.form["person_education"].strip(),

                "person_home_ownership": request.form[
                    "person_home_ownership"
                ].strip().upper(),

                "loan_intent": request.form["loan_intent"].strip().upper(),

                "previous_loan_defaults_on_file": request.form[
                    "previous_loan_defaults_on_file"
                ].strip().capitalize()
            }

            # Convert input into DataFrame
            input_data = pd.DataFrame([applicant])

            # Predict loan status
            prediction = int(model.predict(input_data)[0])
            probabilities = model.predict_proba(input_data)[0]

            status = "Approved" if prediction == 1 else "Rejected"

            print("\n========== LOAN PREDICTION ==========")
            print("Applicant Details:", applicant)
            print("Prediction:", status)
            print("Approved Probability:", probabilities[1])
            print("Rejected Probability:", probabilities[0])

            # Save prediction to MongoDB
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

            return render_template(
                "result.html",
                status=status,
                prediction=prediction
            )

        except Exception as e:
            return f"Prediction error: {str(e)}", 400

    return render_template("predict.html")


if __name__ == "__main__":
    app.run(debug=True)
