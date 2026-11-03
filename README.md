# Loan Approval Prediction

A machine learning project that predicts loan application outcomes from applicant and loan details. It includes a training and analysis notebook, a Flask web interface, a command-line predictor, a saved scikit-learn pipeline, and optional MongoDB storage for predictions.

> **Academic demonstration:** predictions are model estimates and are not lending decisions or financial advice.

## Features

- Explores and preprocesses the loan dataset in Jupyter.
- Compares Logistic Regression, Decision Tree, and Random Forest classifiers.
- Uses a scikit-learn preprocessing pipeline with standardized numeric features and one-hot encoded categorical features.
- Provides browser-based and command-line prediction interfaces.
- Optionally stores prediction records in MongoDB.

## Repository layout

```text
.
├── app.py                         # Flask web application
├── predict_loan.py                # Interactive command-line predictor
├── test_model.py                  # Evaluation script for the saved model
├── Loan_Approval_Prediction.ipynb # Data exploration, training, and MongoDB analysis
├── data/loan_data.csv             # Dataset
├── models/best_loan_approval_model.pkl
├── results/model_comparison.csv   # Model comparison metrics
├── static/style.css
└── templates/                     # Flask HTML pages
```

## Dataset and preprocessing

The CSV contains 45,000 rows and 14 columns. The target, `loan_status`, uses `0` for rejected and `1` for approved. The dataset has 35,000 rejected and 10,000 approved records. The notebook removes the seven records with ages above 100 before model training.

The model uses 13 input features: age, log-transformed income, employment experience, loan amount, interest rate, loan-to-income ratio, credit history length, credit score, gender, education, home ownership, loan intent, and previous loan defaults. Numeric and categorical features are prepared within the model pipeline.

The web form and CLI accept income and loan amount in INR, then convert them to the dataset's assumed USD values using a fixed rate of ₹83 per USD. Update this assumption in `app.py` and `predict_loan.py` if it is no longer appropriate.

## Reported model results

The notebook's saved comparison reports the following metrics on its 80/20 stratified train/test split:

| Model | Accuracy | Precision | Recall | F1 score |
|---|---:|---:|---:|---:|
| Logistic Regression | 85.22% | 61.08% | 92.35% | 73.53% |
| Decision Tree | 88.87% | 69.98% | 87.40% | 77.72% |
| Random Forest | **90.35%** | **73.06%** | **89.65%** | **80.51%** |

The saved artifact is the Random Forest pipeline at `models/best_loan_approval_model.pkl`. Results are dataset evaluation metrics; they do not guarantee performance on new applicants.

## Requirements

- Python 3.9 or newer
- MongoDB Community Server, if prediction storage is needed

Install dependencies from the project directory:

```bash
python -m venv .venv
```

Activate the environment, then run:

```bash
python -m pip install -r requirements.txt
```

## Run the web application

Ensure MongoDB is running locally if you want predictions to be saved. The application connects to `mongodb://localhost:27017/`, database `loan_approval_db`, collection `loan_predictions`.

```bash
python app.py
```

Open the local address printed by Flask (typically `http://127.0.0.1:5000/`). The current application enables Flask debug mode; use an appropriate production server and configuration before deploying publicly.

## Run the command-line predictor

```bash
python predict_loan.py
```

It prompts for applicant details, displays the predicted status and class probabilities, and attempts to save the prediction to MongoDB.

## Train and evaluate

Open `Loan_Approval_Prediction.ipynb` from the repository root and run cells in order to explore data, train models, and save the comparison CSV and model. MongoDB-related notebook cells require a local MongoDB server. The notebook's prediction-analysis section clears existing documents in the `loan_predictions` collection before inserting its evaluation records; take care when running it against a database containing records you need to retain.

To evaluate the saved model using the script:

```bash
python test_model.py
```

## MongoDB data

The application stores submitted applicant features, INR amounts, predicted status, probabilities, and creation time in `loan_approval_db.loan_predictions`. The app and CLI require MongoDB to be available for a successful prediction request because they insert each result. Configure database access and data retention appropriately for your environment.
