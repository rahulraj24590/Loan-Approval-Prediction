# Loan Approval Prediction Using Machine Learning and MongoDB

## 1. Project Overview

The **Loan Approval Prediction System** is a machine learning-based project that predicts whether a loan application should be approved or rejected based on applicant information.

The project combines machine learning algorithms with MongoDB to store, retrieve, and analyze loan prediction results.

It compares multiple classification algorithms and selects the best-performing model for loan approval prediction.

## 2. Project Objectives

- Analyze loan applicant data.
- Perform data cleaning and preprocessing.
- Apply feature selection and encoding techniques.
- Train and compare multiple machine learning classification models.
- Evaluate models using Accuracy, Precision, Recall, and F1-score.
- Save the best-performing machine learning model.
- Store predictions in MongoDB using PyMongo.
- Analyze prediction results using MongoDB Aggregation Framework.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning |
| MongoDB | NoSQL database |
| PyMongo | MongoDB integration |
| Joblib | Model serialization |
| Jupyter Notebook | Development environment |

## 4. Dataset Description

The project uses a loan approval dataset containing applicant information and loan-related attributes.

- Original records: 45,000
- Records after cleaning: 44,993
- Features used: 13
- Target variable: `loan_status`

The target variable represents:

- `0` – Loan Rejected
- `1` – Loan Approved

### Data Preprocessing

The following preprocessing operations were performed:

1. Checked missing values.
2. Checked duplicate records.
3. Removed invalid age records (applicants older than 100).
4. Applied logarithmic transformation to income using `np.log1p()`.
5. Applied StandardScaler to numerical features.
6. Applied OneHotEncoder to categorical features.

## 5. Machine Learning Models

Three classification algorithms were implemented and compared.

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

The dataset was divided into:
- Training data: 80%
- Testing data: 20%

A preprocessing pipeline was created using Scikit-learn's ColumnTransformer and Pipeline.

## 6. Model Evaluation and Comparison

The models were evaluated using four performance metrics.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 85.22% | 61.08% | 92.35% | 73.53% |
| Decision Tree | 88.87% | 69.98% | 87.40% | 77.72% |
| Random Forest | **90.35%** | **73.06%** | 89.65% | **80.51%** |

### Best Performing Model

Random Forest achieved the highest accuracy of **90.35%** and the highest F1-score of **80.51%**.

Therefore, Random Forest was selected as the final model.

The trained model was saved using Joblib.

**Model location:** `models/best_loan_approval_model.pkl`

## 7. MongoDB Integration

MongoDB was integrated into the project using PyMongo.

The database stores machine learning prediction results for further analysis.

### MongoDB Operations

- Connected Python to MongoDB.
- Generated loan approval predictions.
- Stored prediction records in MongoDB.
- Verified inserted documents.
- Retrieved prediction records.
- Used MongoDB Aggregation Framework for analysis.

### MongoDB Collection

Collection name: `loan_predictions`

Total prediction records stored: **8,999**

### Aggregation Analysis

MongoDB aggregation pipelines were used to analyze actual and predicted loan statuses.

**Prediction Analysis:**

| Actual Status | Predicted Status | Total |
|---|---|---:|
| Rejected | Rejected | 6,338 |
| Rejected | Approved | 661 |
| Approved | Rejected | 207 |
| Approved | Approved | 1,793 |

**Loan Approval Summary:**

| Status | Applicants |
|---|---:|
| Rejected | 6,545 |
| Approved | 2,454 |

## 8. Project Visualizations

The project includes visualizations such as:

- Confusion Matrix
- Actual vs Predicted Loan Status
- Predicted Loan Approval Distribution
- Model Performance Comparison

These visualizations help understand model performance and loan approval patterns.

## 9. Project Structure

```text
Loan_Approval_Prediction/
│
├── data/
│   └── loan_data.csv
│
├── models/
│   └── best_loan_approval_model.pkl
│
├── results/
│   └── model_comparison.csv
│
├── Loan_Approval_Prediction.ipynb
├── README.md
└── requirements.txt
```

## 10. Installation and Execution

### Prerequisites

- Python 3.x
- MongoDB Community Server
- MongoDB running locally
- Jupyter Notebook or VS Code

### Step 1: Clone the Repository

```bash
git clone <your-github-repository-url>
```

### Step 2: Navigate to the Project Directory

```bash
cd Loan_Approval_Prediction
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Start MongoDB

Make sure your local MongoDB server is running.

### Step 5: Open the Jupyter Notebook

```bash
jupyter notebook
```

Open `Loan_Approval_Prediction.ipynb` and execute the cells sequentially.

**Note:** Update the MongoDB connection string in the notebook if your MongoDB server uses a different configuration.

## 11. Conclusion

This project demonstrates how machine learning and NoSQL databases can work together to build a loan approval prediction system.

Three classification algorithms were implemented and evaluated. Random Forest achieved the best performance with 90.35% accuracy.

MongoDB was successfully integrated using PyMongo to store 8,999 prediction records and perform aggregation-based analysis.

The project provides practical experience in data preprocessing, machine learning, model evaluation, database integration, and data analysis.

---

**Project Domain:** Machine Learning and NoSQL Databases  
**Programming Language:** Python  
**Database:** MongoDB  
**Best Model:** Random Forest Classifier
