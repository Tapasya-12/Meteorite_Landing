# Credit Card Fraud Detection System

Every digital payment carries risk. Modern financial systems must decide—within milliseconds—whether a transaction should be approved, challenged, or blocked. This project replicates that real-world decision-making process using machine learning.

The **Credit Card Fraud Detection System** is an end-to-end fraud risk engine that combines supervised learning, anomaly detection, and rule-based decision logic to identify suspicious transactions. In addition to predicting fraud probability, the system translates model outputs into actionable risk levels and system decisions through an interactive Streamlit application.

This project demonstrates how financial institutions design fraud detection pipelines that balance accuracy, interpretability, and operational decision-making.

---

## Project Overview

The system simulates a realistic fraud detection workflow commonly used in banking and fintech platforms. It processes transaction attributes, extracts behavioral risk signals, evaluates them using multiple machine learning models, and produces actionable outcomes.

### Key Capabilities
- Fraud probability prediction using multiple machine learning models  
- Anomaly detection for unseen or rare transaction patterns  
- Risk-level classification (Low, Medium, High)  
- Automated decision logic for transaction handling  
- Interactive dashboard for real-time experimentation  

---

## Key Features

- End-to-end fraud detection pipeline  
- Combination of supervised and unsupervised learning models  
- Risk-based decision engine aligned with real-world systems  
- Interactive Streamlit dashboard  
- Feature-engineered transactional dataset  
- Model persistence and reuse  
- Modular and scalable project structure  

---

## Models Used

### Supervised Learning Models
- Logistic Regression (calibrated)
- Random Forest
- Support Vector Machine (SVM)
- XGBoost
- LightGBM
- Multi-Layer Perceptron (Neural Network)

### Unsupervised Learning Model
- Isolation Forest for anomaly detection

---

## Project Structure

Credit Card Fraud Detection/
│
├── data/
│ └── credit_card_fraud_dataset.xlsx
│
├── models/
│ ├── logistic.pkl
│ ├── random_forest.pkl
│ ├── svm.pkl
│ ├── xgboost.pkl
│ ├── lightgbm.pkl
│ ├── mlp.pkl
│ └── isolation_forest.pkl
│
├── src/
│ └── train_models.py
│
├── app.py
├── requirements.txt
└── README.md

yaml
Copy code

---

## Dataset Description

The dataset represents transaction-level information enriched with engineered behavioral features commonly used in fraud detection systems.

### Numerical Features
- amount  
- avg_transaction_amount  
- transaction_count_24h  
- account_age_days  
- amount_to_avg_ratio  
- high_velocity_flag  
- foreign_tx_flag  
- new_account_flag  

### Categorical Features
- transaction_type  
- merchant_category  
- currency  
- card_type  
- customer_region  
- transaction_country  
- billing_country  
- device_type  
- browser  

### Target Variable
- is_fraud  
  - 1 → Fraudulent transaction  
  - 0 → Legitimate transaction  

---

## Feature Engineering Strategy

To enhance detection performance and reflect real-world fraud patterns, the dataset includes engineered behavioral indicators such as:

- Transaction velocity signals  
- Cross-border activity indicators  
- New-account risk flags  
- Spending deviation ratios  
- Behavioral anomaly markers  

These features enable models to learn abnormal transaction behavior rather than relying solely on raw transactional values.

---

## Model Training Pipeline

Model training is implemented in `src/train_models.py` and includes the following steps:

- Train–test split  
- Feature scaling for numerical variables  
- One-hot encoding for categorical variables  
- Class imbalance handling using SMOTE  
- Model training and probability calibration  
- Model serialization using Joblib  

All trained models are stored in the `models/` directory and reused during inference.

---

## Risk Scoring and Decision Logic

### Risk Level Mapping

| Fraud Probability | Risk Level |
|------------------|------------|
| ≥ 0.75           | HIGH       |
| 0.40 – 0.74      | MEDIUM     |
| < 0.40           | LOW        |

### System Actions

| Risk Level | System Decision |
|-----------|------------------|
| HIGH      | Block transaction and send for manual review |
| MEDIUM    | Require step-up authentication |
| LOW       | Approve transaction |

This separation of prediction and decision logic mirrors the design of real-world financial fraud detection systems.

---

## Streamlit Application

The Streamlit application enables:

- Runtime model selection  
- Interactive transaction input  
- Real-time fraud probability prediction  
- Risk classification display  
- Automated decision recommendations  
- Clean and user-friendly interface  

---

## How to Run the Project

### Step 1: Create a virtual environment
```bash
python -m venv venv
Step 2: Activate the virtual environment
Windows

bash
Copy code
venv\Scripts\activate
macOS / Linux

bash
Copy code
source venv/bin/activate
Step 3: Install dependencies
bash
Copy code
pip install -r requirements.txt
Step 4: Train the models
bash
Copy code
cd src
python train_models.py
This step generates trained model files inside the models/ directory.

Step 5: Launch the Streamlit application
bash
Copy code
cd ..
streamlit run app.py
Access the application in your browser at:

arduino
Copy code
http://localhost:8501
Use Cases
Real-Time Transaction Risk Screening
Evaluates card transactions at the payment gateway level to determine whether a transaction should be approved, challenged, or blocked.

Fraud Analyst Decision Support
Provides fraud analysts with fraud probability scores, risk classification, and recommended system actions to accelerate investigation workflows.

Behavioral Fraud Pattern Analysis
Identifies abnormal patterns such as sudden spending spikes, unusual transaction velocity, and inconsistent geographic activity.

Risk Policy Prototyping
Allows experimentation with fraud thresholds and decision rules before production deployment in financial systems.

Portfolio-Grade Machine Learning Demonstration
Showcases end-to-end machine learning pipeline design, model training, evaluation, and interactive deployment.

Future Enhancements
Model explainability using SHAP or LIME

Dynamic and weighted risk thresholds

REST API exposure using FastAPI or Flask

Database integration for transaction storage

Model monitoring and data drift detection

Dockerization and cloud deployment

Batch prediction and advanced visual dashboards

Collaborators
Tapasya Patel

Varun Patel

yaml
Copy code

---
