# 🚀 Smart Threshold Fraud Detection — Low False Positive Focus

## 📌 Project Overview

This project delivers a **complete machine learning pipeline** for detecting fraudulent credit card transactions with a **strong focus on minimizing false positives**.  
In the financial sector, incorrectly flagging legitimate transactions can lead to:

- Customer dissatisfaction and churn
- Increased manual review costs
- Potential loss of revenue

Using a real-world, highly imbalanced dataset, this solution follows a **production-minded approach** aligned with fintech and banking needs.

---

## 📊 Key Results

- **Model:** LightGBM with cost-sensitive threshold optimization  
- **PR-AUC (Test Set):** `0.92` *(example — replace with actual)*  
- **False Positive Rate (FPR):** `< 0.3%` *(example — replace with actual)*  
- **Fraud Loss Avoided:** `$XX,XXX` per 1,000 transactions  
- **Net Business Benefit:** `$XX,XXX` after accounting for false positive costs  

A **time-aware validation strategy** was applied to mimic real-world deployment performance.

---

## 📂 Repository Structure

```

smart-threshold-fraud-detection/
├─ app.py                       # Streamlit app for live predictions
├─ README.md
├─ requirements.txt
├─ .gitignore
│
├─ data/
│  └─ sample\_transactions.csv   # Example input file
│
├─ models/
│  ├─ gbm.joblib                 # Trained LightGBM model
│  ├─ scaler.joblib              # StandardScaler for features
│  └─ features\_list.joblib       # Feature names used in training
│
└─ notebooks/
└─ 01\_project\_intro\_and\_eda.ipynb  # Full pipeline: EDA, training, evaluation

````

---

## 🛠 Technologies

- **Python** — core language
- **Pandas, NumPy** — data manipulation
- **Scikit-learn** — preprocessing & evaluation
- **LightGBM** — gradient boosting model
- **SHAP** — explainability & reason codes
- **Streamlit** — interactive web app

---

## 🚀 How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
````

### 2️⃣ Run the Jupyter Notebook

Open `01_project_intro_and_eda.ipynb` to see:

* EDA & preprocessing
* Model training
* Explainability analysis

### 3️⃣ Launch the Streamlit App

```bash
streamlit run app.py
```

Upload a CSV (matching the training schema) to receive predictions with explainability.

---

## 🔮 Future Improvements

* Advanced feature engineering (e.g., transaction velocity features)
* SHAP force plots in the Streamlit UI
* Real-time prediction API with FastAPI

