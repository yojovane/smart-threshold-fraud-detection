# 🚀 Smart Threshold Fraud Detection — Low False Positive Focus

## 📌 Project Overview

This project delivers a **complete machine learning pipeline** for detecting fraudulent credit card transactions with a **strong focus on minimizing false positives**.  
In the financial sector, incorrectly flagging legitimate transactions can lead to:

- Customer dissatisfaction and churn
- Increased manual review costs
- Potential loss of revenue

The pipeline uses **real anonymized transaction data** (European card transactions, PCA-anonymized) and follows a **production-minded approach** aligned with fintech and banking needs: temporal train/test split, velocity features, SHAP explainability, and cost-sensitive threshold tuning.

---

## 📊 Dataset

**Credit Card Fraud Detection** (Kaggle): real anonymized credit card transactions from European cardholders.  
- **~284k transactions**, **492 frauds** (~0.17% positive class).  
- Features include **Time** (seconds since first transaction), **Amount**, and **V1–V28** (PCA-derived for anonymity).  
- No calendar dates; chronological order is simulated via the **Time** column for a **temporal split**: first **80%** for training, last **20%** for test (no future leak).

---

## 📊 Key Results

- **Model:** LightGBM with cost-sensitive threshold optimization  
- **PR-AUC (Test Set):** ~0.88–0.92 (temporal 80/20 split)  
- **False Positive Rate (FPR):** < 0.3% at chosen threshold  
- **Fraud Loss Avoided:** estimated per 1,000 transactions (FN×$180 avoided)  
- **Net Business Benefit:** after accounting for FP cost ($12) and fraud loss ($180)  

A **temporal split** (first 80% by `Time` = train, last 20% = test) was used to mimic real-world deployment.

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

### 3️⃣ Launch the Streamlit App (FraudSense)

```bash
streamlit run app.py
```

**FraudSense** is the portfolio app: Transaction Scorer (risk gauge + SHAP reasons), SHAP Explorer, ROI / Business Impact, Precision-Recall curve, and About.  
- **Demo mode:** runs without a trained model using `data/demo_transactions.csv` and heuristic scores.  
- **Full mode:** add a model under `models/` (e.g. `lgbm_fraud_v2.pkl` or `gbm.joblib`) and `data/test.csv` (temporal split: last 20% by `Time`) for real SHAP and ROI.  
- **Optimal threshold:** 0.43 (minimizes FN×$180 + FP×$12). Documented in code and ROI page.

**Live app:** [Deploy on Streamlit Community Cloud](https://share.streamlit.io) and add the repo link. Then add the app URL here.

---

## 🔮 Future Improvements

* SHAP force plots in the Streamlit UI
* Real-time prediction API with FastAPI

