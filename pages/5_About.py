"""
FraudSense — About This Project.
Credit Card Fraud (real anonymized data), temporal split 80/20 by Time, tech stack, author card.
"""
from __future__ import annotations

import sys
from pathlib import Path
APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st
from utils.styles import inject_fraud_sense_style, COLORS

st.set_page_config(page_title="FraudSense — About", layout="wide")
inject_fraud_sense_style(st)

st.title("About This Project")

st.subheader("What dataset is used?")
st.markdown("""
The **Credit Card Fraud Detection** dataset (Kaggle) contains **real anonymized** European credit card transactions (~284k transactions, 492 frauds, ~0.17% positive class).  
Features V1–V28 are PCA-anonymized for privacy. **Time** is seconds since the first transaction (no calendar dates). Using **real anonymized data** (not synthetic) gives realistic fraud patterns and class imbalance for model evaluation.
""")

st.subheader("Why temporal split (not random)?")
st.markdown("""
A **random** train/test split would leak future information: later transactions could appear in the training set.  
In production the model only sees the past. We use a **temporal split** based on the **Time** column: **first 80%** of transactions = train (and validation), **last 20%** = test. No future leak; evaluation reflects real deployment.
""")

st.subheader("Tech stack")
st.markdown("""
- **Python** — data and app  
- **LightGBM** — fraud model  
- **SHAP** — explainability and reason codes  
- **Streamlit** — UI  
- **Pandas, Scikit-learn, Plotly** — data and metrics  
""")

st.subheader("Repository")
st.markdown("[GitHub — FraudSense](https://github.com) — README with setup and link to this app.")

st.subheader("Author")
st.markdown("""
**Jovane Pascoal**  
Brooklyn, NY  
[LinkedIn](https://linkedin.com) · [GitHub](https://github.com)
""")
