"""
FraudSense — About This Project.
IEEE-CIS dataset, temporal split, tech stack, author card.
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

st.subheader("What is the IEEE-CIS dataset?")
st.markdown("""
The **IEEE-CIS Fraud Detection** dataset is from a Kaggle competition with real e-commerce transaction data (~590k transactions) provided by Vesta.  
Using **real data** (not synthetic) matters: fraud patterns, class imbalance, and temporal drift reflect production conditions and lead to realistic model evaluation.
""")

st.subheader("Why temporal split (not random)?")
st.markdown("""
In fraud detection, a **random** train/test split leaks future information into training: some "future" transactions (e.g. Oct–Dec) would appear in the training set.  
In production the model only sees the past. So we use a **temporal split**: train on Jan–Sep, test on Oct–Dec. No future leak, and evaluation reflects real deployment.
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
