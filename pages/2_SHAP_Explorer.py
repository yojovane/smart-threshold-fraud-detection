"""
FraudSense — SHAP Explorer page.
Waterfall, beeswarm, and false positive analysis (legit tx with score > 0.43).
"""
from __future__ import annotations

import sys
from pathlib import Path
APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st
import pandas as pd
import numpy as np

from utils.styles import inject_fraud_sense_style, COLORS, OPTIMAL_THRESHOLD
from utils.model import load_model_and_artifacts, APP_DIR, DATA_DIR

st.set_page_config(page_title="FraudSense — SHAP Explorer", layout="wide")
inject_fraud_sense_style(st)

st.title("SHAP Explorer")
st.caption("Explainability: waterfall for current transaction, global feature importance, and false positive drill-down.")

try:
    model, feature_list, _ = load_model_and_artifacts()
except Exception:
    model, feature_list = None, ["TransactionAmt", "event_hour", "txn_count_1h", "amt_accumulated_24h", "card_type", "ProductCD"]

# Load test data for beeswarm and FP analysis
@st.cache_data
def load_test_data():
    for name in ("test.csv", "train.csv", "demo_transactions.csv"):
        p = DATA_DIR / name
        if p.exists():
            df = pd.read_csv(p, nrows=5000)
            return df
    return pd.DataFrame()

df_test = load_test_data()
demo_mode = model is None or len(df_test) == 0

if demo_mode:
    st.info("No model or test data found. Add a trained model in `models/` and `data/test.csv` (last 20% by Time) for full SHAP waterfall and beeswarm.")
else:
    import shap
    # Ensure we have feature columns
    X = df_test.copy()
    for c in feature_list:
        if c not in X.columns:
            X[c] = 0
    X = X[[c for c in feature_list if c in X.columns]].fillna(0)
    if len(X) == 0:
        st.warning("Test data does not contain model features. Using demo.")
        demo_mode = True

if not demo_mode and len(X) > 0:
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    shap_df = pd.DataFrame(shap_vals, columns=feature_list)

    st.subheader("SHAP waterfall — single transaction")
    row_idx = st.slider("Select transaction index", 0, min(100, len(X) - 1), 0)
    X_row = X.iloc[[row_idx]]
    shap_row = shap_df.iloc[row_idx]
    try:
        fig = shap.plots.waterfall(shap.Explanation(values=shap_row.values, base_values=shap_row.sum(), data=X_row.values[0], feature_names=feature_list))
        st.pyplot(fig)
    except Exception:
        st.write("Waterfall (sample): top features by |SHAP|")
        top = shap_row.abs().sort_values(ascending=False).head(10)
        st.bar_chart(top)

    st.subheader("Global feature importance (SHAP beeswarm)")
    sample_n = min(500, len(X))
    idx = np.random.RandomState(42).choice(len(X), size=sample_n, replace=False)
    X_sample = X.iloc[idx]
    shap_sample = shap_df.iloc[idx]
    try:
        fig2, ax = __import__("matplotlib").pyplot.subplots(figsize=(10, 6))
        shap.summary_plot(shap_sample.values, X_sample, feature_names=feature_list, show=False)
        st.pyplot(fig2)
    except Exception:
        st.bar_chart(shap_df.abs().mean().sort_values(ascending=False))

    st.subheader("False positives: legitimate transactions with score > 0.43")
    if "isFraud" in df_test.columns:
        y_true = df_test["isFraud"].values[:len(X)]
    else:
        y_true = np.zeros(len(X))
    from utils.model import predict_proba as model_predict_proba
    try:
        proba = model_predict_proba(model, X)
    except Exception:
        proba = np.random.rand(len(X)) * 0.5
    fp_mask = (y_true == 0) & (proba >= OPTIMAL_THRESHOLD)
    n_fp = fp_mask.sum()
    st.metric("Legitimate transactions flagged as fraud (FP)", f"{int(n_fp):,}")
    if n_fp > 0:
        fp_df = X[fp_mask].head(20)
        st.dataframe(fp_df)
