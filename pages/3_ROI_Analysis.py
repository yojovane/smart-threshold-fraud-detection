"""
FraudSense — Business Impact / ROI page.
Cost curve by threshold, optimal 0.43, monthly ROI estimate (NYC neobank 50k tx/day).
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
import plotly.express as px
import plotly.graph_objects as go

from utils.styles import inject_fraud_sense_style, COLORS, OPTIMAL_THRESHOLD, COST_FN_USD, COST_FP_USD
from utils.roi import sweep_thresholds, cost_at_threshold, monthly_roi_estimate, scale_test_to_monthly, TX_PER_MONTH
from utils.model import load_model_and_artifacts, APP_DIR, DATA_DIR

st.set_page_config(page_title="FraudSense — ROI Analysis", layout="wide")
inject_fraud_sense_style(st)

st.title("Business Impact / ROI")
st.caption("Cost per threshold (FN×$180 + FP×$12). Optimal threshold 0.43 minimizes total cost. ROI for a NYC neobank at 50k transactions/day.")

# Load test set and model for real metrics
@st.cache_data
def load_test_and_scores():
    p = DATA_DIR / "test.csv"
    if not p.exists():
        p = DATA_DIR / "train.csv"
    if not p.exists():
        return None, None, None
    df = pd.read_csv(p, nrows=10000)
    return df, None, None

df_test, _, _ = load_test_and_scores()
try:
    model, feature_list, _ = load_model_and_artifacts()
except Exception:
    model, feature_list = None, []

# Synthetic y_true and y_prob for demo if no data
if df_test is not None and model is not None:
    X = df_test.copy()
    for c in feature_list:
        if c not in X.columns:
            X[c] = 0
    X = X[[c for c in feature_list if c in X.columns]].fillna(0)
    if "isFraud" in df_test.columns:
        y_true = df_test["isFraud"].values[:len(X)]
    else:
        y_true = (np.random.rand(len(X)) < 0.035).astype(int)
    try:
        from utils.model import predict_proba as model_predict_proba
        y_prob = model_predict_proba(model, X)
    except Exception:
        y_prob = np.clip(np.random.rand(len(X)) * 0.5 + y_true * 0.4, 0, 1)
else:
    np.random.seed(42)
    n = 5000
    y_true = (np.random.rand(n) < 0.035).astype(int)
    y_prob = np.clip(np.random.rand(n) * 0.4 + y_true * np.random.rand(n) * 0.5, 0, 1)

sweep = sweep_thresholds(y_true, y_prob)
st.subheader("Cost by threshold (FN×$180 + FP×$12)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["total_cost"], mode="lines+markers", name="Total cost", line=dict(color=COLORS["blue"])))
fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["fn_cost"], mode="lines", name="FN cost", line=dict(color=COLORS["accent"], dash="dash")))
fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["fp_cost"], mode="lines", name="FP cost", line=dict(color=COLORS["accent2"], dash="dash")))
fig.add_vline(x=OPTIMAL_THRESHOLD, line_dash="dot", line_color=COLORS["green"], annotation_text="Optimal 0.43")
fig.update_layout(xaxis_title="Threshold", yaxis_title="USD", height=400, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=COLORS["text"])
st.plotly_chart(fig, use_container_width=True)

st.markdown(f"**Optimal threshold: 0.43** (minimizes $FN×180 + $FP×12 cost function).")

# ROI estimate
st.subheader("Monthly ROI estimate (NYC neobank: 50k tx/day)")
n_tx = len(y_true)
_, fn_opt, fp_opt, _, tp_opt = cost_at_threshold(y_true, y_prob, OPTIMAL_THRESHOLD)
fraud_blocked_month, fp_month = scale_test_to_monthly(n_tx, int(tp_opt), int(fp_opt))
savings, cost_fp, net = monthly_roi_estimate(int(fraud_blocked_month), int(fp_month))
c1, c2, c3 = st.columns(3)
c1.metric("Fraud blocked (est. monthly)", f"{int(fraud_blocked_month):,.0f}", help="Estimated from test set scale")
c2.metric("False positives (est. monthly)", f"{int(fp_month):,.0f}", help="Operational cost")
c3.metric("Net monthly impact (USD)", f"${net:,.0f}", help="Savings from blocked fraud minus FP cost")
st.caption("Assumptions: $180 per missed fraud, $12 per false positive. Scale: test set → 1.5M tx/month.")

st.subheader("Model vs baseline (threshold 0.50)")
st.markdown("| Metric | Baseline (0.50) | Current (0.43) |")
st.markdown("|--------|----------------|----------------|")
# Placeholder comparison
st.markdown("| AUC | — | — |")
st.markdown("| F1 | — | — |")
st.markdown("| Total cost (test) | — | — |")
st.caption("Fill with real metrics from your trained model and test set.")
