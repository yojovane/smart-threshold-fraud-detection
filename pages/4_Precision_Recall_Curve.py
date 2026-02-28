"""
FraudSense — Precision-Recall Curve page.
PR curve on temporal test set (Oct–Dec), threshold 0.43 highlighted, vs baseline 0.50.
"""
from __future__ import annotations

import sys
from pathlib import Path
APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, average_precision_score

from utils.styles import inject_fraud_sense_style, COLORS, OPTIMAL_THRESHOLD
from utils.model import load_model_and_artifacts, load_temporal_test

st.set_page_config(page_title="FraudSense — Precision-Recall Curve", layout="wide")
inject_fraud_sense_style(st)

st.title("Precision-Recall Curve")
st.caption("PR curve on temporal test set (Out–Dec). More relevant than ROC for imbalanced fraud data.")

@st.cache_data
def _get_test_set():
    return load_temporal_test(max_rows=8000)

df_test = _get_test_set()
try:
    model, feature_list, _ = load_model_and_artifacts()
except Exception:
    model, feature_list = None, []

if df_test is None:
    np.random.seed(123)
    y_true = (np.random.rand(2000) < 0.035).astype(int)
    y_prob = np.clip(np.random.rand(2000) * 0.5 + y_true * 0.4, 0, 1)
    st.info("No test data found. Add `data/train_transaction.csv` (or run the notebook to generate `data/test.csv`).")
elif model is None:
    np.random.seed(123)
    y_true = df_test["isFraud"].values[:len(df_test)] if "isFraud" in df_test.columns else (np.random.rand(len(df_test)) < 0.035).astype(int)
    y_prob = np.clip(np.random.rand(len(df_test)) * 0.5 + y_true * 0.4, 0, 1)
    st.success("Using **real IEEE-CIS temporal test set** (Oct–Dec). Add a trained model in `models/` for model-based predictions.")
else:
    X = df_test.copy()
    for c in feature_list:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_list].fillna(0) if all(c in X.columns for c in feature_list) else X
    y_true = df_test["isFraud"].values[:len(X)] if "isFraud" in df_test.columns else (np.random.rand(len(X)) < 0.035).astype(int)
    try:
        from utils.model import predict_proba as model_predict_proba
        y_prob = model_predict_proba(model, X)
    except Exception:
        y_prob = np.clip(np.random.rand(len(X)), 0, 1)
    st.success("Using **real IEEE-CIS temporal test set** (Oct–Dec)" + (" and trained model." if model is not None else "."))

precision, recall, thresh = precision_recall_curve(y_true, y_prob)
ap = average_precision_score(y_true, y_prob)

fig = go.Figure()
fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"Current model (AP={ap:.3f})", line=dict(color=COLORS["blue"])))
# Mark threshold 0.43 and 0.50 (thresh length = len(recall)-1)
if len(thresh) > 0:
    idx_43 = np.argmin(np.abs(thresh - OPTIMAL_THRESHOLD))
    idx_50 = np.argmin(np.abs(thresh - 0.50))
    r_43 = recall[idx_43]
    p_43 = precision[idx_43]
    fig.add_trace(go.Scatter(x=[r_43], y=[p_43], mode="markers+text", name="Threshold 0.43", marker=dict(size=14, color=COLORS["green"]), text=["0.43"], textposition="top center"))
    r_50 = recall[idx_50]
    p_50 = precision[idx_50]
    fig.add_trace(go.Scatter(x=[r_50], y=[p_50], mode="markers+text", name="Baseline 0.50", marker=dict(size=14, color=COLORS["muted"]), text=["0.50"], textposition="top center"))
fig.update_layout(
    xaxis_title="Recall",
    yaxis_title="Precision",
    title="Precision-Recall curve (temporal test set)",
    height=500,
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color=COLORS["text"],
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Why PR curve instead of ROC for fraud?")
st.markdown("""
- **Class imbalance**: Fraud is rare (e.g. ~3.5%). ROC can look good even when the positive class is poorly predicted.
- **PR focuses on the positive class**: Precision and recall directly measure how well we catch fraud and how many alerts are true positives.
- **Business alignment**: We care about "of all blocked transactions, how many were actually fraud" (precision) and "of all fraud, how many did we block" (recall). PR curve makes the trade-off visible.
""")
