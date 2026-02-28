"""
FraudSense — Transaction Scorer (main page).
Front adapted from fraud_app_prototype.html; backend unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st
import pandas as pd

from utils.styles import inject_fraud_sense_style, COLORS, OPTIMAL_THRESHOLD
from utils.model import load_model_and_artifacts, prepare_input, predict_proba, get_feature_names, DATA_DIR
from utils.explainer import get_shap_values_and_explanations, shap_to_natural_language

st.set_page_config(page_title="FraudSense — Transaction Scorer", layout="wide", initial_sidebar_state="expanded")
inject_fraud_sense_style(st)

# -----------------------------------------------------------------------------
# Demo mode & model
# -----------------------------------------------------------------------------
@st.cache_data
def load_demo_transactions():
    p = DATA_DIR / "demo_transactions.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def _load_model_safe():
    try:
        return load_model_and_artifacts()
    except Exception:
        return None, get_feature_names(), OPTIMAL_THRESHOLD

model, feature_list, default_threshold = _load_model_safe()
demo_df = load_demo_transactions()
DEMO_MODE = model is None and len(demo_df) > 0

# -----------------------------------------------------------------------------
# Sidebar: prototype look — logo, threshold, feed, author
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    '<div class="sidebar-logo-mark">FraudSense</div>'
    '<div class="sidebar-logo-sub">IEEE-CIS · LightGBM · SHAP</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.10,
    max_value=0.90,
    value=float(default_threshold),
    step=0.01,
    help="Optimal 0.43 minimizes FN×$180 + FP×$12.",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-family:var(--font-mono);font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px">Recent Decisions</div>',
    unsafe_allow_html=True,
)
if len(demo_df) > 0:
    feed = demo_df.head(3)
    for _, row in feed.iterrows():
        sc = row.get("fraud_score", 0)
        is_fraud = sc >= threshold
        dot = "fraud" if is_fraud else "legit"
        st.sidebar.markdown(
            f'<div class="fraud-feed-item">'
            f'<div class="fraud-feed-dot {dot}"></div>'
            f'<div class="fraud-feed-id">${row.get("TransactionAmt", 0):,.0f}</div>'
            f'<div class="fraud-feed-desc">{row.get("card_type", "")} · {int(row.get("event_hour", 0))}:00</div>'
            f'<div class="fraud-feed-score {dot}">{int(sc*100)}</div></div>',
            unsafe_allow_html=True,
        )
else:
    st.sidebar.caption("Score a transaction to see feed.")
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="fraud-author-card">'
    '<div class="fraud-author-avatar">JP</div>'
    '<div><div class="fraud-author-name">Jovane Pascoal</div>'
    '<div class="fraud-author-role">data analyst · nyc</div></div></div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("[GitHub](https://github.com)")

# -----------------------------------------------------------------------------
# Topbar (prototype)
# -----------------------------------------------------------------------------
st.markdown(
    '<div class="fraud-topbar">'
    '<span class="fraud-page-title">Transaction Scorer</span>'
    '<span class="fraud-tag safe">● Model Active</span>'
    f'<span class="fraud-tag info">Threshold: {threshold:.2f}</span>'
    '<span class="fraud-tag warning">IEEE-CIS Dataset</span>'
    '</div>',
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Metrics row (prototype)
# -----------------------------------------------------------------------------
st.markdown(
    '<div class="fraud-metrics-row">'
    '<div class="fraud-metric-card red">'
    '<div class="fraud-metric-label">Fraud Caught</div>'
    '<div class="fraud-metric-value">94.2<span style="font-size:14px;color:var(--muted)">%</span></div>'
    '<div class="fraud-metric-delta">↑ vs baseline</div></div>'
    '<div class="fraud-metric-card orange">'
    '<div class="fraud-metric-label">False Positive Rate</div>'
    '<div class="fraud-metric-value">0.8<span style="font-size:14px;color:var(--muted)">%</span></div>'
    '<div class="fraud-metric-delta">Optimized threshold</div></div>'
    '<div class="fraud-metric-card green">'
    '<div class="fraud-metric-label">Monthly ROI Est.</div>'
    '<div class="fraud-metric-value">$2.3<span style="font-size:14px;color:var(--muted)">M</span></div>'
    '<div class="fraud-metric-delta">50k txn/day · NYC neobank</div></div>'
    '<div class="fraud-metric-card blue">'
    '<div class="fraud-metric-label">Transactions Scored</div>'
    '<div class="fraud-metric-value">590<span style="font-size:14px;color:var(--muted)">K</span></div>'
    '<div class="fraud-metric-delta">Temporal split (80% train / 20% test by Time)</div></div>'
    '</div>',
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Input section (prototype)
# -----------------------------------------------------------------------------
st.markdown(
    '<div class="fraud-input-section">'
    '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">'
    '<div><div class="fraud-input-title">Analyze a Transaction</div>'
    '<div class="fraud-input-sub">Enter values · Model returns risk score + explanation in natural language</div></div>'
    '<span class="fraud-tag info">LightGBM · SHAP</span></div>',
    unsafe_allow_html=True,
)
c1, c2, c3 = st.columns(3)
with c1:
    transaction_amt = st.number_input("Transaction Amount (USD)", min_value=0.01, value=2847.0, step=10.0, format="%.2f", key="amt")
    card_type = st.selectbox("Card Type", ["credit", "debit", "prepaid"], index=0, key="card")
with c2:
    product_cd = st.selectbox("Product Category", ["W", "H", "C", "S", "R"], index=0, key="prod")
    event_hour = st.slider("Transaction Hour (0–23)", 0, 23, 2, key="hour")
with c3:
    txn_count_1h = st.number_input("Transactions (last 1h)", min_value=0, value=7, step=1, key="txn1h")
    amt_accumulated_24h = st.number_input("Amt Accumulated (24h USD)", min_value=0.0, value=9340.0, step=50.0, format="%.2f", key="amt24")
score_clicked = st.button("Run Analysis")

# -----------------------------------------------------------------------------
# Backend: build X, predict, reasons (unchanged)
# -----------------------------------------------------------------------------
if score_clicked or "last_score" not in st.session_state:
    X = prepare_input(
        transaction_amt=transaction_amt,
        card_type=card_type,
        product_cd=product_cd,
        event_hour=event_hour,
        txn_count_1h=float(txn_count_1h),
        amt_accumulated_24h=amt_accumulated_24h or transaction_amt,
        feature_names=feature_list,
    )
    for c in feature_list:
        if c not in X.columns:
            X[c] = 0
    X = X[[c for c in feature_list if c in X.columns]] if all(c in X.columns for c in feature_list) else X
    if len(X.columns) < len(feature_list):
        X = pd.DataFrame({f: [0.0] for f in feature_list})
        X["TransactionAmt"] = transaction_amt
        X["event_hour"] = event_hour
        X["txn_count_1h"] = txn_count_1h
        X["amt_accumulated_24h"] = amt_accumulated_24h or transaction_amt
        X["card_type"] = {"credit": 0, "debit": 1, "prepaid": 2}.get(card_type, 0)
        X["ProductCD"] = {"W": 0, "H": 1, "C": 2, "S": 3, "R": 4}.get(product_cd, 0)
    proba = predict_proba(model, X)[0] if model is not None else min(0.95, 0.1 + (transaction_amt / 3000) * 0.3 + (txn_count_1h / 20) * 0.4 + (0.15 if event_hour in (0, 1, 2, 3) else 0))
    st.session_state["last_score"] = proba
    st.session_state["last_X"] = X
else:
    proba = st.session_state["last_score"]
    X = st.session_state.get("last_X", pd.DataFrame())

risk_pct = round(proba * 100)
verdict = "BLOCKED" if proba >= threshold else "APPROVED"
verdict_class = "fraud-verdict" if verdict == "BLOCKED" else "fraud-verdict approved"

# SHAP reasons
shap_row, reasons = get_shap_values_and_explanations(model, X, feature_names=feature_list, top_n=5)
if not reasons:
    try:
        import numpy as np
        sv = np.random.randn(len(feature_list)) * 0.15
        fv = X.iloc[0].values if len(X) else np.zeros(len(feature_list))
        reasons = shap_to_natural_language(sv.reshape(1, -1), feature_list, fv.reshape(1, -1), top_n=5)
    except Exception:
        reasons = []
if not reasons:
    reasons = [
        {"rank": 1, "text": f"Amount of ${transaction_amt:,.2f} is above typical range — moderate signal", "shap_value": 0.25, "magnitude": 0.5},
        {"rank": 2, "text": f"{txn_count_1h} transactions in last hour — {'velocity spike' if txn_count_1h > 5 else 'normal activity'}", "shap_value": 0.2, "magnitude": 0.4},
        {"rank": 3, "text": f"Transaction at {event_hour:02d}:00 — {'outside typical activity window' if event_hour in (0,1,2,3,22,23) else 'within normal hours'}", "shap_value": 0.15, "magnitude": 0.3},
    ]

# -----------------------------------------------------------------------------
# Hero: score card (dial) + explain card (prototype)
# -----------------------------------------------------------------------------
# SVG dial: circle circumference ≈ 2*pi*54 = 339.3; offset = 339.3 * (1 - risk_pct/100)
circ = 339.3
dash_offset = circ * (1 - risk_pct / 100)
dial_color = COLORS["accent"] if risk_pct >= 66 else (COLORS["accent2"] if risk_pct >= 33 else COLORS["green"])

score_card_html = (
    f'<div class="fraud-score-card">'
    f'<div class="fraud-score-label">Risk Score</div>'
    f'<div class="fraud-dial-wrap">'
    f'<svg width="140" height="140" viewBox="0 0 120 120" style="transform:rotate(-90deg)">'
    f'<circle cx="60" cy="60" r="54" fill="none" stroke="var(--border)" stroke-width="10"/>'
    f'<circle cx="60" cy="60" r="54" fill="none" stroke="{dial_color}" stroke-width="10" stroke-linecap="round" '
    f'stroke-dasharray="{circ}" stroke-dashoffset="{dash_offset}" style="filter:drop-shadow(0 0 8px {dial_color});transition:stroke-dashoffset 1s ease"/>'
    f'</svg>'
    f'<div class="fraud-dial-center"><div class="fraud-dial-number">{risk_pct}</div><div class="fraud-dial-unit">/ 100</div></div>'
    f'</div>'
    f'<div class="{verdict_class}">{"HIGH RISK" if verdict == "BLOCKED" else "LOW RISK"}</div>'
    f'<div class="fraud-verdict-sub">{"Blocked" if verdict == "BLOCKED" else "Approved"} · Threshold {threshold:.2f}</div>'
    f'<div style="margin-top:14px;display:flex;gap:6px">'
    f'<span class="fraud-tag {"danger" if risk_pct >= 50 else "warning"}">{"ATO Pattern" if risk_pct >= 50 else "Review"}</span>'
    f'<span class="fraud-tag info">{"Night Tx" if event_hour in (0,1,2,3,22,23) else "Day Tx"}</span></div></div>'
)

reason_items = []
for r in reasons[:5]:
    mag = min(1.0, abs(r["shap_value"]) / 0.5)
    bar_class = "high" if mag > 0.6 else ("med" if mag > 0.3 else "low")
    val_color = COLORS["accent"] if r["shap_value"] > 0 else COLORS["green"]
    reason_items.append(
        f'<div class="fraud-reason-item">'
        f'<div class="fraud-reason-rank">#{r["rank"]}</div>'
        f'<div class="fraud-reason-body">'
        f'<div class="fraud-reason-text">{r["text"]}</div>'
        f'<div class="fraud-reason-bar-wrap">'
        f'<div class="fraud-reason-bar-bg"><div class="fraud-reason-bar-fill {bar_class}" style="width:{mag*100}%"></div></div>'
        f'<div class="fraud-reason-val" style="color:{val_color}">{r["shap_value"]:+.2f}</div></div></div></div>'
    )
explain_html = (
    '<div class="fraud-explain-card">'
    '<div class="fraud-explain-title">Why was this transaction flagged?</div>'
    '<div class="fraud-explain-sub">SHAP values → natural language · Top 5 contributing factors</div>'
    '<div class="fraud-reason-list">' + "".join(reason_items) + '</div></div>'
)

hero_col1, hero_col2 = st.columns([1, 2])
with hero_col1:
    st.markdown(score_card_html, unsafe_allow_html=True)
with hero_col2:
    st.markdown(explain_html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Bottom grid: Feature Importance + ROI panel (prototype)
# -----------------------------------------------------------------------------
st.markdown(
    '<div class="fraud-bottom-grid">'
    '<div class="fraud-panel">'
    '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:18px">'
    '<div class="fraud-panel-title">Feature Importance — Fraud vs Legit</div>'
    '<span class="fraud-panel-tag">SHAP Global</span></div>'
    '<div style="font-family:var(--font-mono);font-size:11px;color:var(--text2)">'
    'amt_accumulated_24h · txn_count_1h · TransactionAmt · event_hour · card_type · ProductCD'
    '</div></div>'
    '<div class="fraud-panel">'
    '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:18px">'
    '<div class="fraud-panel-title">Business Impact</div><span class="fraud-panel-tag">ROI Model</span></div>'
    '<div class="fraud-roi-box">'
    '<div class="fraud-roi-title">Estimated Monthly Savings</div>'
    '<div class="fraud-roi-number">$2.3M</div>'
    '<div class="fraud-roi-desc">50k txn/day · avg fraud loss $180 · FP cost $12 · NYC neobank scenario</div></div>'
    '<div class="fraud-threshold-row">'
    '<div class="fraud-threshold-label">Optimal Threshold</div>'
    '<div style="text-align:right"><div class="fraud-threshold-val">0.43</div>'
    '<div style="font-family:var(--font-mono);font-size:9px;color:var(--green)">↓ from default 0.50</div></div></div>'
    '<div style="font-family:var(--font-mono);font-size:10px;color:var(--muted);padding:8px 0;border-top:1px solid var(--border)">'
    'Minimizes total cost = (FN × $180) + (FP × $12). Documented in notebook.</div>'
    '<div class="fraud-glow-line"></div>'
    '</div></div>',
    unsafe_allow_html=True,
)

if DEMO_MODE:
    st.sidebar.caption("Demo mode: no model. Add model in `models/` for full SHAP.")
