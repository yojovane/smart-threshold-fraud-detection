"""
app.py — Streamlit Fraud Detection Demo
---------------------------------------
Purpose
    Demonstrate a production-leaning fraud detection workflow focused on
    minimizing false positives (FPR) while preserving recall — a critical
    business trade-off for fintech and e-commerce.

Audience
    Risk analysts, fraud leads, and hiring managers evaluating real-world,
    end-to-end data science skills (from validation choices to explainability).

What this app does
    • Loads a trained LightGBM model and the exact training feature list.
    • Lets users upload a CSV with the same schema baseline used in training.
    • Applies a tunable probability threshold (slider) to reflect business policy.
    • Generates SHAP-based "reason codes" per flagged transaction for auditability.
    • Provides a downloadable CSV with predictions + explanations.

How to run
    1) Ensure model artifacts are under ./models:
        - gbm.txt (native LightGBM) OR gbm.joblib
        - features_list.joblib  (exact feature order used in training)
        - threshold.json        (optional; default 0.50 if missing)
        - scaler.joblib         (optional; not typically needed for LightGBM)
    2) pip install -r requirements.txt
    3) streamlit run app.py

Notes
    • This code mirrors training-time feature expectations (e.g., Time → event_hour/day).
    • Threshold tuning is business-driven: lower threshold → more recall, higher FP;
      higher threshold → fewer FP, potential FN increase. Pick based on cost & SLA.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt


# =============================================================================
# Streamlit config & header
# =============================================================================
st.set_page_config(page_title="Fraud Detection — Low FPR", layout="wide")
st.title("🚀 Fraud Detection — Low False Positive Focus")

st.markdown(
    """
This demo loads a **LightGBM** model trained to detect fraudulent transactions while
**minimizing false positives**. Upload a CSV aligned with training columns, adjust the
decision threshold, and review **SHAP reason codes** for flagged rows.
"""
)


# =============================================================================
# Artifact paths
# =============================================================================
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
MODEL_TXT = MODELS_DIR / "gbm.txt"               # optional (preferred for portability)
MODEL_JOB = MODELS_DIR / "gbm.joblib"            # fallback (pickled booster)
FEATURES_JOB = MODELS_DIR / "features_list.joblib"
THRESHOLD_JSON = MODELS_DIR / "threshold.json"   # optional
SCALER_JOB = MODELS_DIR / "scaler.joblib"        # optional (rarely needed for LGBM)


# =============================================================================
# Load artifacts (cached)
# =============================================================================
@st.cache_resource
def load_artifacts():
    """
    Load model, features, default threshold, and optional scaler.
    Prefers native LightGBM text model for portability; falls back to joblib.
    """
    # Model
    if MODEL_TXT.exists():
        model = lgb.Booster(model_file=str(MODEL_TXT))
    elif MODEL_JOB.exists():
        model = joblib.load(MODEL_JOB)
    else:
        raise FileNotFoundError(
            "Model not found. Add 'gbm.txt' (preferred) or 'gbm.joblib' under ./models/"
        )

    # Feature list (required: preserves exact training order)
    if not FEATURES_JOB.exists():
        raise FileNotFoundError("Missing 'features_list.joblib' in ./models/")
    feature_list: List[str] = joblib.load(FEATURES_JOB)

    # Default threshold (optional)
    thr_default = 0.50
    if THRESHOLD_JSON.exists():
        try:
            with open(THRESHOLD_JSON, "r") as f:
                thr_default = float(json.load(f).get("threshold", 0.50))
        except Exception:
            # keep safe default if malformed
            thr_default = 0.50

    # Optional scaler (only if used during training)
    scaler = joblib.load(SCALER_JOB) if SCALER_JOB.exists() else None

    return model, feature_list, thr_default, scaler


try:
    model, feature_list, thr_default, scaler = load_artifacts()
except Exception as e:
    st.error(f"Artifact loading error: {e}")
    st.stop()


# =============================================================================
# Inference-time preprocessing (must mirror training assumptions)
# =============================================================================
def preprocess_inference(df_raw: pd.DataFrame, feature_list: List[str], scaler=None) -> pd.DataFrame:
    """
    Minimal, deterministic transformations to align with training:
      • Time → event_hour, event_day (if 'Time' exists)
      • Create any missing training columns (filled with 0)
      • Drop extra columns and preserve EXACT training order
      • Optionally apply scaler (only if used at training-time)
    """
    df = df_raw.copy()

    # Time-derived helpers (consistent with notebook)
    if "Time" in df.columns:
        start = pd.Timestamp("2013-01-01")
        df["event_dt"] = start + pd.to_timedelta(df["Time"], unit="s")
        df["event_hour"] = df["event_dt"].dt.hour
        df["event_day"] = df["event_dt"].dt.day

    # Ensure all training columns exist
    for col in feature_list:
        if col not in df.columns:
            # Safe default for engineered numerics
            df[col] = 0

    # Keep only training columns (exact order)
    df = df[feature_list]

    # Optional scaling (rare for LGBM)
    if scaler is not None:
        df[feature_list] = scaler.transform(df[feature_list])

    return df


# =============================================================================
# Sidebar controls (business-facing)
# =============================================================================
st.sidebar.header("Decision Threshold")
threshold = st.sidebar.slider(
    "Probability threshold",
    min_value=0.01,
    max_value=0.99,
    value=float(thr_default),
    step=0.01,
    help="Lower → more recall, higher FP. Higher → fewer FP, potential FN increase. Tune to your policy.",
)


# =============================================================================
# File upload
# =============================================================================
uploaded = st.file_uploader("📂 Upload CSV (same baseline schema as training)", type=["csv"])

if uploaded is not None:
    # Read CSV
    try:
        df_raw = pd.read_csv(uploaded)
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df_raw.head())
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    # Preprocess to training schema
    try:
        X = preprocess_inference(df_raw, feature_list, scaler=scaler)
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        st.stop()

    # Predict probabilities with LightGBM Booster or sklearn-like API
    try:
        if isinstance(model, lgb.Booster):
            proba = model.predict(X, num_iteration=model.best_iteration)
        else:
            proba = model.predict_proba(X)[:, 1]
    except Exception as e:
        st.error(f"Inference error: {e}")
        st.stop()

    pred = (proba >= threshold).astype(int)

    # KPIs (business-facing summary)
    st.subheader("📊 Prediction Summary")
    total_tx = len(X)
    flagged = int(pred.sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Transactions", f"{total_tx:,}")
    c2.metric("Flagged as Fraud", f"{flagged:,}")
    c3.metric("Flag Rate", f"{(flagged / max(total_tx, 1)):.2%}")

    # SHAP explanations → reason codes
    st.subheader("🔍 Reason Codes for Flagged Transactions")
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)

        # Normalize to positive-class 2D matrix (LightGBM binary → list [neg, pos])
        if isinstance(shap_vals, list):
            shap_pos = shap_vals[1]
        else:
            shap_pos = shap_vals
        if shap_pos.ndim == 1:
            shap_pos = shap_pos.reshape(-1, 1)

        def reason_codes(row_shap: np.ndarray, feature_names: List[str], top_k: int = 3) -> str:
            idx = np.argsort(np.abs(row_shap))[::-1][:top_k]
            return ", ".join([feature_names[i] for i in idx])

        reasons = [
            reason_codes(shap_pos[i], feature_list, top_k=3) if pred[i] == 1 else ""
            for i in range(len(X))
        ]

        out = df_raw.copy()
        out["fraud_probability"] = proba
        out["fraud_pred"] = pred
        out["reason_codes"] = reasons

        st.dataframe(out.sort_values("fraud_probability", ascending=False).head(50))

        st.download_button(
            "💾 Download Predictions CSV",
            out.to_csv(index=False).encode("utf-8"),
            "fraud_predictions.csv",
            "text/csv",
        )

        # Optional: global SHAP (beeswarm) — sample for speed
        with st.expander("📈 Global Feature Importance (SHAP)"):
            sample_idx = np.random.choice(len(X), size=min(2000, len(X)), replace=False)
            fig, ax = plt.subplots(figsize=(8, 6))
            shap.summary_plot(shap_pos[sample_idx], X.iloc[sample_idx], show=False)
            st.pyplot(fig)

    except Exception as e:
        # If SHAP fails (e.g., model format mismatch), continue with predictions only
        st.warning(f"SHAP explanations skipped: {e}")
