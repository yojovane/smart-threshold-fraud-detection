"""
Model loader for FraudSense. Supports LightGBM (.pkl, .txt) and demo mode
when no model is present (pre-computed scores from test.csv or sample).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

# Paths relative to project root (parent of utils/)
APP_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = APP_DIR / "models"
DATA_DIR = APP_DIR / "data"

MODEL_PKL = MODELS_DIR / "lgbm_fraud_v2.pkl"
MODEL_TXT = MODELS_DIR / "gbm.txt"
MODEL_JOB = MODELS_DIR / "gbm.joblib"
FEATURES_JOB = MODELS_DIR / "features_list.joblib"
THRESHOLD_JSON = MODELS_DIR / "threshold.json"

# Default optimal threshold: minimizes (FN * $180) + (FP * $12)
DEFAULT_THRESHOLD = 0.43
REFERENCE_DT = pd.Timestamp("2017-12-01 00:00:00")  # IEEE-CIS TransactionDT reference


def load_temporal_test(max_rows: int = 8000) -> Optional[pd.DataFrame]:
    """
    Load temporal test set (Oct–Dec) for PR curve / ROI.
    Tries: test.csv → train.csv → build from train_transaction.csv (TransactionDT, month >= 10).
    """
    # 1. Pre-built splits
    for name in ("test.csv", "train.csv"):
        p = DATA_DIR / name
        if p.exists():
            return pd.read_csv(p, nrows=max_rows)
    # 2. Build from raw IEEE-CIS
    p = DATA_DIR / "train_transaction.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, nrows=max_rows * 2)  # read extra so after filter we have ~max_rows
        if "TransactionDT" not in df.columns:
            return None
        df["transaction_date"] = REFERENCE_DT + pd.to_timedelta(df["TransactionDT"], unit="s")
        df["transaction_month"] = df["transaction_date"].dt.month
        df["event_hour"] = df["transaction_date"].dt.hour
        test = df[df["transaction_month"] >= 10].head(max_rows)
        if "TransactionAMt" in test.columns and "TransactionAmt" not in test.columns:
            test["TransactionAmt"] = test["TransactionAMt"]
        return test
    except Exception:
        return None


def get_feature_names() -> List[str]:
    """Expected feature order for inference (matches training)."""
    return [
        "TransactionAmt",
        "event_hour",
        "txn_count_1h",
        "amt_accumulated_24h",
        "card_type",
        "ProductCD",
    ]


def load_model_and_artifacts() -> Tuple[Any, List[str], float]:
    """
    Load model, feature list, and default threshold.
    Returns (model, feature_names, threshold).
    If no model exists, returns (None, get_feature_names(), DEFAULT_THRESHOLD) for demo mode.
    """
    feature_list = get_feature_names()
    if FEATURES_JOB.exists():
        feature_list = joblib.load(FEATURES_JOB)

    thr = DEFAULT_THRESHOLD
    if THRESHOLD_JSON.exists():
        try:
            with open(THRESHOLD_JSON, "r") as f:
                thr = float(json.load(f).get("threshold", DEFAULT_THRESHOLD))
        except Exception:
            pass

    model = None
    if lgb is not None:
        if MODEL_TXT.exists():
            model = lgb.Booster(model_file=str(MODEL_TXT))
        elif MODEL_JOB.exists():
            model = joblib.load(MODEL_JOB)
        elif MODEL_PKL.exists():
            model = joblib.load(MODEL_PKL)

    return model, feature_list, thr


def prepare_input(
    transaction_amt: float,
    card_type: str,
    product_cd: str,
    event_hour: int,
    txn_count_1h: float = 0.0,
    amt_accumulated_24h: float = 0.0,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build a single-row DataFrame for prediction (same column order as training)."""
    if feature_names is None:
        feature_names = get_feature_names()
    # Map categoricals to numeric if needed (model may expect encoded)
    card_map = {"credit": 0, "debit": 1, "prepaid": 2}
    product_map = {"W": 0, "H": 1, "C": 2, "S": 3, "R": 4}
    row = {
        "TransactionAmt": transaction_amt,
        "event_hour": event_hour,
        "txn_count_1h": txn_count_1h,
        "amt_accumulated_24h": amt_accumulated_24h if amt_accumulated_24h else transaction_amt,
        "card_type": card_map.get(card_type.lower(), 0),
        "ProductCD": product_map.get(product_cd.upper(), 0),
    }
    return pd.DataFrame([row])[feature_names] if all(k in row for k in feature_names) else pd.DataFrame([row])


def predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return probability of positive class (fraud)."""
    if model is None:
        return np.array([0.0] * len(X))
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # LightGBM Booster
    try:
        n = getattr(model, "best_iteration", None)
        if n is None:
            n = getattr(model, "num_trees", 100)
        if callable(n):
            n = n()
        return model.predict(X, num_iteration=n)
    except Exception:
        return model.predict(X)
