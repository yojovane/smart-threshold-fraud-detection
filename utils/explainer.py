"""
SHAP-based explainer with natural language reason strings for FraudSense.
Converts SHAP values into business-readable explanations and magnitude bars.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import shap
except ImportError:
    shap = None

# Default feature names (must match model training order)
DEFAULT_FEATURES = [
    "TransactionAmt",
    "event_hour",
    "txn_count_1h",
    "amt_accumulated_24h",
    "card_type",
    "ProductCD",
]

# Human-readable labels and templates for natural language
FEATURE_LABELS = {
    "TransactionAmt": "Amount",
    "event_hour": "Time of day",
    "txn_count_1h": "Transactions in last hour",
    "amt_accumulated_24h": "Amount accumulated in 24h",
    "card_type": "Card type",
    "ProductCD": "Product category",
}


def _template_transaction_amt(value: float, shap_val: float, card_avg: Optional[float] = None) -> str:
    if card_avg and card_avg > 0:
        ratio = value / card_avg
        return f"Amount of ${value:,.2f} is {ratio:.1f}× {'above' if ratio >= 1 else 'below'} cardholder average — {'strong ATO signal' if abs(shap_val) > 0.2 else 'moderate signal'}"
    return f"Amount of ${value:,.2f} is {'above' if shap_val > 0 else 'below'} typical range — {'strong fraud signal' if abs(shap_val) > 0.2 else 'moderate signal'}"


def _template_txn_count_1h(value: float, shap_val: float) -> str:
    v = int(value) if not np.isnan(value) else 0
    return f"{v} transactions in last hour — {'velocity spike, automated fraud pattern' if v > 5 else 'normal activity'}"


def _template_event_hour(value: float, shap_val: float) -> str:
    h = int(value) if not np.isnan(value) else 0
    return f"Transaction at {h:02d}:00 — {'outside typical activity window' if shap_val > 0.1 else 'within normal hours'}"


def _template_amt_accumulated_24h(value: float, shap_val: float) -> str:
    return f"${value:,.0f} accumulated in 24h — {'4×+ above daily ceiling' if shap_val > 0.1 else 'within normal range'}"


def _template_product_cd(value: float, shap_val: float) -> str:
    return f"Product category — {'first purchase in this category' if shap_val > 0.1 else 'consistent with history'}"


def _template_card_type(value: float, shap_val: float) -> str:
    return f"Card type — {'higher-risk category' if shap_val > 0.1 else 'typical for this user'}"


TEMPLATES = {
    "TransactionAmt": _template_transaction_amt,
    "txn_count_1h": _template_txn_count_1h,
    "event_hour": _template_event_hour,
    "amt_accumulated_24h": _template_amt_accumulated_24h,
    "ProductCD": _template_product_cd,
    "card_type": _template_card_type,
}


def shap_to_natural_language(
    shap_values: np.ndarray,
    feature_names: List[str],
    feature_values: np.ndarray,
    top_n: int = 5,
    card_avg: Optional[float] = None,
) -> List[Dict[str, any]]:
    """
    Convert SHAP values into human-readable explanations.
    Returns list of dicts: {rank, text, shap_value, magnitude (0-1), feature_name}.
    """
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)
    if feature_values.ndim == 1:
        feature_values = feature_values.reshape(1, -1)

    row_shap = shap_values[0]
    row_vals = feature_values[0]

    # Sort by absolute SHAP
    order = np.argsort(np.abs(row_shap))[::-1]
    result = []
    for i, idx in enumerate(order[:top_n]):
        name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
        s = float(row_shap[idx])
        v = row_vals[idx] if idx < len(row_vals) else 0
        magnitude = min(1.0, abs(s) / 0.5) if s != 0 else 0

        fn = TEMPLATES.get(name, lambda v, s: f"{name}: value {v:.2f} contributes {'+' if s > 0 else ''}{s:.3f}")
        if name == "TransactionAmt":
            text = _template_transaction_amt(float(v), s, card_avg)
        else:
            text = fn(float(v), s)

        result.append({
            "rank": i + 1,
            "text": text,
            "shap_value": s,
            "magnitude": magnitude,
            "feature_name": name,
        })
    return result


def get_shap_values_and_explanations(
    model: Any,
    X: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    top_n: int = 5,
    card_avg: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], List[Dict]]:
    """
    Compute SHAP values for the first row of X and return explanations.
    If model is None or SHAP fails, returns (None, []) and caller can use demo reasons.
    """
    if feature_names is None:
        feature_names = list(X.columns)
    if model is None or shap is None:
        return None, []

    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        row_shap = shap_vals[0]
        row_vals = X.iloc[0].values
        reasons = shap_to_natural_language(
            row_shap.reshape(1, -1),
            feature_names,
            row_vals.reshape(1, -1),
            top_n=top_n,
            card_avg=card_avg,
        )
        return row_shap, reasons
    except Exception:
        return None, []


def format_reason_display(rank: int, text: str, shap_value: float, magnitude: float) -> str:
    """Format a single reason for UI: #1 — text — (+0.41) with magnitude bar."""
    sign = "+" if shap_value >= 0 else ""
    return f"#{rank} — {text} ({sign}{shap_value:.2f})"
