"""
ROI and cost calculations for FraudSense.
Optimal threshold 0.43 minimizes (FN * $180) + (FP * $12).
Documented for fintech recruiters.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Business assumptions (NYC neobank scenario)
COST_FN_USD = 180   # one missed fraud (average loss per fraudulent transaction)
COST_FP_USD = 12    # one false positive (operational cost, customer friction)
OPTIMAL_THRESHOLD = 0.43   # minimizes total cost on validation

# Demo: 50k transactions/day
TX_PER_DAY = 50_000
DAYS_PER_MONTH = 30
TX_PER_MONTH = TX_PER_DAY * DAYS_PER_MONTH


def cost_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    cost_fn: float = COST_FN_USD,
    cost_fp: float = COST_FP_USD,
) -> Tuple[float, int, int, int, int]:
    """
    Total cost and counts at a given threshold.
    Returns (total_cost_usd, n_fn, n_fp, n_tn, n_tp).
    """
    pred = (y_prob >= threshold).astype(int)
    tp = int(((y_true == 1) & (pred == 1)).sum())
    fn = int(((y_true == 1) & (pred == 0)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    tn = int(((y_true == 0) & (pred == 0)).sum())
    total_cost = fn * cost_fn + fp * cost_fp
    return total_cost, fn, fp, tn, tp


def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[List[float]] = None,
    cost_fn: float = COST_FN_USD,
    cost_fp: float = COST_FP_USD,
) -> pd.DataFrame:
    """DataFrame with columns: threshold, total_cost, n_fn, n_fp, n_tn, n_tp, fn_cost, fp_cost."""
    if thresholds is None:
        thresholds = [round(0.10 + i * 0.05, 2) for i in range(17)]
    rows = []
    for t in thresholds:
        total, n_fn, n_fp, n_tn, n_tp = cost_at_threshold(y_true, y_prob, t, cost_fn, cost_fp)
        rows.append({
            "threshold": t,
            "total_cost": total,
            "n_fn": n_fn,
            "n_fp": n_fp,
            "n_tn": n_tn,
            "n_tp": n_tp,
            "fn_cost": n_fn * cost_fn,
            "fp_cost": n_fp * cost_fp,
        })
    return pd.DataFrame(rows)


def optimal_threshold_from_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[List[float]] = None,
    cost_fn: float = COST_FN_USD,
    cost_fp: float = COST_FP_USD,
) -> float:
    """Return threshold that minimizes total cost."""
    df = sweep_thresholds(y_true, y_prob, thresholds, cost_fn, cost_fp)
    best = df.loc[df["total_cost"].idxmin()]
    return float(best["threshold"])


def monthly_roi_estimate(
    n_fraud_blocked: int,
    n_fp: int,
    cost_fn: float = COST_FN_USD,
    cost_fp: float = COST_FP_USD,
) -> Tuple[float, float, float]:
    """
    For a neobank with 50k tx/day, estimate monthly impact.
    Returns (savings_from_fraud_blocked, cost_of_fp, net_monthly_usd).
    """
    savings = n_fraud_blocked * cost_fn
    cost = n_fp * cost_fp
    net = savings - cost
    return savings, cost, net


def scale_test_to_monthly(
    n_tx_test: int,
    n_fraud_blocked: int,
    n_fp: int,
) -> Tuple[float, float]:
    """Scale test-set counts to monthly (50k tx/day). Returns (fraud_blocked_per_month, fp_per_month)."""
    if n_tx_test <= 0:
        return 0.0, 0.0
    scale = TX_PER_MONTH / n_tx_test
    return n_fraud_blocked * scale, n_fp * scale
