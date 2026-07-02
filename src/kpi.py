# src/kpi.py
import numpy as np

def branch_totals(sales: np.ndarray) -> np.ndarray:
    """Total revenue per branch across all 30 days. Shape: (8,)"""
    return np.sum(sales, axis=0)          # collapse rows -> one total per branch

def branch_daily_avg(sales: np.ndarray) -> np.ndarray:
    """Mean daily revenue per branch. Shape: (8,)"""
    return np.mean(sales, axis=0)

def branch_volatility(sales: np.ndarray) -> np.ndarray:
    """
    Coefficient of Variation (std / mean) per branch.
    Higher value = more volatile / unpredictable branch.
    Uses np.std and np.mean ufuncs.
    """
    return np.std(sales, axis=0) / np.mean(sales, axis=0)

def flag_underperformers(sales: np.ndarray, threshold_pct: float = 0.5) -> np.ndarray:
    """
    For each (day, branch) cell, flag True if sales fell below
    (threshold_pct * that branch's mean).

    Uses np.where and broadcasting.
    Returns boolean mask of shape (30, 8).
    """
    branch_means = np.mean(sales, axis=0)          # shape (8,)
    threshold    = branch_means * threshold_pct    # shape (8,)
    return sales < threshold                        # boolean mask (30, 8)

def log_scaled_revenue(sales: np.ndarray) -> np.ndarray:
    """
    Apply natural log to compress the revenue scale.
    Useful for visualisation and ML feature prep.
    """
    return np.log(sales)                            # ufunc applied element-wise