"""
Statistical Analysis Utilities
────────────────────────────────
Welch t-test, Cohen's d, bootstrap CIs, and summary statistics used in
Section VI of the MLP paper.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats


INF = float("inf")


def summary_stats(values: List[float]) -> Dict:
    """Mean, std, min, max, and 95 % bootstrap CI on the mean."""
    arr = np.array(values, dtype=float)
    arr_clean = arr[np.isfinite(arr)]
    if len(arr_clean) == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "ci95": None}
    return {
        "mean": float(arr_clean.mean()),
        "std":  float(arr_clean.std(ddof=1)) if len(arr_clean) > 1 else 0.0,
        "min":  float(arr_clean.min()),
        "max":  float(arr_clean.max()),
        "ci95": bootstrap_ci(arr_clean),
        "n":    len(arr_clean),
    }


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 2_000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """
    Percentile bootstrap 95 % CI on the mean.

    Parameters
    ----------
    values      : 1-D array of finite values
    n_resamples : number of bootstrap resamples (paper uses 2 000)
    alpha       : significance level
    rng         : numpy random generator (for reproducibility)
    """
    rng = rng or np.random.default_rng(0)
    means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_resamples)
    ])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def welch_ttest(
    a: List[float],
    b: List[float],
) -> Dict:
    """
    Two-sided Welch t-test comparing group `a` (baseline) with group `b`.

    Returns dict with: t_stat, p_val, cohen_d, stars, n_a, n_b.
    Significance stars: * p<.05, ** p<.01, *** p<.001
    """
    a_arr = np.array([v for v in a if np.isfinite(v)], dtype=float)
    b_arr = np.array([v for v in b if np.isfinite(v)], dtype=float)

    if len(a_arr) < 2 or len(b_arr) < 2:
        return {"t_stat": None, "p_val": None, "cohen_d": None, "stars": ""}

    t_stat, p_val = scipy_stats.ttest_ind(b_arr, a_arr, equal_var=False)
    d = cohen_d(a_arr, b_arr)

    if p_val < 0.001:
        stars = "***"
    elif p_val < 0.01:
        stars = "**"
    elif p_val < 0.05:
        stars = "*"
    else:
        stars = ""

    return {
        "t_stat":  float(t_stat),
        "p_val":   float(p_val),
        "cohen_d": float(d),
        "stars":   stars,
        "n_a":     len(a_arr),
        "n_b":     len(b_arr),
    }


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size: (mean_b - mean_a) / pooled_std."""
    pooled_std = np.sqrt(
        ((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2)
        / (len(a) + len(b) - 2 + 1e-12)
    )
    return float((b.mean() - a.mean()) / (pooled_std + 1e-12))


def compute_ma(rewards: List[float], window: int = 100) -> np.ndarray:
    """Moving average with `window` episodes."""
    arr = np.array(rewards, dtype=float)
    if len(arr) < window:
        return np.full(len(arr), np.nan)
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def episodes_to_solve(
    rewards: List[float],
    threshold: float = 200.0,
    window: int = 100,
) -> float:
    """First episode at which MA-100 ≥ threshold; returns inf if never."""
    ma = compute_ma(rewards, window)
    idx = np.where(ma >= threshold)[0]
    if len(idx) == 0:
        return INF
    return float(idx[0] + window)   # offset: MA starts after `window` episodes


def post_convergence_stats(
    rewards: List[float],
    threshold: float = 200.0,
    window: int = 100,
) -> Dict:
    """
    Statistics on the MA-100 curve *after* the first solve crossing.
    Returns mean, std, min.  If never solved, returns None for all.
    """
    ma = compute_ma(rewards, window)
    idx = np.where(ma >= threshold)[0]
    if len(idx) == 0:
        return {"post_conv_mean": None, "post_conv_std": None, "post_conv_min": None}
    ma_post = ma[idx[0]:]
    return {
        "post_conv_mean": float(ma_post.mean()),
        "post_conv_std":  float(ma_post.std(ddof=1)) if len(ma_post) > 1 else 0.0,
        "post_conv_min":  float(ma_post.min()),
    }
