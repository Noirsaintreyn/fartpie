"""Stress-test helpers (Bailey-Lopez de Prado PSR/DSR, stationary block bootstrap, sign-flip test, drawdown).  Used by reports 337-339."""
import numpy as np, pandas as pd
from scipy.stats import norm, skew, kurtosis
EG = 0.5772156649
def sharpe(x, ann=252): x = np.asarray(x, float); return x.mean() / x.std(ddof=1) * np.sqrt(ann) if x.std(ddof=1) > 0 else np.nan
def psr(x, sr0=0.0):
    """Probabilistic Sharpe: P(true SR > sr0) using per-period SR, skew, kurtosis (sr0 per-period)."""
    x = np.asarray(x, float); T = len(x); s = x.mean() / x.std(ddof=1); g3 = skew(x); g4 = kurtosis(x, fisher=False)
    return norm.cdf((s - sr0) * np.sqrt(T - 1) / np.sqrt(1 - g3 * s + (g4 - 1) / 4 * s * s))
def dsr(x, n_trials, sr_sd_ann=0.5, ann=252):
    """Deflated Sharpe: PSR against the expected maximum SR of n_trials skill-less trials whose annualised SR has sd sr_sd_ann (assumption, stated in reports)."""
    sd = sr_sd_ann / np.sqrt(ann); sr0 = sd * ((1 - EG) * norm.ppf(1 - 1 / n_trials) + EG * norm.ppf(1 - 1 / (n_trials * np.e))) if n_trials > 1 else 0.0
    return psr(x, sr0), sr0 * np.sqrt(ann)
def stationary_bootstrap(x, n=5000, mean_block=20, seed=1):
    """indices of n resamples (rows) with Politis-Romano stationary blocks"""
    rng = np.random.default_rng(seed); T = len(x); p = 1 / mean_block; out = np.empty((n, T), dtype=np.int32)
    for i in range(n):
        idx = np.empty(T, dtype=np.int32); idx[0] = rng.integers(T)
        jumps = rng.random(T) < p; starts = rng.integers(0, T, T)
        for t in range(1, T): idx[t] = starts[t] if jumps[t] else (idx[t - 1] + 1) % T
        out[i] = idx
    return out
def maxdd_cum(c):  # c = cumulative equity (additive)
    return float((c - np.maximum.accumulate(c)).min())
def signflip_p(x, n=10000, seed=2):
    """two-sided-free one-sided p: P(mean of randomly sign-flipped series >= observed mean); null = no directional edge, same magnitudes"""
    rng = np.random.default_rng(seed); x = np.asarray(x, float); m = x.mean(); s = rng.choice([-1, 1], size=(n, len(x))); return float(((s * x).mean(axis=1) >= m).mean())
