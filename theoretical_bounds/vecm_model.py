"""
theoretical_bounds/vecm_model.py
----------------------------------
Rolling VECM backbone:
  - lag / cointegration rank selection
  - 1-step-ahead forecast of (mid_log, range_log)
  - disequilibrium score
  - optional cointegration diagnostics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.vecm import VECM, select_order, select_coint_rank

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class VECMConfig:
    train_window: int     = 250      # rolling window in days
    min_train_window: int = 180      # minimum to attempt a fit
    max_lags: int         = 5        # max lag order considered by AIC
    deterministic: str    = "co"     # "co" = constant in coint relation
    coint_rank: Optional[int] = 1    # None = auto-select via trace test
    refit_every: int      = 1        # refit every N rows (1 = daily)


# ---------------------------------------------------------------------------
# Lag / rank selection
# ---------------------------------------------------------------------------

def _select_lag_and_rank(
    endog: pd.DataFrame,
    cfg: VECMConfig,
) -> Tuple[int, int]:
    """
    Selects VAR lag order (AIC) and cointegration rank (Johansen trace).
    Falls back to (1, 1) on any error.
    """
    selected_lags = 1
    try:
        order_res = select_order(
            endog,
            maxlags=cfg.max_lags,
            deterministic=cfg.deterministic,
        )
        if order_res.aic is not None:
            selected_lags = max(1, int(order_res.aic))
        else:
            logger.info("AIC lag selection returned None; falling back to lags=1")
    except Exception as exc:
        logger.info("AIC lag selection failed (%s); falling back to lags=1", exc)
        selected_lags = 1

    selected_rank = cfg.coint_rank if cfg.coint_rank is not None else 1
    if cfg.coint_rank is None:
        try:
            rank_res = select_coint_rank(
                endog,
                det_order=0,
                k_ar_diff=max(1, selected_lags - 1),
                method="trace",
                signif=0.05,
            )
            selected_rank = max(1, int(rank_res.rank))
        except Exception as exc:
            logger.info("Cointegration rank selection failed (%s); falling back to rank=1", exc)
            selected_rank = 1

    return selected_lags, selected_rank


# ---------------------------------------------------------------------------
# Single fit + forecast
# ---------------------------------------------------------------------------

def fit_and_forecast(
    endog: pd.DataFrame,
    cfg: VECMConfig,
) -> Dict[str, Any]:
    """
    Fit VECM on endog (columns: mid_log, range_log) and return 1-step forecast.

    Returns
    -------
    dict with:
        mid_hat_log      : float
        range_hat_log    : float
        diseq_score      : float  (L2 norm of error-correction term)
        selected_lags    : int
        selected_rank    : int
        alpha            : np.ndarray or None
        beta             : np.ndarray or None
    """
    lags, rank = _select_lag_and_rank(endog, cfg)

    model = VECM(
        endog,
        k_ar_diff=max(1, lags - 1),
        coint_rank=rank,
        deterministic=cfg.deterministic,
    )
    res = model.fit()

    forecast_arr = res.predict(steps=1)          # shape (1, 2)
    mid_hat_log   = float(forecast_arr[0, 0])
    range_hat_log = float(forecast_arr[0, 1])

    # Disequilibrium score: ||beta' * y_t||
    last_y = endog.iloc[-1].values.reshape(-1, 1)
    try:
        ect = res.beta.T @ last_y
        diseq_score = float(np.linalg.norm(ect))
    except Exception:
        diseq_score = np.nan

    return {
        "mid_hat_log":   mid_hat_log,
        "range_hat_log": range_hat_log,
        "diseq_score":   diseq_score,
        "selected_lags": lags,
        "selected_rank": rank,
        "alpha":         res.alpha if hasattr(res, "alpha") else None,
        "beta":          res.beta if hasattr(res, "beta") else None,
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def run_diagnostics(series: pd.Series, label: str = "") -> Dict[str, Any]:
    """
    ADF + KPSS on a univariate series.
    Used to confirm (near-)stationarity of range_log and I(1) of mid_log.
    """
    clean = series.dropna()
    results: Dict[str, Any] = {"label": label, "n": len(clean)}

    # ADF
    try:
        adf_stat, adf_pval, _, _, adf_crit, _ = adfuller(clean, autolag="AIC")
        results["adf_stat"]    = adf_stat
        results["adf_pval"]    = adf_pval
        results["adf_crit_5"]  = adf_crit["5%"]
        results["adf_reject_unit_root_5pct"] = adf_pval < 0.05
    except Exception as e:
        results["adf_error"] = str(e)

    # KPSS
    try:
        kpss_stat, kpss_pval, _, kpss_crit = kpss(clean, regression="c", nlags="auto")
        results["kpss_stat"]   = kpss_stat
        results["kpss_pval"]   = kpss_pval
        results["kpss_crit_5"] = kpss_crit["5%"]
        results["kpss_reject_stationarity_5pct"] = kpss_stat > kpss_crit["5%"]
    except Exception as e:
        results["kpss_error"] = str(e)

    return results


def run_all_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Run diagnostics on mid_log, range_log, and their first differences.
    Returns a summary DataFrame.
    """
    targets = {
        "mid_log":        data["mid_log"],
        "range_log":      data["range_log"],
        "d_mid_log":      data["mid_log"].diff().dropna(),
        "d_range_log":    data["range_log"].diff().dropna(),
    }

    rows = [run_diagnostics(s, label=lbl) for lbl, s in targets.items()]
    return pd.DataFrame(rows).set_index("label")


# ---------------------------------------------------------------------------
# Rolling fit loop (no reconstruction — see pipeline.py)
# ---------------------------------------------------------------------------

def rolling_vecm_forecast(
    data: pd.DataFrame,
    cfg: Optional[VECMConfig] = None,
) -> pd.DataFrame:
    """
    Rolling 1-step-ahead VECM forecast over the full data.

    Parameters
    ----------
    data : DataFrame with columns mid_log, range_log (from features.add_mid_range)
    cfg  : VECMConfig (defaults if None)

    Returns
    -------
    DataFrame indexed like data, with forecast columns and diagnostics.
    Rows before min_train_window are skipped.
    """
    cfg = cfg or VECMConfig()
    endog_cols = ["mid_log", "range_log"]

    records: List[Dict[str, Any]] = []
    last_fit_idx = -cfg.refit_every  # force fit on first eligible row

    cached_result: Optional[Dict[str, Any]] = None

    for i in range(len(data)):
        if i < cfg.min_train_window:
            continue

        start = max(0, i - cfg.train_window)
        train = data.iloc[start:i][endog_cols].dropna()

        if len(train) < cfg.min_train_window:
            continue

        try:
            if (i - last_fit_idx) >= cfg.refit_every or cached_result is None:
                cached_result = fit_and_forecast(train, cfg)
                last_fit_idx = i

            rec = {
                "timestamp":      data.index[i],
                "mid_hat_log":    cached_result["mid_hat_log"],
                "range_hat_log":  cached_result["range_hat_log"],
                "diseq_score":    cached_result["diseq_score"],
                "selected_lags":  cached_result["selected_lags"],
                "selected_rank":  cached_result["selected_rank"],
            }
        except Exception as e:
            rec = {
                "timestamp":  data.index[i],
                "vecm_error": str(e),
            }

        records.append(rec)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).set_index("timestamp")
