"""
theoretical_bounds/reconstruct.py
------------------------------------
Back-transform (mid_log, range_log) forecasts into theoretical HOD/LOD.
Enforces hard constraints and optionally adds confidence bands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ReconstructConfig:
    # Range clip: fraction of close price
    clip_range_min_pct: float = 0.001   # 0.1% minimum range
    clip_range_max_pct: float = 0.15    # 15% maximum range

    # Optional confidence bands (multiples of range)
    add_bands: bool = True
    band_width: float = 0.25            # +/- 25% of theoretical range


# ---------------------------------------------------------------------------
# Single-row reconstruction
# ---------------------------------------------------------------------------

def reconstruct_bounds(
    mid_hat_log: float,
    range_hat_log: float,
    close_ref: float,
    cfg: ReconstructConfig,
) -> Dict[str, float]:
    """
    Convert log-space forecast to price-space theoretical HOD/LOD.

    H = exp(m + r/2)
    L = exp(m - r/2)
    """
    # Clip range in log space
    min_r_log = np.log(1.0 + cfg.clip_range_min_pct)
    max_r_log = np.log(1.0 + cfg.clip_range_max_pct)
    range_hat_log = float(np.clip(range_hat_log, min_r_log, max_r_log))

    high_hat = float(np.exp(mid_hat_log + 0.5 * range_hat_log))
    low_hat  = float(np.exp(mid_hat_log - 0.5 * range_hat_log))

    # Safety swap
    if low_hat > high_hat:
        low_hat, high_hat = high_hat, low_hat

    mid_hat   = 0.5 * (high_hat + low_hat)
    range_hat = high_hat - low_hat

    result = {
        "theoretical_high":  high_hat,
        "theoretical_low":   low_hat,
        "theoretical_mid":   mid_hat,
        "theoretical_range": range_hat,
        "range_hat_log_clipped": range_hat_log,
        "close_ref":         close_ref,
    }

    if cfg.add_bands:
        half_band = cfg.band_width * range_hat
        result["theoretical_high_upper"] = high_hat + half_band
        result["theoretical_high_lower"] = high_hat - half_band
        result["theoretical_low_upper"]  = low_hat  + half_band
        result["theoretical_low_lower"]  = low_hat  - half_band

    return result


# ---------------------------------------------------------------------------
# Apply correction from LightGBM residual model
# ---------------------------------------------------------------------------

def apply_correction(
    theoretical_mid: float,
    theoretical_range: float,
    e_mid_hat: float,
    e_range_hat: float,
    atr: float,
    cfg: ReconstructConfig,
) -> Dict[str, float]:
    """
    Apply normalised ML residual corrections.

    Corrector predicts (e_mid_norm, e_range_norm) = residuals / ATR.
    Denormalise and add to theoretical bounds.
    """
    e_mid_final   = e_mid_hat   * atr
    e_range_final = e_range_hat * atr

    corrected_mid   = theoretical_mid   + e_mid_final
    corrected_range = theoretical_range + e_range_final

    # Enforce range positivity
    min_range = atr * 0.05
    corrected_range = max(corrected_range, min_range)

    corrected_high = corrected_mid + 0.5 * corrected_range
    corrected_low  = corrected_mid - 0.5 * corrected_range

    return {
        "corrected_high":   corrected_high,
        "corrected_low":    corrected_low,
        "corrected_mid":    corrected_mid,
        "corrected_range":  corrected_range,
        "e_mid_applied":    e_mid_final,
        "e_range_applied":  e_range_final,
    }


# ---------------------------------------------------------------------------
# Vectorised reconstruction over a DataFrame
# ---------------------------------------------------------------------------

def reconstruct_dataframe(
    forecast_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    cfg: Optional[ReconstructConfig] = None,
) -> pd.DataFrame:
    """
    Apply reconstruct_bounds() row-wise to a forecast DataFrame.

    forecast_df : must have mid_hat_log, range_hat_log columns
    ohlcv_df    : original OHLCV, used for close_ref alignment
    """
    cfg = cfg or ReconstructConfig()

    close_aligned = ohlcv_df["Close"].reindex(forecast_df.index)

    rows = []
    for ts, row in forecast_df.iterrows():
        if "vecm_error" in row and pd.notna(row.get("vecm_error")):
            rows.append({"timestamp": ts, "reconstruct_error": row["vecm_error"]})
            continue

        try:
            close_ref = float(close_aligned.loc[ts])
        except Exception:
            close_ref = np.nan

        try:
            recon = reconstruct_bounds(
                mid_hat_log=float(row["mid_hat_log"]),
                range_hat_log=float(row["range_hat_log"]),
                close_ref=close_ref,
                cfg=cfg,
            )
            recon["timestamp"] = ts
            rows.append(recon)
        except Exception as e:
            rows.append({"timestamp": ts, "reconstruct_error": str(e)})

    result = pd.DataFrame(rows).set_index("timestamp")
    return result
