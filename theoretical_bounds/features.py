"""
theoretical_bounds/features.py
--------------------------------
OHLCV validation, midpoint/range transform, and residual feature engineering.

Regime / instrument conditioning
---------------------------------
Two modes for multi-asset use:

  one_hot  : adds a binary column per symbol  (small, fixed symbol universe)
  embedding: adds a single integer symbol_id   (looked up from a registry)
             LightGBM treats it as a categorical feature natively when you
             set `categorical_feature=["symbol_id"]` in the Booster params.

Pass `symbol` (string) to build_residual_features() to activate either mode.
"""

from __future__ import annotations

from typing import Optional, Dict, List
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Symbol registry — stable, serialisable, instance-scoped
# ---------------------------------------------------------------------------

class SymbolRegistry:
    """
    Manages stable symbol → integer id mappings across training and inference.

    Key guarantee: once a symbol is assigned an id, that id never changes,
    even if the registry is saved and reloaded between runs.

    Usage
    -----
    registry = SymbolRegistry(["NQ", "ES", "CL"])  # pre-populate for training
    registry.get_id("NQ")   # → 0
    registry.save("registry.json")

    # Later, at inference:
    registry = SymbolRegistry.load("registry.json")
    registry.get_id("NQ")   # → 0  (stable)
    """

    def __init__(self, symbols: Optional[List[str]] = None):
        self._map: Dict[str, int] = {}
        if symbols:
            for s in symbols:
                self.get_id(s)  # pre-populate in order

    def get_id(self, symbol: str) -> int:
        """Return stable integer id for symbol, registering it if new."""
        if symbol not in self._map:
            self._map[symbol] = len(self._map)
        return self._map[symbol]

    def known_symbols(self) -> List[str]:
        """Return symbols in insertion order."""
        return sorted(self._map, key=self._map.get)  # type: ignore[arg-type]

    def to_dict(self) -> Dict[str, int]:
        return dict(self._map)

    def save(self, path: str) -> None:
        import json
        with open(path, "w") as f:
            json.dump(self._map, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SymbolRegistry":
        import json
        inst = cls()
        with open(path) as f:
            inst._map = {k: int(v) for k, v in json.load(f).items()}
        return inst

    def __repr__(self) -> str:
        return f"SymbolRegistry({self._map})"


# Module-level default registry (convenience for single-process use)
# For multi-process or persistent deployments, instantiate your own.
_DEFAULT_REGISTRY = SymbolRegistry()


def symbol_registry() -> SymbolRegistry:
    """Return the module-level default SymbolRegistry instance."""
    return _DEFAULT_REGISTRY


# ---------------------------------------------------------------------------
# OHLCV validation
# ---------------------------------------------------------------------------

def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean OHLCV data.
    Returns a sorted, cleaned copy with no NaN in critical columns.
    """
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy().sort_index()

    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["High", "Low", "Close"])
    out = out[(out["High"] > 0) & (out["Low"] > 0) & (out["Close"] > 0)]
    out = out[out["High"] >= out["Low"]]

    return out


# ---------------------------------------------------------------------------
# Core transforms
# ---------------------------------------------------------------------------

def add_mid_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add log midpoint and log range series.

    m_t = 0.5 * (log H_t + log L_t)
    r_t = log H_t - log L_t
    """
    out = df.copy()

    log_h = np.log(out["High"])
    log_l = np.log(out["Low"])

    out["mid_log"]    = 0.5 * (log_h + log_l)
    out["range_log"]  = log_h - log_l

    out["mid_actual"]   = 0.5 * (out["High"] + out["Low"])
    out["range_actual"] = out["High"] - out["Low"]

    return out


# ---------------------------------------------------------------------------
# Residual feature builder (feeds LightGBM corrector)
# ---------------------------------------------------------------------------

def build_residual_features(
    bounds_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    atr_period: int = 14,
    lag_periods: int = 5,
    symbol: Optional[str] = None,
    symbol_mode: str = "embedding",   # "embedding" | "one_hot" | "none"
    registry: Optional[SymbolRegistry] = None,
) -> pd.DataFrame:
    """
    Build feature matrix for the LightGBM residual corrector.

    Parameters
    ----------
    bounds_df   : output of rolling_theoretical_bounds() — must contain
                  e_mid, e_range, diseq_score, theoretical columns
    ohlcv_df    : original validated OHLCV frame (used for ATR, gap, open loc)
    atr_period  : lookback for ATR calculation
    lag_periods : number of lag periods for residual features
    symbol      : instrument identifier string e.g. "NQ", "ES", "CL"
                  If None, no symbol features are added.
    symbol_mode : how to encode the symbol
                  "embedding" → single integer `symbol_id` column
                                (set categorical_feature=["symbol_id"] in LGB)
                  "one_hot"   → binary column `sym_<symbol>` for this symbol
                  "none"      → no encoding
    registry    : SymbolRegistry instance to use for stable id lookup.
                  If None, uses the module-level default registry.
                  Pass the same registry instance at train and inference time
                  to guarantee id stability.

    Returns
    -------
    DataFrame with features aligned on the same index as bounds_df,
    plus normalised target columns: e_mid_norm, e_range_norm.
    """
    ohlcv = validate_ohlcv(ohlcv_df)

    # -----------------------------------------------------------------------
    # ATR (Wilder's)
    # -----------------------------------------------------------------------
    high  = ohlcv["High"]
    low   = ohlcv["Low"]
    close = ohlcv["Close"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(span=atr_period, adjust=False).mean()
    atr.name = "atr"

    # -----------------------------------------------------------------------
    # Realized vol — multiple horizons
    # -----------------------------------------------------------------------
    log_ret  = np.log(close / close.shift(1))
    real_vol5  = log_ret.rolling(5).std();   real_vol5.name  = "realized_vol_5"
    real_vol20 = log_ret.rolling(20).std();  real_vol20.name = "realized_vol_20"
    real_vol60 = log_ret.rolling(60).std();  real_vol60.name = "realized_vol_60"

    # Vol ratio: short/long — high = vol expanding, low = vol compressing
    vol_ratio = (real_vol5 / real_vol60.replace(0, np.nan)).rename("vol_ratio_5_60")

    # -----------------------------------------------------------------------
    # Vol regime — ordinal quintile label (0–4) over rolling 252 days
    # -----------------------------------------------------------------------
    vol_regime = atr.rolling(252, min_periods=60).apply(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")[-1]
        if len(x) >= 60 else np.nan
    )
    vol_regime.name = "vol_regime"

    # Vol regime percentile — continuous [0,1] rank within rolling window
    vol_regime_pctile = atr.rolling(252, min_periods=60).rank(pct=True)
    vol_regime_pctile.name = "vol_regime_pctile"

    # -----------------------------------------------------------------------
    # Trend regime — three-state: strong up / flat / strong down
    #   +1 : close > SMA20 AND SMA20 slope positive
    #    0 : neither
    #   -1 : close < SMA20 AND SMA20 slope negative
    # -----------------------------------------------------------------------
    sma20       = close.rolling(20).mean()
    sma20_slope = sma20 - sma20.shift(5)                         # 5-day momentum of SMA
    above_sma   = (close > sma20).astype(int)
    sma_rising  = (sma20_slope > 0).astype(int)

    trend_regime = (above_sma * sma_rising - (1 - above_sma) * (1 - sma_rising)).astype(float)
    trend_regime.name = "trend_regime"                           # -1 / 0 / +1

    # Legacy binary flag kept for backward compat
    trend_flag = above_sma.astype(float)
    trend_flag.name = "trend_flag"

    # -----------------------------------------------------------------------
    # Range regime — expanding vs contracting (ATR vs 5-day avg ATR)
    # -----------------------------------------------------------------------
    balance_flag = (atr > atr.rolling(5).mean()).astype(float)
    balance_flag.name = "balance_flag"

    # -----------------------------------------------------------------------
    # Overnight gap % relative to prior close
    # -----------------------------------------------------------------------
    gap_pct = (ohlcv["Open"] - prev_close) / prev_close
    gap_pct.name = "gap_pct"

    # Gap direction persistence: was prior gap same sign?
    gap_sign_persist = (gap_pct * gap_pct.shift(1) > 0).astype(float)
    gap_sign_persist.name = "gap_sign_persist"

    # -----------------------------------------------------------------------
    # Open location within prior day range  (0 = at prior low, 1 = at prior high)
    # -----------------------------------------------------------------------
    prior_high  = high.shift(1)
    prior_low   = low.shift(1)
    prior_range = (prior_high - prior_low).replace(0, np.nan)
    open_loc    = (ohlcv["Open"] - prior_low) / prior_range
    open_loc    = open_loc.clip(0, 1)
    open_loc.name = "open_loc_in_prior_range"

    # -----------------------------------------------------------------------
    # Combine base features
    # -----------------------------------------------------------------------
    base_features = pd.concat(
        [
            atr,
            real_vol5, real_vol20, real_vol60, vol_ratio,
            vol_regime, vol_regime_pctile,
            trend_regime, trend_flag,
            balance_flag,
            gap_pct, gap_sign_persist,
            open_loc,
        ],
        axis=1,
    )

    # -----------------------------------------------------------------------
    # Align with bounds_df
    # -----------------------------------------------------------------------
    feat = bounds_df.join(base_features, how="left")

    # -----------------------------------------------------------------------
    # Lag the VECM residuals (e_mid, e_range) and diseq_score
    # -----------------------------------------------------------------------
    for lag in range(1, lag_periods + 1):
        feat[f"e_mid_lag{lag}"]    = feat["e_mid"].shift(lag)
        feat[f"e_range_lag{lag}"]  = feat["e_range"].shift(lag)
        feat[f"diseq_lag{lag}"]    = feat["diseq_score"].shift(lag)

    # -----------------------------------------------------------------------
    # Normalise targets by ATR for regime stability
    # -----------------------------------------------------------------------
    atr_safe = feat["atr"].replace(0, np.nan)
    feat["e_mid_norm"]   = feat["e_mid"]   / atr_safe
    feat["e_range_norm"] = feat["e_range"] / atr_safe

    for lag in range(1, lag_periods + 1):
        feat[f"e_mid_lag{lag}_norm"]   = feat[f"e_mid_lag{lag}"]   / atr_safe
        feat[f"e_range_lag{lag}_norm"] = feat[f"e_range_lag{lag}"] / atr_safe

    # -----------------------------------------------------------------------
    # Symbol / instrument conditioning
    # -----------------------------------------------------------------------
    if symbol is not None and symbol_mode != "none":
        reg = registry if registry is not None else _DEFAULT_REGISTRY
        if symbol_mode == "embedding":
            feat["symbol_id"] = reg.get_id(symbol)
        elif symbol_mode == "one_hot":
            feat[f"sym_{symbol}"] = 1.0
        else:
            raise ValueError(f"symbol_mode must be 'embedding', 'one_hot', or 'none'. Got: {symbol_mode!r}")

    return feat


def get_feature_columns(
    lag_periods: int = 5,
    symbol: Optional[str] = None,
    symbol_mode: str = "embedding",
    known_symbols: Optional[List[str]] = None,
) -> List[str]:
    """
    Return the ordered list of feature column names used by the LightGBM corrector.
    Must match build_residual_features() exactly.

    Parameters
    ----------
    lag_periods    : must match the value used in build_residual_features
    symbol         : the current instrument symbol (for one_hot mode)
    symbol_mode    : "embedding" | "one_hot" | "none"
    known_symbols  : for one_hot mode, list of ALL symbols in training set
                     so the full one-hot column set is returned consistently
    """
    base = [
        "atr",
        "realized_vol_5",
        "realized_vol_20",
        "realized_vol_60",
        "vol_ratio_5_60",
        "vol_regime",
        "vol_regime_pctile",
        "trend_regime",
        "trend_flag",
        "balance_flag",
        "gap_pct",
        "gap_sign_persist",
        "open_loc_in_prior_range",
        "diseq_score",
        "theoretical_range",
        "theoretical_mid",
    ]
    lag_feats: List[str] = []
    for lag in range(1, lag_periods + 1):
        lag_feats += [
            f"e_mid_lag{lag}_norm",
            f"e_range_lag{lag}_norm",
            f"diseq_lag{lag}",
        ]

    symbol_feats: List[str] = []
    if symbol_mode == "embedding":
        symbol_feats = ["symbol_id"]
    elif symbol_mode == "one_hot":
        syms = known_symbols if known_symbols else ([symbol] if symbol else [])
        symbol_feats = [f"sym_{s}" for s in syms]

    return base + lag_feats + symbol_feats
