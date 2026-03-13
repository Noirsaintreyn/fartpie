"""
theoretical_bounds/live_predictor.py
--------------------------------------
Pre-open daily HOD/LOD predictor.

Produces today's corrected_high / corrected_low using only information
that is available before today's session opens:
  - Full OHLCV history up to and including yesterday
  - Today's open price (or a pre-open estimate)

Zero look-ahead: no same-day actuals are used at inference time.

Usage
-----
    from theoretical_bounds.live_predictor import LivePredictor

    # --- Once, after training: ---
    predictor = LivePredictor.from_trained_pipeline(pipe, model_dir="./models/tb")

    # --- Each morning, pre-open: ---
    forecast = predictor.predict_today(
        history_ohlcv=ohlcv_up_to_yesterday,
        today_open=19_450.0,        # from pre-market or prior close
        today_date=pd.Timestamp("2025-03-12"),
    )
    print(forecast)
    # {
    #   "date":             Timestamp("2025-03-12"),
    #   "corrected_high":   19_823.0,
    #   "corrected_low":    19_201.0,
    #   "corrected_mid":    19_512.0,
    #   "corrected_range":    622.0,
    #   "theoretical_high": 19_791.0,
    #   "theoretical_low":  19_215.0,
    #   "diseq_score":          0.031,
    #   "atr":                142.3,
    #   "vol_regime_pctile":    0.72,
    #   "trend_regime":         1.0,
    # }
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .features import (
    validate_ohlcv,
    add_mid_range,
    build_residual_features,
    SymbolRegistry,
)
from .vecm_model import VECMConfig, fit_and_forecast
from .reconstruct import ReconstructConfig, reconstruct_bounds, apply_correction
from .corrector import ResidualCorrector
from .pipeline import PipelineConfig, TheoreticalBoundsPipeline


# ---------------------------------------------------------------------------
# LivePredictor
# ---------------------------------------------------------------------------

@dataclass
class LiveForecast:
    """
    Typed container for a single day's pre-open HOD/LOD forecast.

    All prices are in native instrument units (points for NQ/ES, etc.).
    Breach thresholds are provided as convenience fields for stop placement.
    """
    date:              pd.Timestamp
    corrected_high:    float
    corrected_low:     float
    corrected_mid:     float
    corrected_range:   float
    theoretical_high:  float
    theoretical_low:   float
    theoretical_mid:   float
    theoretical_range: float
    diseq_score:       float
    atr:               float
    vol_regime_pctile: float
    trend_regime:      float
    # Convenience: breach levels at 0.5 / 1.0 / 2.0 ATR beyond envelope
    breach_high_05atr: float = 0.0
    breach_high_1atr:  float = 0.0
    breach_high_2atr:  float = 0.0
    breach_low_05atr:  float = 0.0
    breach_low_1atr:   float = 0.0
    breach_low_2atr:   float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    def __str__(self) -> str:
        lines = [
            f"LiveForecast  {self.date.date()}",
            f"  Corrected  H: {self.corrected_high:>10.2f}   L: {self.corrected_low:>10.2f}   "
            f"Range: {self.corrected_range:>8.2f}",
            f"  Structural H: {self.theoretical_high:>10.2f}   L: {self.theoretical_low:>10.2f}   "
            f"Mid: {self.theoretical_mid:>10.2f}",
            f"  ATR: {self.atr:.2f}   VolPctile: {self.vol_regime_pctile:.2f}   "
            f"TrendRegime: {self.trend_regime:+.0f}   Diseq: {self.diseq_score:.4f}",
            f"  Breach levels (above high / below low):",
            f"    0.5 ATR: {self.breach_high_05atr:.2f} / {self.breach_low_05atr:.2f}",
            f"    1.0 ATR: {self.breach_high_1atr:.2f}  / {self.breach_low_1atr:.2f}",
            f"    2.0 ATR: {self.breach_high_2atr:.2f}  / {self.breach_low_2atr:.2f}",
        ]
        return "\n".join(lines)


class LivePredictor:
    """
    Thin inference wrapper around a trained TheoreticalBoundsPipeline.

    Responsibilities
    ----------------
    1. Build one synthetic OHLCV row for today (using today's open,
       yesterday's close, and a placeholder H/L = open for feature alignment).
    2. Run one VECM step on the tail window of history.
    3. Build the ex-ante feature row (no same-day H/L used).
    4. Call corrector.predict() for the residual correction.
    5. Return a fully typed LiveForecast.

    No re-training occurs during prediction.
    """

    def __init__(
        self,
        cfg:       PipelineConfig,
        corrector: ResidualCorrector,
    ):
        self.cfg       = cfg
        self.corrector = corrector

    # ------------------------------------------------------------------
    # Factory: build from an already-trained pipeline
    # ------------------------------------------------------------------

    @classmethod
    def from_trained_pipeline(
        cls,
        pipeline:  TheoreticalBoundsPipeline,
        model_dir: Optional[str] = None,
    ) -> "LivePredictor":
        """
        Create a LivePredictor from a trained TheoreticalBoundsPipeline.

        Optionally saves the corrector and registry to model_dir so the
        predictor can be reconstructed without the full pipeline later.
        """
        if model_dir is not None:
            pipeline.save_corrector(model_dir)

        return cls(
            cfg=pipeline.cfg,
            corrector=pipeline.corrector,
        )

    @classmethod
    def from_saved_models(
        cls,
        model_dir: str,
        cfg:       PipelineConfig,
    ) -> "LivePredictor":
        """
        Reconstruct a LivePredictor from persisted models (no retraining).

        cfg should match the config used during training — especially
        symbol, symbol_mode, atr_period, and lag_periods.
        """
        pipe = TheoreticalBoundsPipeline(cfg)
        pipe.load_corrector(model_dir)
        return cls(cfg=cfg, corrector=pipe.corrector)

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def predict_today(
        self,
        history_ohlcv: pd.DataFrame,
        today_open:    float,
        today_date:    Optional[pd.Timestamp] = None,
    ) -> LiveForecast:
        """
        Produce pre-open HOD/LOD forecast for today.

        Parameters
        ----------
        history_ohlcv : OHLCV DataFrame containing data up to and including
                        yesterday. Must NOT include any same-day rows.
        today_open    : today's open price or pre-market estimate
        today_date    : timestamp for today's row (default: last history date + 1 bday)

        Returns
        -------
        LiveForecast with corrected and theoretical levels plus breach thresholds
        """
        hist = validate_ohlcv(history_ohlcv).sort_index()

        if today_date is None:
            today_date = hist.index[-1] + pd.tseries.offsets.BDay(1)

        if today_date <= hist.index[-1]:
            raise ValueError(
                f"today_date ({today_date.date()}) must be after the last "
                f"history date ({hist.index[-1].date()}). "
                "Pass history_ohlcv up to yesterday only."
            )

        # ------------------------------------------------------------------
        # Step 1: Extend history with a synthetic today row
        #
        # H = L = open for the placeholder.  These values are ONLY used to
        # compute today's ATR/vol/gap features — they do NOT leak because:
        #   - ATR uses prior bars (ewm with shift built into rolling_vecm)
        #   - VECM trains only on hist (no today row in endog)
        #   - Residual features only use lagged residuals + prior-day values
        # The synthetic row is stripped from VECM training below.
        # ------------------------------------------------------------------
        today_row = pd.DataFrame(
            {
                "Open":   today_open,
                "High":   today_open,   # placeholder — not used in forecast
                "Low":    today_open,   # placeholder — not used in forecast
                "Close":  today_open,   # will be overwritten by corrected mid
                "Volume": 0.0,
            },
            index=[today_date],
        )
        extended = pd.concat([hist, today_row])

        # ------------------------------------------------------------------
        # Step 2: VECM forecast — train on history only, predict 1 step ahead
        # ------------------------------------------------------------------
        data_with_mid = add_mid_range(extended)

        train_window = self.cfg.vecm.train_window
        min_window   = self.cfg.vecm.min_train_window

        # Use only yesterday's window (exclude today's synthetic row)
        hist_mid = data_with_mid.iloc[:-1][["mid_log", "range_log"]].dropna()
        train_endog = hist_mid.iloc[-train_window:]

        if len(train_endog) < min_window:
            raise ValueError(
                f"Insufficient history for VECM: need {min_window} bars, "
                f"have {len(train_endog)}."
            )

        vecm_result = fit_and_forecast(train_endog, self.cfg.vecm)

        recon = reconstruct_bounds(
            mid_hat_log=vecm_result["mid_hat_log"],
            range_hat_log=vecm_result["range_hat_log"],
            close_ref=float(hist["Close"].iloc[-1]),
            cfg=self.cfg.reconstruct,
        )

        # ------------------------------------------------------------------
        # Step 3: Ex-ante feature row
        #
        # We build the full feature frame on 'extended' (which includes the
        # synthetic today row with open=high=low), but only take the LAST row.
        # All features are computed from prior-bar data:
        #   - ATR: ewm of prior TRs (today's TR = |open - prior_close| only)
        #   - vol: log_ret rolling std (today's ret = open/prior_close)
        #   - gap_pct: today's open vs yesterday's close ← this IS a real signal
        #   - open_loc: today's open location in yesterday's range ← real signal
        #   - trend_flag, balance_flag: based on prior closes and ATR
        #
        # The VECM residual lags (e_mid_lagN, e_range_lagN) come from
        # yesterday's and earlier VECM misses — no leakage.
        # ------------------------------------------------------------------

        # Build a fake bounds_df for the feature builder:
        # We only need the structural columns for the last row.
        # Use a single-row frame with today's theoretical values.
        bounds_today = pd.DataFrame(
            {
                "theoretical_high":  recon["theoretical_high"],
                "theoretical_low":   recon["theoretical_low"],
                "theoretical_mid":   recon["theoretical_mid"],
                "theoretical_range": recon["theoretical_range"],
                "diseq_score":       vecm_result["diseq_score"],
                # Residuals are unknown for today (no actuals yet) — set 0
                # so they don't contaminate lags in the single-row frame.
                # The lagged residual features pick up from history.
                "e_mid":   0.0,
                "e_range": 0.0,
            },
            index=[today_date],
        )

        # We need lagged residuals from history.
        # Re-run a lightweight version: theoretical bounds on hist only,
        # so we can get the lagged e_mid / e_range from recent days.
        hist_bounds = self._get_hist_bounds(hist)

        if hist_bounds is not None and len(hist_bounds) > 0:
            # Append today's structural row
            combined_bounds = pd.concat([hist_bounds, bounds_today])
        else:
            combined_bounds = bounds_today

        feat_frame = build_residual_features(
            bounds_df=combined_bounds,
            ohlcv_df=extended,
            atr_period=self.cfg.atr_period,
            lag_periods=self.cfg.lag_periods,
            symbol=self.cfg.symbol,
            symbol_mode=self.cfg.symbol_mode,
            registry=self.cfg.registry,
        )

        # Take only today's row
        today_feat = feat_frame.loc[[today_date]]

        # ------------------------------------------------------------------
        # Step 4: Corrector prediction
        # ------------------------------------------------------------------
        preds = self.corrector.predict(today_feat)

        e_mid_hat_norm   = float(preds["e_mid_hat_norm"].iloc[0])
        e_range_hat_norm = float(preds["e_range_hat_norm"].iloc[0])
        atr_val          = float(today_feat["atr"].iloc[0])

        corrected = apply_correction(
            theoretical_mid=recon["theoretical_mid"],
            theoretical_range=recon["theoretical_range"],
            e_mid_hat=e_mid_hat_norm,
            e_range_hat=e_range_hat_norm,
            atr=atr_val if not np.isnan(atr_val) else 1.0,
            cfg=self.cfg.reconstruct,
        )

        # ------------------------------------------------------------------
        # Step 5: Assemble forecast
        # ------------------------------------------------------------------
        ch = corrected["corrected_high"]
        cl = corrected["corrected_low"]

        return LiveForecast(
            date=today_date,
            corrected_high=ch,
            corrected_low=cl,
            corrected_mid=corrected["corrected_mid"],
            corrected_range=corrected["corrected_range"],
            theoretical_high=recon["theoretical_high"],
            theoretical_low=recon["theoretical_low"],
            theoretical_mid=recon["theoretical_mid"],
            theoretical_range=recon["theoretical_range"],
            diseq_score=vecm_result["diseq_score"],
            atr=atr_val,
            vol_regime_pctile=float(today_feat.get("vol_regime_pctile", pd.Series([np.nan])).iloc[0]),
            trend_regime=float(today_feat.get("trend_regime", pd.Series([0.0])).iloc[0]),
            # Breach convenience levels
            breach_high_05atr=ch + 0.5 * atr_val,
            breach_high_1atr= ch + 1.0 * atr_val,
            breach_high_2atr= ch + 2.0 * atr_val,
            breach_low_05atr= cl - 0.5 * atr_val,
            breach_low_1atr=  cl - 1.0 * atr_val,
            breach_low_2atr=  cl - 2.0 * atr_val,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_hist_bounds(self, hist: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Lightweight pass to get historical theoretical bounds for lag features.
        Uses only the tail needed to populate lag_periods days of residuals.
        Returns None if history is too short.
        """
        needed = self.cfg.vecm.min_train_window + self.cfg.lag_periods + 10
        if len(hist) < needed:
            return None

        tail = hist.iloc[-needed:]
        data = add_mid_range(tail)

        train_endog = data[["mid_log", "range_log"]].dropna()
        if len(train_endog) < self.cfg.vecm.min_train_window:
            return None

        rows = []
        for i in range(self.cfg.vecm.min_train_window, len(train_endog)):
            window = train_endog.iloc[max(0, i - self.cfg.vecm.train_window):i]
            try:
                res = fit_and_forecast(window, self.cfg.vecm)
                rec = reconstruct_bounds(
                    mid_hat_log=res["mid_hat_log"],
                    range_hat_log=res["range_hat_log"],
                    close_ref=float(tail["Close"].iloc[i - 1]),
                    cfg=self.cfg.reconstruct,
                )
                actual_h = float(tail["High"].iloc[i])
                actual_l = float(tail["Low"].iloc[i])
                rows.append({
                    "timestamp":         tail.index[i],
                    "theoretical_high":  rec["theoretical_high"],
                    "theoretical_low":   rec["theoretical_low"],
                    "theoretical_mid":   rec["theoretical_mid"],
                    "theoretical_range": rec["theoretical_range"],
                    "diseq_score":       res["diseq_score"],
                    "e_mid":   0.5 * (actual_h + actual_l) - rec["theoretical_mid"],
                    "e_range": (actual_h - actual_l)       - rec["theoretical_range"],
                })
            except Exception:
                continue

        if not rows:
            return None

        return pd.DataFrame(rows).set_index("timestamp")
