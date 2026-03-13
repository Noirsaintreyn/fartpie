"""
theoretical_bounds/pipeline.py
---------------------------------
Full pipeline:
  1. OHLCV → midpoint/range transform
  2. Rolling VECM → theoretical envelope
  3. Residual feature construction
  4. LightGBM corrector training + prediction
  5. Final corrected HOD/LOD reconstruction
  6. Benchmark comparison (VECM vs corrected vs baseline)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

from .features import validate_ohlcv, add_mid_range, build_residual_features, SymbolRegistry
from .vecm_model import VECMConfig, rolling_vecm_forecast, run_all_diagnostics, fit_and_forecast
from .reconstruct import ReconstructConfig, reconstruct_dataframe, apply_correction, reconstruct_bounds
from .corrector import CorrectorConfig, ResidualCorrector


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    vecm:        VECMConfig        = field(default_factory=VECMConfig)
    reconstruct: ReconstructConfig = field(default_factory=ReconstructConfig)
    corrector:   CorrectorConfig   = field(default_factory=CorrectorConfig)
    atr_period:  int               = 14
    lag_periods: int               = 5
    run_diagnostics: bool          = True
    # Instrument conditioning
    symbol:      Optional[str]     = None
    symbol_mode: str               = "embedding"  # "embedding" | "one_hot" | "none"
    # Pass a shared SymbolRegistry so train and inference use identical id mappings.
    # If None, a fresh registry is created on first run() call.
    registry:    Optional[SymbolRegistry] = None


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _trading_metrics(
    df: pd.DataFrame,
    pred_high_col: str,
    pred_low_col: str,
    actual_high_col: str = "actual_high",
    actual_low_col:  str = "actual_low",
    atr_col:         str = "atr",
    breach_k_values: tuple = (0.5, 1.0, 2.0),
) -> Dict[str, float]:
    """
    Full trading-oriented metric suite for envelope forecasts.

    Includes:
      - MAE / RMSE / bias  (forecast accuracy)
      - hit_rate           : fraction of days where actual_high ≤ pred_high
                             AND actual_low ≥ pred_low  (envelope containment)
      - high_containment   : fraction of days actual_high ≤ pred_high alone
      - low_containment    : fraction of days actual_low  ≥ pred_low  alone
      - breach_high_k_atr  : fraction of days actual_high > pred_high + k*ATR
      - breach_low_k_atr   : fraction of days actual_low  < pred_low  - k*ATR
      - mean_breach_high   : mean overshoot on days that breach the high (in ATR)
      - mean_breach_low    : mean undershoot on days that breach the low (in ATR)
    """
    cols = [pred_high_col, pred_low_col, actual_high_col, actual_low_col, atr_col]
    sub  = df[[c for c in cols if c in df.columns]].dropna()
    if sub.empty:
        return {}

    ph = sub[pred_high_col]
    pl = sub[pred_low_col]
    ah = sub[actual_high_col]
    al = sub[actual_low_col]
    atr = sub[atr_col] if atr_col in sub.columns else pd.Series(1.0, index=sub.index)

    h_err = ph - ah    # positive = predicted too high (conservative)
    l_err = al - pl    # positive = predicted too low  (conservative)

    # Containment
    high_contained = (ah <= ph)
    low_contained  = (al >= pl)
    both_contained = high_contained & low_contained

    # Breach magnitude in ATR multiples (only on breach days)
    high_breach_atr = ((ah - ph) / atr.replace(0, np.nan)).clip(lower=0)
    low_breach_atr  = ((pl - al) / atr.replace(0, np.nan)).clip(lower=0)

    result: Dict[str, float] = {
        # Accuracy
        f"{pred_high_col}_mae":  float(h_err.abs().mean()),
        f"{pred_high_col}_rmse": float(np.sqrt((h_err ** 2).mean())),
        f"{pred_high_col}_bias": float(h_err.mean()),
        f"{pred_low_col}_mae":   float(l_err.abs().mean()),
        f"{pred_low_col}_rmse":  float(np.sqrt((l_err ** 2).mean())),
        f"{pred_low_col}_bias":  float(l_err.mean()),
        "range_mae": float(
            ((ph - pl) - (ah - al)).abs().mean()
        ),
        # Containment / hit-rate
        "hit_rate":          float(both_contained.mean()),
        "high_containment":  float(high_contained.mean()),
        "low_containment":   float(low_contained.mean()),
        # Breach stats
        "mean_breach_high_atr": float(high_breach_atr[high_breach_atr > 0].mean())
            if (high_breach_atr > 0).any() else 0.0,
        "mean_breach_low_atr":  float(low_breach_atr[low_breach_atr > 0].mean())
            if (low_breach_atr > 0).any() else 0.0,
        "n": len(sub),
    }

    # Breach rate at each k-ATR threshold
    for k in breach_k_values:
        result[f"breach_high_{k}atr"] = float((ah > ph + k * atr).mean())
        result[f"breach_low_{k}atr"]  = float((al < pl - k * atr).mean())

    return result


def _baseline_high_low(ohlcv: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """
    Simple baseline: prior-close +/- ATR/2.
    This approximates a naive 'static level + vol' system.
    """
    close = ohlcv["Close"]
    high  = ohlcv["High"]
    low   = ohlcv["Low"]
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(span=atr_period, adjust=False).mean().shift(1)

    return pd.DataFrame({
        "baseline_high": prev_close + 0.5 * atr,
        "baseline_low":  prev_close - 0.5 * atr,
    }, index=ohlcv.index)


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class TheoreticalBoundsPipeline:
    """
    End-to-end theoretical HOD/LOD pipeline.

    Quick start
    -----------
    pipe = TheoreticalBoundsPipeline()
    results = pipe.run(ohlcv_df)
    print(results[["theoretical_high","theoretical_low","corrected_high","corrected_low"]].tail(10))
    """

    def __init__(self, cfg: Optional[PipelineConfig] = None):
        self.cfg       = cfg or PipelineConfig()
        # Ensure a stable registry exists for the lifetime of this pipeline
        if self.cfg.registry is None:
            self.cfg.registry = SymbolRegistry(
                [self.cfg.symbol] if self.cfg.symbol else []
            )
        self.corrector = ResidualCorrector(self.cfg.corrector)
        self._results:  Optional[pd.DataFrame] = None
        self._diag:     Optional[pd.DataFrame] = None
        self._metrics:  Dict[str, Any]         = {}

    # ------------------------------------------------------------------
    # Step 1+2: VECM structural envelope
    # ------------------------------------------------------------------

    def _run_vecm(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns (vecm_forecasts, recon_df)
        """
        vecm_df = rolling_vecm_forecast(data, self.cfg.vecm)
        recon_df = reconstruct_dataframe(vecm_df, data, self.cfg.reconstruct)
        return vecm_df, recon_df

    # ------------------------------------------------------------------
    # Step 3: Residual features
    # ------------------------------------------------------------------

    def _build_features(
        self,
        recon_df:  pd.DataFrame,
        vecm_df:   pd.DataFrame,
        ohlcv:     pd.DataFrame,
    ) -> pd.DataFrame:
        # Attach actual highs/lows to recon_df for residual computation
        actuals = ohlcv[["High", "Low", "Close"]].rename(
            columns={"High": "actual_high", "Low": "actual_low", "Close": "actual_close"}
        )
        mid_range = add_mid_range(ohlcv)[["mid_actual", "range_actual"]]

        full = recon_df.join(actuals).join(mid_range).join(
            vecm_df[["diseq_score", "selected_lags", "selected_rank"]]
        )

        # Raw residuals
        full["e_mid"]   = full["actual_high"].add(full["actual_low"]).div(2) - full["theoretical_mid"]
        full["e_range"] = full["actual_high"].sub(full["actual_low"])        - full["theoretical_range"]
        full["e_high"]  = full["actual_high"] - full["theoretical_high"]
        full["e_low"]   = full["actual_low"]  - full["theoretical_low"]

        feat_df = build_residual_features(
            bounds_df=full,
            ohlcv_df=ohlcv,
            atr_period=self.cfg.atr_period,
            lag_periods=self.cfg.lag_periods,
            symbol=self.cfg.symbol,
            symbol_mode=self.cfg.symbol_mode,
            registry=self.cfg.registry,
        )
        return feat_df

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def run(
        self,
        ohlcv_df:    pd.DataFrame,
        train: bool  = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run the full pipeline on ohlcv_df.

        Parameters
        ----------
        ohlcv_df : raw OHLCV DataFrame
        train    : if True, trains the LightGBM corrector on available data
        verbose  : print training and benchmark metrics

        Returns
        -------
        DataFrame with columns:
          theoretical_high / theoretical_low  (VECM only)
          corrected_high   / corrected_low    (VECM + LightGBM)
          plus all intermediate columns
        """
        # --- Validate + transform
        ohlcv = validate_ohlcv(ohlcv_df)
        data  = add_mid_range(ohlcv)

        # --- Diagnostics
        if self.cfg.run_diagnostics:
            self._diag = run_all_diagnostics(data)
            if verbose:
                print("\n=== Series Diagnostics ===")
                print(self._diag.to_string())

        # --- VECM envelope
        if verbose:
            print("\n=== Running rolling VECM... ===")
        vecm_df, recon_df = self._run_vecm(data)

        # --- Residual features
        if verbose:
            print("=== Building residual features... ===")
        feat_df = self._build_features(recon_df, vecm_df, ohlcv)

        # --- LightGBM corrector
        if train:
            if verbose:
                print("=== Training LightGBM corrector... ===")
            train_metrics = self.corrector.train(feat_df, verbose=verbose)
            self._metrics["lgb_train"] = train_metrics
            if verbose:
                print("\nLightGBM training metrics:")
                for tgt, m in train_metrics.items():
                    print(f"  {tgt}: MAE={m['test_mae']:.4f}, RMSE={m['test_rmse']:.4f}")

        # --- Predict residuals and apply correction
        preds = self.corrector.predict(feat_df)
        feat_df = feat_df.join(preds)

        # Apply corrections row-wise
        corrected_rows = []
        for ts, row in feat_df.iterrows():
            if pd.isna(row.get("theoretical_mid")) or pd.isna(row.get("e_mid_hat_norm")):
                corrected_rows.append({
                    "timestamp": ts,
                    "corrected_high": np.nan,
                    "corrected_low":  np.nan,
                    "corrected_mid":  np.nan,
                    "corrected_range": np.nan,
                })
                continue

            corr = apply_correction(
                theoretical_mid=float(row["theoretical_mid"]),
                theoretical_range=float(row["theoretical_range"]),
                e_mid_hat=float(row["e_mid_hat_norm"]),
                e_range_hat=float(row["e_range_hat_norm"]),
                atr=float(row["atr"]) if not pd.isna(row.get("atr")) else 1.0,
                cfg=self.cfg.reconstruct,
            )
            corr["timestamp"] = ts
            corrected_rows.append(corr)

        corrected_df = pd.DataFrame(corrected_rows).set_index("timestamp")
        results = feat_df.join(corrected_df, rsuffix="_corr")

        # --- Baseline
        baseline = _baseline_high_low(ohlcv, atr_period=self.cfg.atr_period)
        results = results.join(baseline)

        self._results = results

        # --- Benchmark
        if verbose:
            self._print_benchmark(results)

        return results

    # ------------------------------------------------------------------
    # Benchmark reporting
    # ------------------------------------------------------------------

    def _print_benchmark(self, results: pd.DataFrame) -> None:
        print("\n=== Benchmark: VECM vs Corrected vs Baseline ===")

        model_pairs = {
            "VECM (structural)":    ("theoretical_high", "theoretical_low"),
            "Corrected (VECM+LGB)": ("corrected_high",   "corrected_low"),
            "Baseline (close+ATR)": ("baseline_high",    "baseline_low"),
        }

        all_metrics: Dict[str, Dict] = {}
        for label, (hcol, lcol) in model_pairs.items():
            m = _trading_metrics(results, pred_high_col=hcol, pred_low_col=lcol)
            all_metrics[label] = m

        # ---- Compact aligned table ----
        metric_order = [
            ("hit_rate",             "Hit rate (both)"),
            ("high_containment",     "High containment"),
            ("low_containment",      "Low containment"),
            ("breach_high_0.5atr",   "Breach high >0.5 ATR"),
            ("breach_high_1.0atr",   "Breach high >1.0 ATR"),
            ("breach_high_2.0atr",   "Breach high >2.0 ATR"),
            ("breach_low_0.5atr",    "Breach low  >0.5 ATR"),
            ("breach_low_1.0atr",    "Breach low  >1.0 ATR"),
            ("breach_low_2.0atr",    "Breach low  >2.0 ATR"),
            ("mean_breach_high_atr", "Mean high breach (ATR)"),
            ("mean_breach_low_atr",  "Mean low  breach (ATR)"),
            ("range_mae",            "Range MAE"),
            ("n",                    "N days"),
        ]

        col_w  = 24
        labels = list(all_metrics.keys())
        header = f"{'Metric':<30}" + "".join(f"{l:<{col_w}}" for l in labels)
        print("\n" + header)
        print("-" * (30 + col_w * len(labels)))

        for key, display in metric_order:
            row = f"{display:<30}"
            for label in labels:
                val = all_metrics[label].get(key, float("nan"))
                if key == "n":
                    row += f"{int(val):<{col_w}}"
                elif isinstance(val, float):
                    # rates as % for containment/breach, raw for others
                    if "containment" in key or "hit_rate" in key or "breach_" in key:
                        row += f"{val * 100:>8.1f}%{'':<{col_w-10}}"
                    else:
                        row += f"{val:>{col_w-2}.4f}  "
                else:
                    row += f"{str(val):<{col_w}}"
            print(row)

        self._metrics["benchmark"] = all_metrics

    def benchmark(self) -> Dict[str, Any]:
        """Return stored benchmark metrics dict."""
        return self._metrics.get("benchmark", {})

    def diagnostics(self) -> Optional[pd.DataFrame]:
        """Return series diagnostics DataFrame."""
        return self._diag

    def feature_importance(self) -> pd.DataFrame:
        """Return LightGBM feature importances."""
        return self.corrector.feature_importance()

    def save_corrector(self, directory: str) -> None:
        """Persist trained LightGBM models and symbol registry."""
        from pathlib import Path
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.corrector.save(directory)
        if self.cfg.registry is not None:
            self.cfg.registry.save(str(path / "symbol_registry.json"))

    def load_corrector(self, directory: str) -> None:
        """Load persisted LightGBM models and symbol registry (skip training)."""
        from pathlib import Path
        path = Path(directory)
        self.corrector.load(directory)
        registry_path = path / "symbol_registry.json"
        if registry_path.exists():
            self.cfg.registry = SymbolRegistry.load(str(registry_path))
