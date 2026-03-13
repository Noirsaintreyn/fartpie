"""
theoretical_bounds/corrector.py
---------------------------------
LightGBM residual corrector.
Targets: e_mid_norm = e_mid / ATR,  e_range_norm = e_range / ATR
Predicts how much the VECM structural forecast misses, then corrects it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False
    warnings.warn("LightGBM not installed. Install with: pip install lightgbm")

from .features import get_feature_columns


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CorrectorConfig:
    lag_periods: int = 5

    # LightGBM hyperparams
    n_estimators: int     = 400
    learning_rate: float  = 0.03
    num_leaves: int       = 31
    min_child_samples: int = 20
    subsample: float      = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float      = 0.1
    reg_lambda: float     = 0.1

    # Training split
    test_fraction: float  = 0.2        # held-out evaluation set
    min_train_rows: int   = 120        # minimum rows to train

    # Clipping predictions (in ATR multiples)
    clip_e_mid_atr: float   = 3.0
    clip_e_range_atr: float = 3.0

    # Instrument conditioning
    symbol_mode:    str               = "embedding"   # "embedding" | "one_hot" | "none"
    known_symbols:  Optional[List[str]] = None        # for one_hot: all symbols in training set


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class ResidualCorrector:
    """
    Two LightGBM models:
      - model_mid   : predicts e_mid_norm
      - model_range : predicts e_range_norm

    Usage
    -----
    corrector = ResidualCorrector(cfg)
    corrector.train(feature_df)
    predictions = corrector.predict(feature_df)
    """

    def __init__(self, cfg: Optional[CorrectorConfig] = None):
        if not _LGB_AVAILABLE:
            raise ImportError("LightGBM required. pip install lightgbm")
        self.cfg = cfg or CorrectorConfig()
        self.model_mid:   Optional[lgb.Booster] = None
        self.model_range: Optional[lgb.Booster] = None
        self.feature_cols: List[str] = get_feature_columns(
            lag_periods=self.cfg.lag_periods,
            symbol_mode=self.cfg.symbol_mode,
            known_symbols=self.cfg.known_symbols,
        )
        self._categorical_features: List[str] = (
            ["symbol_id"] if self.cfg.symbol_mode == "embedding" else []
        )
        self._train_metrics: Dict = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        feature_df: pd.DataFrame,
        verbose: bool = True,
    ) -> Dict:
        """
        Train both correctors on feature_df.
        feature_df must be the output of build_residual_features().

        Returns training metrics dict.
        """
        df = feature_df.copy()
        df = df.dropna(subset=self.feature_cols + ["e_mid_norm", "e_range_norm"])

        if len(df) < self.cfg.min_train_rows:
            raise ValueError(
                f"Need at least {self.cfg.min_train_rows} clean rows to train, "
                f"got {len(df)}"
            )

        # Train / test split (time-ordered)
        split_idx = int(len(df) * (1 - self.cfg.test_fraction))
        train_df  = df.iloc[:split_idx]
        test_df   = df.iloc[split_idx:]

        X_train = train_df[self.feature_cols]
        X_test  = test_df[self.feature_cols]

        metrics = {}

        for target_col, attr_name in [
            ("e_mid_norm",   "model_mid"),
            ("e_range_norm", "model_range"),
        ]:
            y_train = train_df[target_col]
            y_test  = test_df[target_col]

            lgb_train = lgb.Dataset(
                X_train,
                label=y_train,
                categorical_feature=self._categorical_features or "auto",
            )
            lgb_val = lgb.Dataset(
                X_test,
                label=y_test,
                reference=lgb_train,
                categorical_feature=self._categorical_features or "auto",
            )

            params = {
                "objective":        "regression",
                "metric":           "rmse",
                "num_leaves":       self.cfg.num_leaves,
                "learning_rate":    self.cfg.learning_rate,
                "min_child_samples": self.cfg.min_child_samples,
                "subsample":        self.cfg.subsample,
                "colsample_bytree": self.cfg.colsample_bytree,
                "reg_alpha":        self.cfg.reg_alpha,
                "reg_lambda":       self.cfg.reg_lambda,
                "verbose":          -1,
            }

            callbacks = [lgb.early_stopping(50, verbose=False)]
            if verbose:
                callbacks.append(lgb.log_evaluation(50))

            model = lgb.train(
                params,
                lgb_train,
                num_boost_round=self.cfg.n_estimators,
                valid_sets=[lgb_val],
                callbacks=callbacks,
            )

            if attr_name == "model_mid":
                self.model_mid = model
            else:
                self.model_range = model

            preds = model.predict(X_test)
            mae   = float(np.mean(np.abs(preds - y_test.values)))
            rmse  = float(np.sqrt(np.mean((preds - y_test.values) ** 2)))
            metrics[target_col] = {
                "test_mae":  mae,
                "test_rmse": rmse,
                "n_train":   len(train_df),
                "n_test":    len(test_df),
                "best_iteration": model.best_iteration,
            }

        self._train_metrics = metrics
        return metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict normalised residuals for all rows in feature_df.

        Returns DataFrame with columns:
          e_mid_hat_norm, e_range_hat_norm  (clipped)
        """
        if self.model_mid is None or self.model_range is None:
            raise RuntimeError("Models not trained. Call .train() first.")

        df = feature_df.copy()
        present_feat_cols = [c for c in self.feature_cols if c in df.columns]
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            warnings.warn(f"Missing feature columns (will fill with 0): {missing}")
            for c in missing:
                df[c] = 0.0

        X = df[self.feature_cols].fillna(0.0)

        e_mid_hat_norm   = self.model_mid.predict(X)
        e_range_hat_norm = self.model_range.predict(X)

        # Clip in normalised space
        e_mid_hat_norm   = np.clip(
            e_mid_hat_norm,
            -self.cfg.clip_e_mid_atr,
            self.cfg.clip_e_mid_atr,
        )
        e_range_hat_norm = np.clip(
            e_range_hat_norm,
            -self.cfg.clip_e_range_atr,
            self.cfg.clip_e_range_atr,
        )

        return pd.DataFrame(
            {
                "e_mid_hat_norm":   e_mid_hat_norm,
                "e_range_hat_norm": e_range_hat_norm,
            },
            index=df.index,
        )

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self) -> pd.DataFrame:
        """Return sorted feature importance for both targets."""
        if self.model_mid is None:
            raise RuntimeError("Models not trained.")

        imp_mid = pd.Series(
            self.model_mid.feature_importance(importance_type="gain"),
            index=self.feature_cols,
            name="importance_mid",
        )
        imp_range = pd.Series(
            self.model_range.feature_importance(importance_type="gain"),
            index=self.feature_cols,
            name="importance_range",
        )
        return (
            pd.concat([imp_mid, imp_range], axis=1)
            .sort_values("importance_mid", ascending=False)
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Save both models to directory."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        if self.model_mid:
            self.model_mid.save_model(str(path / "model_mid.lgb"))
        if self.model_range:
            self.model_range.save_model(str(path / "model_range.lgb"))

    def load(self, directory: str) -> None:
        """Load both models from directory."""
        path = Path(directory)
        self.model_mid   = lgb.Booster(model_file=str(path / "model_mid.lgb"))
        self.model_range = lgb.Booster(model_file=str(path / "model_range.lgb"))
