"""
theoretical_bounds
------------------
Theoretical HOD/LOD forecasting pipeline.

Architecture:
  Layer 1: VECM structural envelope on (mid_log, range_log)
  Layer 2: LightGBM residual corrector on normalised (e_mid, e_range)
  Layer 3: Hard constraints + confidence bands
  Layer 4: Corrected HOD/LOD for entry/exit trigger integration

Quick start:
    from theoretical_bounds import TheoreticalBoundsPipeline
    pipe = TheoreticalBoundsPipeline()
    results = pipe.run(ohlcv_df)
"""

from .pipeline      import TheoreticalBoundsPipeline, PipelineConfig
from .vecm_model    import VECMConfig
from .reconstruct   import ReconstructConfig
from .corrector     import ResidualCorrector, CorrectorConfig
from .features      import validate_ohlcv, add_mid_range, build_residual_features, symbol_registry, SymbolRegistry
from .live_predictor import LivePredictor, LiveForecast

__all__ = [
    "TheoreticalBoundsPipeline",
    "PipelineConfig",
    "VECMConfig",
    "ReconstructConfig",
    "ResidualCorrector",
    "CorrectorConfig",
    "validate_ohlcv",
    "add_mid_range",
    "build_residual_features",
    "symbol_registry",
    "SymbolRegistry",
    "LivePredictor",
    "LiveForecast",
]
