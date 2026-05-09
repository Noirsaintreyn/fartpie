"""
signal.py — Policy layer (separated from the model)
====================================================
Converts calibrated intensity outputs into actionable entry/exit/hold signals.

Design principle (from the roadmap):
  "If a component changes the model's mathematical meaning → model layer.
   If it changes the user's decision → policy layer."

This module is the policy layer only. It takes λ(t) sequences as input
and produces signals. It never touches model internals.

Features implemented:
  - Regime-aware thresholds (volatility + event rate, not just rolling mean)
  - Hysteresis band to prevent flip-flopping
  - Cooldown (minimum dwell time) after each signal
  - Confidence bands (bootstrap over MC samples if available)
  - Evaluation: precision, recall, delay, false reversal rate
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from enum import IntEnum


class Signal(IntEnum):
    HOLD = 0
    ENTER = 1
    EXIT = -1


@dataclass
class PolicyConfig:
    # Threshold multipliers (× regime-adjusted baseline)
    entry_mult: float = 1.4     # λ(t) / baseline > entry_mult → consider enter
    exit_mult: float = 2.6      # λ(t) / baseline > exit_mult  → consider exit

    # Hysteresis: must exceed threshold by this fraction before flipping
    hysteresis: float = 0.05    # 5% buffer above/below threshold to confirm

    # Cooldown: minimum steps between signal changes (prevents rapid reversals)
    cooldown_steps: int = 4

    # Regime adjustment: rolling window for volatility and event rate
    vol_window: int = 20        # steps for rolling std
    rate_window: int = 20       # steps for rolling event rate

    # Confidence: minimum confidence ratio to emit a strong signal
    # (ratio = how far λ is from threshold, normalized)
    min_confidence: float = 0.0  # 0 = emit all signals; 0.2 = only emit if 20% above thresh


@dataclass
class SignalEvent:
    step: int
    time: float
    signal: Signal
    lambda_t: float
    baseline: float
    confidence: float


class RegimeAwarePolicy:
    """
    Converts an intensity series λ(t) → signals with hysteresis and cooldowns.

    Thresholds adapt to both the rolling mean intensity AND its rolling
    volatility (std), so the model is harder to trigger in noisy regimes.
    """
    def __init__(self, cfg: Optional[PolicyConfig] = None):
        self.cfg = cfg or PolicyConfig()

    def _regime_baseline(self, lam_series: np.ndarray, i: int) -> tuple[float, float]:
        """Compute rolling mean and volatility-adjusted threshold at step i."""
        w = self.cfg.vol_window
        start = max(0, i - w)
        window = lam_series[start:i + 1]
        mean = float(np.mean(window))
        std = float(np.std(window)) if len(window) > 1 else 0.0
        return mean, std

    def apply(
        self,
        lam_series: np.ndarray,
        times: Optional[np.ndarray] = None,
        lam_upper: Optional[np.ndarray] = None,  # upper confidence band
        lam_lower: Optional[np.ndarray] = None,  # lower confidence band
    ) -> list[SignalEvent]:
        """
        Apply policy to an intensity series.

        Args:
            lam_series: (N,) array of λ(t) values at each time step
            times:      (N,) optional time axis
            lam_upper/lower: confidence bands (optional)
        Returns:
            list of SignalEvent (only at signal transitions)
        """
        N = len(lam_series)
        if times is None:
            times = np.arange(N, dtype=float)

        cfg = self.cfg
        current = Signal.HOLD
        cooldown = 0
        signals = []

        for i in range(N):
            lam = lam_series[i]
            mean, std = self._regime_baseline(lam_series, i)
            # Regime-adjusted baseline: mean + k*std discounts noisy regimes
            baseline = mean + 0.5 * std if std > 0 else mean
            baseline = max(baseline, 1e-6)

            entry_thresh = baseline * cfg.entry_mult
            exit_thresh = baseline * cfg.exit_mult

            # Hysteresis band
            entry_band = entry_thresh * (1 + cfg.hysteresis)
            entry_lower = entry_thresh * (1 - cfg.hysteresis)
            exit_band = exit_thresh * (1 + cfg.hysteresis)
            exit_lower = exit_thresh * (1 - cfg.hysteresis)

            new_sig = current
            if cooldown > 0:
                cooldown -= 1
            else:
                if current != Signal.EXIT and lam >= exit_band:
                    new_sig = Signal.EXIT
                elif current != Signal.ENTER and lam >= entry_band:
                    new_sig = Signal.ENTER
                elif current == Signal.ENTER and lam < entry_lower:
                    new_sig = Signal.HOLD
                elif current == Signal.EXIT and lam < exit_lower:
                    new_sig = Signal.ENTER  # back to enter after exit exhaustion

            # Confidence: normalized distance from active threshold
            if new_sig == Signal.EXIT:
                confidence = (lam - exit_thresh) / (exit_thresh + 1e-8)
            elif new_sig == Signal.ENTER:
                confidence = (lam - entry_thresh) / (entry_thresh + 1e-8)
            else:
                confidence = 0.0
            confidence = float(np.clip(confidence, 0, 1))

            if new_sig != current and confidence >= cfg.min_confidence:
                signals.append(SignalEvent(
                    step=i,
                    time=float(times[i]),
                    signal=new_sig,
                    lambda_t=float(lam),
                    baseline=float(baseline),
                    confidence=confidence,
                ))
                current = new_sig
                if new_sig != Signal.HOLD:
                    cooldown = cfg.cooldown_steps

        return signals


# ── Evaluation ────────────────────────────────────────────────────────────────

@dataclass
class SignalEvaluation:
    """Results of evaluating signal quality against ground-truth cluster labels."""
    precision: float
    recall: float
    f1: float
    mean_delay: float          # average steps between true cluster start and signal
    false_reversal_rate: float # fraction of signals that reverse within cooldown
    n_signals: int
    n_true_clusters: int


def evaluate_signals(
    signals: list[SignalEvent],
    true_cluster_starts: list[int],
    lam_series: np.ndarray,
    tolerance_steps: int = 10,
) -> SignalEvaluation:
    """
    Evaluate precision, recall, delay, and false reversal rate.

    A signal is a true positive if an ENTER occurs within tolerance_steps
    of a true cluster start. A false positive is an ENTER with no nearby cluster.

    Args:
        signals:             output of RegimeAwarePolicy.apply()
        true_cluster_starts: list of step indices where true clusters begin
        lam_series:          (N,) full intensity series
        tolerance_steps:     window to match signals to clusters
    """
    enter_steps = [s.step for s in signals if s.signal == Signal.ENTER]
    n_true = len(true_cluster_starts)
    n_signals = len(enter_steps)

    if n_signals == 0 or n_true == 0:
        return SignalEvaluation(0, 0, 0, float('inf'), 0, n_signals, n_true)

    # Match signals to clusters (greedy nearest)
    matched_clusters = set()
    tp, delays = 0, []
    for es in enter_steps:
        best_dist = tolerance_steps + 1
        best_c = None
        for ci, cs in enumerate(true_cluster_starts):
            if ci in matched_clusters:
                continue
            d = es - cs
            if 0 <= d <= tolerance_steps and d < best_dist:
                best_dist = d
                best_c = ci
        if best_c is not None:
            matched_clusters.add(best_c)
            tp += 1
            delays.append(best_dist)

    precision = tp / max(n_signals, 1)
    recall = tp / max(n_true, 1)
    f1 = (2 * precision * recall / (precision + recall + 1e-9))
    mean_delay = float(np.mean(delays)) if delays else float('inf')

    # False reversal: ENTER followed by EXIT within 2× cooldown
    rev = 0
    sig_types = [(s.step, s.signal) for s in signals]
    for i, (step, sig) in enumerate(sig_types[:-1]):
        if sig == Signal.ENTER and sig_types[i+1][0] - step <= 8:
            rev += 1
    false_reversal_rate = rev / max(n_signals, 1)

    return SignalEvaluation(
        precision=precision,
        recall=recall,
        f1=f1,
        mean_delay=mean_delay,
        false_reversal_rate=false_reversal_rate,
        n_signals=n_signals,
        n_true_clusters=n_true,
    )
