"""
policy.py — Kalman-based signal generation policy
==================================================
Generates ENTER/EXIT signals from Kalman-smoothed NHP intensity slopes.

Uses slope_z (standardized rate of change of smoothed λ) rather than
raw λ thresholds — detects "λ is rising" (cluster forming → ENTER) and
"λ is falling" (cluster exhausting → EXIT).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from kalman import KalmanOutput


@dataclass
class Signal:
    step:       int
    signal:     str      # 'ENTER' or 'EXIT'
    confidence: float    # |slope_z| / 3.0, clipped to [0, 1]
    slope_z:    float    # raw slope_z value at this step
    lambda_t:   float    # smoothed lambda at this step


@dataclass
class KalmanPolicyConfig:
    entry_slope_z:   float = 1.0    # slope_z threshold for ENTER (λ rising)
    exit_slope_z:    float = -1.0   # slope_z threshold for EXIT (λ falling)
    min_confidence:  float = 0.0    # minimum confidence to emit signal
    cooldown_steps:  int   = 6      # bars between signals
    min_lambda:      float = 0.0    # minimum smoothed λ for any signal


@dataclass
class EvalResult:
    precision: float
    recall:    float
    f1:        float
    n_signals: int
    n_true:    int
    tp:        int
    fp:        int
    fn:        int


class KalmanPolicy:
    def __init__(self, cfg: KalmanPolicyConfig):
        self.cfg = cfg

    def apply(self, kout: KalmanOutput) -> list[Signal]:
        """
        Generate signals from Kalman-filtered intensity.

        ENTER when slope_z > entry_slope_z (intensity rising = cluster forming)
        EXIT  when slope_z < exit_slope_z  (intensity falling = cluster exhausting)
        """
        n = len(kout.slope_z)
        signals = []
        cooldown = 0

        for t in range(1, n):
            if cooldown > 0:
                cooldown -= 1
                continue

            sz = kout.slope_z[t]
            lam = kout.smoothed[t]
            conf = min(abs(sz) / 3.0, 1.0)

            if conf < self.cfg.min_confidence:
                continue
            if lam < self.cfg.min_lambda:
                continue

            if sz > self.cfg.entry_slope_z:
                signals.append(Signal(
                    step=t, signal='ENTER', confidence=conf,
                    slope_z=sz, lambda_t=lam,
                ))
                cooldown = self.cfg.cooldown_steps

            elif sz < self.cfg.exit_slope_z:
                signals.append(Signal(
                    step=t, signal='EXIT', confidence=conf,
                    slope_z=sz, lambda_t=lam,
                ))
                cooldown = self.cfg.cooldown_steps

        return signals


def evaluate_signals(
    signals:      list[Signal],
    true_starts:  list[int],
    lam_np:       np.ndarray,
    tolerance:    int = 3,
) -> EvalResult:
    """
    Evaluate signal precision/recall against known cluster starts.

    A signal is a true positive if it's within `tolerance` steps of a
    true cluster start. Only ENTER signals are evaluated (cluster detection).
    """
    enter_signals = [s for s in signals if s.signal == 'ENTER']
    n_signals = len(enter_signals)
    n_true = len(true_starts)

    if n_signals == 0 and n_true == 0:
        return EvalResult(1.0, 1.0, 1.0, 0, 0, 0, 0, 0)
    if n_signals == 0:
        return EvalResult(0.0, 0.0, 0.0, 0, n_true, 0, 0, n_true)
    if n_true == 0:
        return EvalResult(0.0, 0.0, 0.0, n_signals, 0, 0, n_signals, 0)

    # Match signals to true starts (greedy, closest first)
    matched_true = set()
    matched_sig = set()

    for i, sig in enumerate(enter_signals):
        for j, ts in enumerate(true_starts):
            if j in matched_true:
                continue
            if abs(sig.step - ts) <= tolerance:
                matched_sig.add(i)
                matched_true.add(j)
                break

    tp = len(matched_sig)
    fp = n_signals - tp
    fn = n_true - len(matched_true)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return EvalResult(precision, recall, f1, n_signals, n_true, tp, fp, fn)
