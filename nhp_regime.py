"""
regime.py — Volatility features + HMM regime detector
=======================================================
Based on:
  - Fabre & Toke (2025): Markov-modulated Hawkes for regime-switching intensity
  - Shi & Cartlidge (2022): State-dependent NHP — market state fed alongside events
  - Bacry et al. (2015): Realized vol directly linked to Hawkes clustering

Design:
  VolatilityState: computes realized vol, vol-of-vol, return skew from price series
  RegimeDetector:  fits a 3-state Gaussian HMM on vol features → LOW/NORMAL/HIGH
  StateConditioner: gates NHP intensity by regime probability

The regime state is fed into the event embedding (not the intensity head directly),
following Shi & Cartlidge's state-dependent parallel NHP design.
"""

import numpy as np
import warnings
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from hmmlearn import hmm


# ── Regime labels ─────────────────────────────────────────────────────────────

class Regime(IntEnum):
    LOW = 0       # quiet, mean-reverting
    NORMAL = 1    # baseline trending
    HIGH = 2      # volatile, jump-prone


REGIME_LABELS = {Regime.LOW: "Low vol", Regime.NORMAL: "Normal", Regime.HIGH: "High vol"}


# ── Volatility feature extraction ─────────────────────────────────────────────

@dataclass
class VolFeatures:
    """
    Realized volatility surface computed from a price series.
    All features are causal (no lookahead).
    """
    realized_vol: np.ndarray    # rolling std of log returns
    vol_of_vol:   np.ndarray    # rolling std of realized vol (2nd-order clustering)
    return_skew:  np.ndarray    # rolling skewness (asymmetric jump indicator)
    log_returns:  np.ndarray    # raw log returns
    times:        np.ndarray    # timestamps aligned to features


def compute_vol_features(
    prices: np.ndarray,
    times: np.ndarray,
    short_window: int = 20,    # ~1 day at 5-min bars
    long_window: int = 60,     # ~3 days
) -> VolFeatures:
    """
    Compute causal volatility features from a price series.

    Args:
        prices: (N,) array of prices (any frequency)
        times:  (N,) timestamps (seconds or bar index)
        short_window: bars for realized vol
        long_window:  bars for vol-of-vol
    """
    assert len(prices) == len(times), "prices and times must be same length"
    n = len(prices)

    log_ret = np.diff(np.log(np.maximum(prices, 1e-10)), prepend=0.0)

    realized_vol = np.zeros(n)
    vol_of_vol   = np.zeros(n)
    return_skew  = np.zeros(n)

    for i in range(1, n):
        w = short_window
        start = max(0, i - w)
        window = log_ret[start:i]
        if len(window) > 1:
            realized_vol[i] = float(np.std(window))
        else:
            realized_vol[i] = realized_vol[i - 1]

        # Vol-of-vol: rolling std of the realized vol series
        lw = long_window
        vstart = max(0, i - lw)
        vwindow = realized_vol[vstart:i]
        if len(vwindow) > 1:
            vol_of_vol[i] = float(np.std(vwindow))
        else:
            vol_of_vol[i] = vol_of_vol[i - 1]

        # Skew proxy: (mean of cubed returns) / vol^3
        if len(window) > 2 and realized_vol[i] > 1e-10:
            return_skew[i] = float(np.mean(window ** 3)) / (realized_vol[i] ** 3 + 1e-10)
            return_skew[i] = float(np.clip(return_skew[i], -5, 5))

    return VolFeatures(
        realized_vol=realized_vol,
        vol_of_vol=vol_of_vol,
        return_skew=return_skew,
        log_returns=log_ret,
        times=times,
    )


# ── HMM Regime Detector ───────────────────────────────────────────────────────

class RegimeDetector:
    """
    3-state Gaussian HMM on (realized_vol, vol_of_vol, |return_skew|).
    States are relabeled by mean realized vol: LOW=0, NORMAL=1, HIGH=2.

    Follows Fabre & Toke (2025) Markov-modulated Hawkes: the hidden Markov
    chain modulates the intensity — here it gates the NHP signal layer.

    Training uses only the train split; inference is strictly causal
    (Viterbi decode on expanding window, or online filtering).
    """

    def __init__(self, n_states: int = 3, n_iter: int = 200, random_state: int = 42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.model: Optional[hmm.GaussianHMM] = None
        self._state_map: dict[int, Regime] = {}  # raw HMM state → Regime enum

    def _feature_matrix(self, vf: VolFeatures) -> np.ndarray:
        """Stack vol features into (N, 3) observation matrix."""
        X = np.column_stack([
            vf.realized_vol,
            vf.vol_of_vol,
            np.abs(vf.return_skew),
        ])
        # Standardize to help HMM convergence
        self._feat_mean = X.mean(axis=0)
        self._feat_std  = X.std(axis=0) + 1e-8
        return (X - self._feat_mean) / self._feat_std

    def fit(self, vf: VolFeatures) -> "RegimeDetector":
        """Fit HMM on training volatility features."""
        X = self._feature_matrix(vf)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type="diag",
                n_iter=self.n_iter,
                random_state=self.random_state,
                tol=1e-4,
            )
            self.model.fit(X)

        # Relabel states by ascending mean realized vol
        raw_means = self.model.means_[:, 0]  # first feature = realized vol
        order = np.argsort(raw_means)
        self._state_map = {int(order[i]): Regime(i) for i in range(self.n_states)}
        return self

    def _normalize(self, vf: VolFeatures) -> np.ndarray:
        X = np.column_stack([
            vf.realized_vol,
            vf.vol_of_vol,
            np.abs(vf.return_skew),
        ])
        return (X - self._feat_mean) / self._feat_std

    def predict(self, vf: VolFeatures) -> tuple[np.ndarray, np.ndarray]:
        """
        Decode regimes on new data (Viterbi path).
        Returns:
            regimes:     (N,) array of Regime enum values
            probs:       (N, 3) posterior probabilities per state
        """
        assert self.model is not None, "Call fit() first"
        X = self._normalize(vf)
        raw_states = self.model.predict(X)
        probs_raw  = self.model.predict_proba(X)

        regimes = np.array([self._state_map[s] for s in raw_states], dtype=int)

        # Reorder probability columns to match LOW/NORMAL/HIGH
        probs = np.zeros((len(X), self.n_states))
        for raw_s, regime in self._state_map.items():
            probs[:, int(regime)] = probs_raw[:, raw_s]

        return regimes, probs

    def current_regime(self, vf: VolFeatures, idx: int) -> tuple[Regime, float]:
        """
        Return regime and confidence at a single time step.
        Uses causal prediction up to idx.
        """
        regimes, probs = self.predict(vf)
        r = Regime(regimes[idx])
        conf = float(probs[idx, int(r)])
        return r, conf


# ── State feature vector for NHP embedding ───────────────────────────────────

def state_vector_at(
    vf: VolFeatures,
    regimes: np.ndarray,
    regime_probs: np.ndarray,
    idx: int,
) -> np.ndarray:
    """
    Build a state context vector at time step idx for injection into
    the NHP event embedding (following Shi & Cartlidge 2022).

    Returns: (6,) float32 vector:
        [realized_vol, vol_of_vol, |skew|, P(LOW), P(NORMAL), P(HIGH)]
    """
    vol  = float(vf.realized_vol[idx])
    vvol = float(vf.vol_of_vol[idx])
    skew = float(abs(vf.return_skew[idx]))
    probs = regime_probs[idx].tolist()
    return np.array([vol, vvol, skew] + probs, dtype=np.float32)


# ── Regime-conditioned signal gate ────────────────────────────────────────────

@dataclass
class RegimeGateConfig:
    """
    Controls which regime states allow which signals.
    Based on the literature finding that signals are most reliable
    when regime state is well-identified (high confidence).
    """
    # Minimum HMM posterior confidence to trust the regime label
    min_confidence: float = 0.65

    # Which regimes allow ENTER signals
    # LOW vol: mean-reverting, clustering starting → good entry
    # NORMAL: ok
    # HIGH vol: overcrowded, exit risk → suppress entry
    enter_allowed_regimes: tuple = (Regime.LOW, Regime.NORMAL)

    # Which regimes allow EXIT signals
    # HIGH vol: cluster exhaustion → good exit
    # NORMAL: ok if intensity spikes
    exit_allowed_regimes: tuple = (Regime.NORMAL, Regime.HIGH)

    # Volatility ceiling: don't enter if realized vol > this multiple of median
    vol_ceiling_mult: float = 2.5

    # Volatility floor: don't exit if vol is too quiet (false alarm)
    vol_floor_mult: float = 0.5


class RegimeGate:
    """
    Filters NHP signals through regime + volatility conditions.
    This is strictly a policy-layer component (no model changes).
    """

    def __init__(self, cfg: Optional[RegimeGateConfig] = None):
        self.cfg = cfg or RegimeGateConfig()

    def allows_enter(
        self,
        regime: Regime,
        regime_conf: float,
        realized_vol: float,
        median_vol: float,
    ) -> tuple[bool, str]:
        c = self.cfg
        if regime_conf < c.min_confidence:
            return False, f"regime confidence {regime_conf:.2f} < {c.min_confidence}"
        if regime not in c.enter_allowed_regimes:
            return False, f"regime {REGIME_LABELS[regime]} not in enter-allowed set"
        if median_vol > 0 and realized_vol > median_vol * c.vol_ceiling_mult:
            return False, f"vol {realized_vol:.4f} exceeds ceiling {median_vol * c.vol_ceiling_mult:.4f}"
        return True, "ok"

    def allows_exit(
        self,
        regime: Regime,
        regime_conf: float,
        realized_vol: float,
        median_vol: float,
    ) -> tuple[bool, str]:
        c = self.cfg
        if regime_conf < c.min_confidence:
            return False, f"regime confidence {regime_conf:.2f} < {c.min_confidence}"
        if regime not in c.exit_allowed_regimes:
            return False, f"regime {REGIME_LABELS[regime]} not in exit-allowed set"
        if median_vol > 0 and realized_vol < median_vol * c.vol_floor_mult:
            return False, f"vol {realized_vol:.4f} below floor {median_vol * c.vol_floor_mult:.4f}"
        return True, "ok"
