"""
Production-callable version of the validated V1 target model
(backtest_target_selection_eq.py + calibrate_target_model_walkforward.py,
2026-09-23): given an arbitrary entry/stop/target/horizon, returns a
calibrated P(target reached before stop).

Deliberately the SIMPLEST surviving version - plain Gaussian-OU + isotonic
calibration. The two "improvements" tried on top (GJR-skewt tails,
regime-HMM+SETAR) did NOT survive walk-forward calibration and are not
used here - see project_target_selection_calibration memory.

Validated result: isotonic-calibrated Gaussian-OU beat naive base-rate
Brier in 4/5 walk-forward folds spanning ~2 years (one fold - the most
recent, 2026-06 to 2026-09 - showed AUC collapsing to 0.565, so treat this
as a signal that needs occasional re-validation, not a permanent fact).

horizon_bars is NOT auto-derived here - the backtest used each real
pivot-ratchet trade's own bars_held (only knowable in hindsight), so for
live use the caller must supply how many bars they actually intend to
hold for. Checked clean of horizon-leakage (correlation with outcome
~0 - a longer race window doesn't mechanically favor either barrier,
unlike the MFE/MAE excursion work where the same feature WAS leaky).
"""
import pickle

import numpy as np

import backend  # module-level import (not `from backend import X`) so this
                 # stays safe if backend.py itself imports this module -
                 # attribute access happens at call time, after both
                 # modules have finished loading, avoiding a circular-import failure
from backtest_ou_zones import fit_ou_process
from backtest_levels import compute_atr

CALIBRATOR_PATH = 'target_probability_calibrator.pkl'
_calibrator = None

# simple placeholder mapping, not calendar-aware (doesn't account for
# overnight/weekend gaps in trading hours) - good enough until the
# options-Greeks-driven horizon (theta-adaptive) replaces this later
BAR_HOURS = {'1h': 1, '4h': 4, '1d': 24}


def _get_calibrator():
    global _calibrator
    if _calibrator is None:
        with open(CALIBRATOR_PATH, 'rb') as f:
            _calibrator = pickle.load(f)
    return _calibrator


def _simulate_two_barrier_race(x0, theta, mu, sigma, target_deviation, stop_deviation,
                                horizon, n_sims=2000, dt=1.0, rng=None):
    """Same math as backtest_target_selection_eq.simulate_two_barrier_race,
    Gaussian-innovation branch only (the version that survived calibration)."""
    if rng is None:
        rng = np.random.default_rng()
    X = np.full(n_sims, x0, dtype=float)
    hit_target = np.full(n_sims, horizon + 1)
    hit_stop = np.full(n_sims, horizon + 1)

    for step in range(1, horizon + 1):
        prev_X = X
        Z = rng.standard_normal(n_sims)
        X = X + theta * (mu - X) * dt + sigma * np.sqrt(dt) * Z

        stop_cross = ((prev_X - stop_deviation) * (X - stop_deviation) <= 0) & (hit_stop > horizon)
        hit_stop[stop_cross] = step
        target_cross = ((prev_X - target_deviation) * (X - target_deviation) <= 0) & (hit_target > horizon)
        hit_target[target_cross] = step

    return float(np.mean(hit_target < hit_stop))


def _sample_paths(x0, theta, mu, sigma, target_deviation, stop_deviation, horizon,
                   current_vwap, n_paths=40, rng=None):
    """Small, separate, cheap simulation purely for visualization - same
    dynamics as _simulate_two_barrier_race, just fewer paths and returns
    the full trajectories (converted back to absolute price) instead of
    only the race outcome. Does not affect the actual probability
    calculation at all - additive, for the degencap.uk animation."""
    if rng is None:
        rng = np.random.default_rng()
    X = np.full(n_paths, x0, dtype=float)
    paths = np.zeros((n_paths, horizon + 1))
    paths[:, 0] = x0
    hit_step = np.full(n_paths, -1)  # -1 = neither hit within horizon
    hit_what = np.array([''] * n_paths, dtype=object)
    for step in range(1, horizon + 1):
        prev_X = X
        Z = rng.standard_normal(n_paths)
        X = X + theta * (mu - X) + sigma * Z
        paths[:, step] = X
        stop_cross = ((prev_X - stop_deviation) * (X - stop_deviation) <= 0) & (hit_step < 0)
        target_cross = ((prev_X - target_deviation) * (X - target_deviation) <= 0) & (hit_step < 0)
        hit_step[stop_cross] = step
        hit_what[stop_cross] = 'stop'
        hit_step[target_cross] = step
        hit_what[target_cross] = 'target'
    return {
        'paths_price': (paths + current_vwap).round(4).tolist(),
        'outcome': hit_what.tolist(), 'hit_step': hit_step.tolist(),
    }


def predict_target_probability(highs, lows, closes, volumes, timestamps,
                                entry_price, stop_price, target_price,
                                horizon_hours, timeframe='1h', n_sims=2000, lookback=150,
                                include_sample_paths=False, n_sample_paths=40):
    """
    highs/lows/closes/volumes/timestamps: arrays for the LOOKBACK window
    immediately preceding entry (most recent bar last) - same convention
    as every other window-based function in this repo (backend.py's
    score_and_filter_levels_v2, build_market_state_snapshot_v3, etc.)
    entry_price/stop_price/target_price: absolute prices.
    horizon_hours: how many HOURS you intend to hold for - converted to
    bars internally via BAR_HOURS[timeframe] (not auto-derived from the
    market itself; this is still the caller's call, just expressed in a
    timeframe-independent unit instead of raw bar count. A placeholder
    until an options-Greeks-driven horizon replaces it).
    timeframe: '1h', '4h', or '1d' - must match the bar size of the
    highs/lows/closes/volumes arrays passed in.

    Returns {'raw_probability', 'calibrated_probability', 'theta', 'mu',
    'sigma', 'error'} - error is set (and probabilities None) if the
    window doesn't support a valid OU fit (e.g. too few bars, no
    mean-reversion detected - fit_ou_process's own degenerate-fit guards).
    """
    highs, lows, closes, volumes = map(np.asarray, (highs, lows, closes, volumes))
    if len(closes) < lookback:
        return {'error': f'need at least {lookback} bars, got {len(closes)}',
                'raw_probability': None, 'calibrated_probability': None}
    if timeframe not in BAR_HOURS:
        return {'error': f"timeframe must be one of {list(BAR_HOURS)}", 'raw_probability': None, 'calibrated_probability': None}
    horizon_bars = max(1, round(horizon_hours / BAR_HOURS[timeframe]))
    win_h, win_l, win_c, win_v = highs[-lookback:], lows[-lookback:], closes[-lookback:], volumes[-lookback:]
    win_dt = np.asarray(timestamps)[-lookback:]

    atr = compute_atr(win_h, win_l, win_c)
    if atr <= 0:
        return {'error': 'degenerate ATR (zero range)', 'raw_probability': None, 'calibrated_probability': None}

    vwap_result = backend.calculate_vwap(win_h, win_l, win_c, win_v, timestamps=win_dt)
    if vwap_result is None:
        return {'error': 'VWAP calculation failed', 'raw_probability': None, 'calibrated_probability': None}
    vwap_series = vwap_result['vwap_series']
    current_vwap = vwap_series[-1]
    deviation = win_c - vwap_series

    fit = fit_ou_process(deviation)
    if fit is None:
        return {'error': 'no valid mean-reverting OU fit for this window (degenerate or non-mean-reverting)',
                'raw_probability': None, 'calibrated_probability': None}
    theta, mu, stationary_std = fit
    sigma = stationary_std * np.sqrt(2 * theta)

    x0 = entry_price - current_vwap
    stop_dev = stop_price - current_vwap
    target_dev = target_price - current_vwap

    p_raw = _simulate_two_barrier_race(x0, theta, mu, sigma, target_dev, stop_dev,
                                        horizon_bars, n_sims=n_sims)
    p_cal = float(_get_calibrator().predict([p_raw])[0])

    result = {
        'raw_probability': round(p_raw, 4), 'calibrated_probability': round(p_cal, 4),
        'theta': round(float(theta), 4), 'mu_vwap_relative': round(float(mu), 4),
        'sigma': round(float(sigma), 4), 'atr': round(float(atr), 4),
        'horizon_bars_used': horizon_bars, 'error': None,
    }
    if include_sample_paths:
        result['sample_paths'] = _sample_paths(x0, theta, mu, sigma, target_dev, stop_dev,
                                                 horizon_bars, current_vwap, n_paths=n_sample_paths)
    return result
