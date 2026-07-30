"""
Reference-point levels: prices real market participants actually watch and
place orders around, as opposed to statistically-discovered clusters of
past price (HDBSCAN/GMM/KDE/etc, all of which failed to beat random in
backtest_levels.py). No training, no clustering - each is a simple
deterministic lookup against calendar history.

Every function takes the FULL history up to (not including) bar index t,
since e.g. a prior month's high/low isn't visible in a 150-bar window.
"""
import numpy as np
import pandas as pd


def prior_day_levels(dates, highs, lows, closes, t):
    current_date = dates[t - 1]
    mask_prior = dates[:t] < current_date
    if not mask_prior.any():
        return []
    prior_dates = dates[:t][mask_prior]
    last_prior_date = prior_dates.max()
    day_mask = dates[:t] == last_prior_date
    day_highs, day_lows, day_closes = highs[:t][day_mask], lows[:t][day_mask], closes[:t][day_mask]
    if len(day_closes) == 0:
        return []
    return [
        {'price': float(day_highs.max()), 'category': 'Reference', 'type': 'Prior Day High'},
        {'price': float(day_lows.min()), 'category': 'Reference', 'type': 'Prior Day Low'},
        {'price': float(day_closes[-1]), 'category': 'Reference', 'type': 'Prior Day Close'},
    ]


def _prior_period_levels(period_keys, highs, lows, t, label):
    current_key = period_keys[t - 1]
    mask_prior = period_keys[:t] < current_key
    if not mask_prior.any():
        return []
    prior_keys = period_keys[:t][mask_prior]
    last_prior_key = prior_keys.max()
    period_mask = period_keys[:t] == last_prior_key
    period_highs, period_lows = highs[:t][period_mask], lows[:t][period_mask]
    if len(period_highs) == 0:
        return []
    return [
        {'price': float(period_highs.max()), 'category': 'Reference', 'type': f'Prior {label} High'},
        {'price': float(period_lows.min()), 'category': 'Reference', 'type': f'Prior {label} Low'},
    ]


def prior_week_levels(week_keys, highs, lows, t):
    return _prior_period_levels(week_keys, highs, lows, t, 'Week')


def prior_month_levels(month_keys, highs, lows, t):
    return _prior_period_levels(month_keys, highs, lows, t, 'Month')


def session_open_level(dates, opens, t):
    current_date = dates[t - 1]
    day_mask = dates[:t] == current_date
    day_opens = opens[:t][day_mask]
    if len(day_opens) == 0:
        return []
    return [{'price': float(day_opens[0]), 'category': 'Reference', 'type': 'Session Open'}]


def round_number_levels(current_price, increments=(25, 50, 100, 250, 500, 1000), pct_range=0.03):
    if current_price <= 0:
        return []
    lo, hi = current_price * (1 - pct_range), current_price * (1 + pct_range)
    levels, seen = [], set()
    for inc in increments:
        start = int(np.floor(lo / inc)) * inc
        end = int(np.ceil(hi / inc)) * inc
        for p in range(start, end + inc, inc):
            if lo <= p <= hi and p not in seen:
                seen.add(p)
                levels.append({'price': float(p), 'category': 'Reference', 'type': f'Round-{inc}'})
    return levels


def vwap_levels(backend, highs, lows, closes, volumes, timestamps, n_sigma_bands=(1, 2, 3)):
    result = backend.calculate_vwap(highs, lows, closes, volumes, timestamps=timestamps, n_sigma_bands=n_sigma_bands)
    if result is None:
        return []
    levels = [{'price': result['vwap'], 'category': 'Reference', 'type': 'VWAP'}]
    for s, b in result['bands'].items():
        levels.append({'price': b['upper'], 'category': 'Reference', 'type': f'VWAP+{s}sigma'})
        levels.append({'price': b['lower'], 'category': 'Reference', 'type': f'VWAP-{s}sigma'})
    return levels
