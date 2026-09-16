"""
Options-state and intraday-flow context panel for QQQ (translated to other
underlyings via price ratio at time of use - the same validated NQ/QQQ
ratio approach as backtest_options_zone_confluence.py, extremely stable at
41.10 +/- 0.18, <0.5% CV). Two independent data-generating processes, per
spec:

  STATE panel: ATM IV regime, term structure, 25-delta skew, expected
  move - refreshed every 1-5 min from the options chain (quotes/greeks).

  FLOW panel: net premium/volume/traded-delta/traded-gamma from
  timestamped trade prints - a DIFFERENT signal (current trading
  activity), computed independently of the state panel.

Deliberately does NOT build strike-concentration or gamma-exposure-by-OI.
London Strategic Edge's API has no open-interest field anywhere - checked
directly against both /options/chain and /options/flow (2026-08-14):
fields are ticker/strike/expiry/contract_type/last_price/volume_today/
premium_today/underlying_price/dte/iv/delta/gamma/theta/vega/rho/
last_trade_at/updated_at. No oi/open_interest field exists. Faking strike
concentration or gamma exposure from today's VOLUME would misrepresent
trading activity as accumulated positioning. Those two modules report
"not_available", never an estimated number.

DATA ADAPTATION FROM SPEC: the API also has no bid/ask (same field list
above) - no mid-price, quoted spread, or conventional quote-quality
metric is calculable. Quality fields are named accordingly
(last_trade_recency_seconds not quote_age_seconds, iv_surface_coverage
not quote_coverage, trade_derived_expected_move not
mid_straddle_expected_move) and every state panel carries
last_trade_based=True so nothing downstream can mistake this for a
quote-derived measure.

FLOW CAP: /options/flow returns at most 5000 prints per call. A window
that hits exactly 5000 is CENSORED - premium/volume/delta/gamma/z-score
in that window are lower bounds, not complete counts, and flow-shock
labels are suppressed rather than risk a false read exactly when a real
shock would be most likely to hit the cap.
"""
import os
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

import backend

load_dotenv()

TRADING_DAYS_PER_YEAR = 252

BASE_URL = 'https://api.londonstrategicedge.com/vault'
API_KEY = os.getenv('LSE_API_KEY')
ET = ZoneInfo('America/New_York')

BUCKETS = {'front': (3, 10), 'near': (14, 30), 'back': (31, 60)}

IV_HARD_MIN, IV_HARD_MAX = 0.02, 2.00
IV_MAD_THRESHOLD = 5.0

STATE_DELTA_BAND = (0.15, 0.85)
ATM_DELTA_BAND = (0.40, 0.60)
STATE_MIN_VOLUME = 5
MAX_LAST_TRADE_RECENCY_MIN = 45

FLOW_PRINT_CAP = 5000


def market_session_status(now_utc=None):
    """
    Coarse regular-hours check (9:30-16:00 ET, Mon-Fri) - no exchange
    holiday calendar, a known simplification flagged here rather than
    silently assumed correct. Returns 'regular_hours' or 'market_closed'.
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    now_et = now_utc.astimezone(ET)
    if now_et.weekday() >= 5:  # Sat/Sun
        return 'market_closed'
    open_t = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return 'regular_hours' if open_t <= now_et <= close_t else 'market_closed'


def _get(path, params):
    resp = requests.get(f'{BASE_URL}/{path}', headers={'x-api-key': API_KEY}, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def fetch_chain_bucket(underlying, min_dte, max_dte):
    rows = _get('options/chain', {'underlying': underlying, 'min_dte': min_dte, 'max_dte': max_dte})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df['last_trade_at'] = pd.to_datetime(df['last_trade_at'])
    if df['last_trade_at'].dt.tz is None:
        df['last_trade_at'] = df['last_trade_at'].dt.tz_localize('UTC')
    df['abs_delta'] = df['delta'].abs()
    return df


def _mad_outlier_mask(s):
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return pd.Series(True, index=s.index)
    robust_z = 0.6745 * (s - med).abs() / mad
    return robust_z <= IV_MAD_THRESHOLD


def clean_contracts(df, delta_band, min_volume=STATE_MIN_VOLUME, now=None):
    """Liquidity + plausibility screen. Returns (clean_df, rejection_counts dict)."""
    if df.empty:
        return df, {'input': 0}
    now = now or datetime.now(timezone.utc)
    counts = {'input': len(df)}

    d = df.copy()
    d = d[d['last_price'] > 0]
    counts['after_price_positive'] = len(d)

    d = d[d['volume_today'] >= min_volume]
    counts['after_volume_floor'] = len(d)

    recency_sec = (now - d['last_trade_at']).dt.total_seconds().abs()
    d = d[recency_sec <= MAX_LAST_TRADE_RECENCY_MIN * 60]
    counts['after_last_trade_recency'] = len(d)

    d = d[(d['iv'] >= IV_HARD_MIN) & (d['iv'] <= IV_HARD_MAX)]
    counts['after_iv_hard_range'] = len(d)

    d = d[(d['abs_delta'] >= delta_band[0]) & (d['abs_delta'] <= delta_band[1])]
    counts['after_delta_band'] = len(d)

    if len(d):
        keep_mask = d.groupby(['dte', 'contract_type'])['iv'].transform(_mad_outlier_mask)
        d = d[keep_mask]
    counts['after_iv_outlier_reject'] = len(d)

    return d, counts


def estimate_atm_iv(chain_df, spot, now=None):
    clean, _ = clean_contracts(chain_df, ATM_DELTA_BAND, now=now)
    if clean.empty:
        return None, None
    clean = clean.copy()
    clean['dist_to_spot'] = (clean['strike'] - spot).abs()
    nearest_strike = clean.loc[clean['dist_to_spot'].idxmin(), 'strike']
    band = clean[(clean['strike'] - nearest_strike).abs() <= max(1.0, spot * 0.005)]
    if band.empty:
        band = clean.nsmallest(4, 'dist_to_spot')
    atm_iv = float(band['iv'].mean())
    dte_used = int(band['dte'].mode().iloc[0])
    return atm_iv, dte_used


def estimate_25d_skew(chain_df, atm_iv, now=None):
    clean, _ = clean_contracts(chain_df, STATE_DELTA_BAND, now=now)
    if clean.empty or atm_iv is None:
        return None, None

    put_skew = None
    puts = clean[clean['contract_type'] == 'put'].sort_values('abs_delta')
    if len(puts) >= 2 and puts['abs_delta'].min() <= 0.25 <= puts['abs_delta'].max():
        iv_25d_put = float(np.interp(0.25, puts['abs_delta'], puts['iv']))
        put_skew = iv_25d_put - atm_iv

    call_skew = None
    calls = clean[clean['contract_type'] == 'call'].sort_values('abs_delta')
    if len(calls) >= 2 and calls['abs_delta'].min() <= 0.25 <= calls['abs_delta'].max():
        iv_25d_call = float(np.interp(0.25, calls['abs_delta'], calls['iv']))
        call_skew = iv_25d_call - atm_iv

    return put_skew, call_skew


def estimate_expected_move(chain_df, spot, now=None):
    """trade_derived_expected_move: ATM straddle built from last_price on
    each leg (NOT a mid-price straddle - no bid/ask exists). Explicitly a
    last-trade-based approximation, can be stale or a nonrepresentative
    execution - see last_trade_based flag in the panel output."""
    clean, _ = clean_contracts(chain_df, (0.30, 0.70), min_volume=1, now=now)
    if clean.empty:
        return None, None, None
    clean = clean.copy()
    clean['dist_to_spot'] = (clean['strike'] - spot).abs()
    strikes_with_both = (clean.groupby('strike')['contract_type'].nunique() == 2)
    valid_strikes = strikes_with_both[strikes_with_both].index
    if len(valid_strikes) == 0:
        return None, None, None
    nearest = clean[clean['strike'].isin(valid_strikes)].nsmallest(1, 'dist_to_spot')['strike'].iloc[0]
    leg = clean[clean['strike'] == nearest]
    call_px = leg[leg['contract_type'] == 'call']['last_price'].iloc[0]
    put_px = leg[leg['contract_type'] == 'put']['last_price'].iloc[0]
    em_pts = float(call_px + put_px)
    em_pct = em_pts / spot
    dte_used = int(leg['dte'].iloc[0])
    return em_pts, em_pct, dte_used


def estimate_gjr_forecast_annualized(underlying, horizon_days):
    """
    Reuses backend.py's already-cached-elsewhere GJR-GARCH(1,1,1) fit
    (same function the v3 state snapshot uses) on this underlying's own
    daily closes, annualized (*sqrt(252)) so it's directly comparable to
    IV, which options markets always quote annualized regardless of the
    specific DTE it was priced from. Tags horizon_days with whichever
    expiry bucket's ATM IV this gets compared against - the number itself
    is annualized, the tag just records which comparison it's for.
    Returns None if the fit fails (too little data, non-convergence).
    """
    try:
        hist = yf.Ticker(underlying).history(period='1y', interval='1d')
        closes = hist['Close'].values
        if len(closes) < 60:
            return None
        returns_pct = np.diff(np.log(closes)) * 100
        daily_vol_pct = backend.fit_gjr_garch_vol_forecast_pct(returns_pct)
        if daily_vol_pct is None:
            return None
        daily_vol = daily_vol_pct / 100.0
        return {
            'model': 'gjr_garch', 'horizon_days': horizon_days,
            'annualized_vol': float(daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)),
            # same-horizon expected move as the options straddle (sqrt-time scaled from
            # the 1-day forecast, NOT the annualized figure) - what range_agreement compares against
            'horizon_expected_move_pct': float(daily_vol * np.sqrt(horizon_days)),
        }
    except Exception:
        return None


def build_state_panel(underlying='QQQ', last_valid_snapshot=None):
    """
    last_valid_snapshot: optional dict of the most recent 'ok' snapshot
    (from persistence) - when the market is closed, this is returned
    as explicitly-labeled stale reference data (panel_status=
    'market_closed', not 'ok') instead of silently returning nothing.
    """
    now = datetime.now(timezone.utc)
    session = market_session_status(now)

    if session == 'market_closed':
        out = {'schema_version': 'options_context_v1', 'as_of_vendor': None,
               'collected_at_utc': now.isoformat(), 'underlying': underlying,
               'market_session_status': 'market_closed',
               'panel_status': 'market_closed'}
        if last_valid_snapshot:
            out['reference_snapshot'] = last_valid_snapshot
            out['note'] = 'market closed - serving last valid session snapshot as stale reference, not live'
        else:
            out['note'] = 'market closed - no prior valid snapshot available to reference'
        return out

    bucket_data = {}
    for name, (lo, hi) in BUCKETS.items():
        bucket_data[name] = fetch_chain_bucket(underlying, lo, hi)
        time.sleep(0.2)

    front, near = bucket_data['front'], bucket_data['near']
    base = {'schema_version': 'options_context_v1', 'collected_at_utc': now.isoformat(),
            'underlying': underlying, 'market_session_status': session, 'last_trade_based': True}

    if front.empty:
        return {**base, 'panel_status': 'insufficient_liquid_options_data', 'as_of_vendor': None}

    spot = float(front['underlying_price'].iloc[0])
    as_of_vendor = front['last_trade_at'].max().isoformat()

    front_atm_iv, front_dte = estimate_atm_iv(front, spot, now=now)
    near_atm_iv, near_dte = estimate_atm_iv(near, spot, now=now) if not near.empty else (None, None)

    if front_atm_iv is None:
        return {**base, 'panel_status': 'insufficient_liquid_options_data', 'as_of_vendor': as_of_vendor, 'spot': spot}

    put_skew, call_skew = estimate_25d_skew(front, front_atm_iv, now=now)
    em_pts, em_pct, em_dte = estimate_expected_move(front, spot, now=now)

    clean_front, counts_front = clean_contracts(front, STATE_DELTA_BAND, now=now)
    valid_contracts = counts_front.get('after_iv_outlier_reject', 0)
    raw_contracts = counts_front.get('input', 0)
    recency_sec = (now - clean_front['last_trade_at']).dt.total_seconds().abs() if len(clean_front) else pd.Series([np.nan])

    term_state = None
    front_minus_near = None
    if front_atm_iv is not None and near_atm_iv is not None:
        front_minus_near = front_atm_iv - near_atm_iv
        # front IV materially above near = event/stress pricing; below = normal carry
        term_state = ('front_loaded_risk' if front_minus_near > 0.01
                       else ('flat' if abs(front_minus_near) <= 0.01 else 'normal_contango'))

    panel_status = 'ok' if valid_contracts >= 10 else ('degraded' if valid_contracts >= 3 else 'insufficient_liquid_options_data')
    iv_surface_coverage = float(valid_contracts / raw_contracts) if raw_contracts else 0.0

    realized_vol_forecast = estimate_gjr_forecast_annualized(underlying, front_dte)
    iv_realized_vol_gap = None
    range_agreement = None
    if realized_vol_forecast is not None:
        gjr_ann = realized_vol_forecast['annualized_vol']
        iv_minus_gjr = front_atm_iv - gjr_ann
        iv_realized_vol_gap = {
            'atm_iv': front_atm_iv, 'gjr_forecast': gjr_ann, 'iv_minus_gjr': iv_minus_gjr,
            'relative_premium': float(iv_minus_gjr / gjr_ann) if gjr_ann else None,
            'state': 'iv_above_realized_vol_forecast' if iv_minus_gjr > 0 else 'iv_below_realized_vol_forecast',
        }
        if em_pct is not None:
            gjr_em_pct = realized_vol_forecast['horizon_expected_move_pct']
            disagreement = (em_pct - gjr_em_pct) / gjr_em_pct if gjr_em_pct else None
            range_agreement = {
                'implied_expected_move_pct': em_pct, 'gjr_expected_move_pct': gjr_em_pct,
                'disagreement': disagreement,
                'label': ('aligned' if disagreement is None or abs(disagreement) <= 0.15
                          else ('options_wider' if disagreement > 0 else 'options_narrower')),
            }

    return {
        **base,
        'as_of_vendor': as_of_vendor,
        'panel_status': panel_status,
        'spot': spot,
        'quality': {
            'valid_contract_count': int(valid_contracts),
            'raw_contract_count': int(raw_contracts),
            'iv_surface_coverage': iv_surface_coverage,
            'last_trade_recency_seconds': float(recency_sec.max()) if len(recency_sec.dropna()) else None,
            'rejection_counts': counts_front,
            'data_gaps': ['no_open_interest_field', 'no_bid_ask_fields'],
        },
        'iv_regime': {'atm_iv': front_atm_iv, 'atm_dte': front_dte},
        'term_structure': {
            'front_dte': front_dte, 'near_dte': near_dte,
            'front_atm_iv': front_atm_iv, 'near_atm_iv': near_atm_iv,
            'front_minus_near': front_minus_near, 'state': term_state,
        },
        'skew': {'put_25d_minus_atm': put_skew, 'call_25d_minus_atm': call_skew},
        'expected_move': {
            'expiry_dte': em_dte, 'pct': em_pct, 'points': em_pts,
            'method': 'trade_derived_expected_move',
        },
        'realized_vol_forecast': realized_vol_forecast,
        'iv_realized_vol_gap': iv_realized_vol_gap,
        'range_agreement': range_agreement,
        'strike_concentration': {'status': 'not_available', 'reason': 'no_open_interest_field'},
        'gamma_exposure': {'status': 'not_available', 'reason': 'no_open_interest_field'},
    }


def fetch_flow(underlying, start=None, end=None, min_premium=0):
    params = {'underlying': underlying, 'min_premium': min_premium}
    if start:
        params['start'] = start
    if end:
        params['end'] = end
    rows = _get('options/flow', params)
    df = pd.DataFrame(rows)
    n_returned = len(df)
    if df.empty:
        return df, n_returned
    df['ts'] = pd.to_datetime(df['ts'])
    df['abs_delta'] = df['delta'].abs()
    df['contracts'] = df['volume']
    df['traded_delta'] = df['contracts'] * 100 * df['delta']
    df['traded_gamma'] = df['contracts'] * 100 * df['gamma']
    return df, n_returned


def build_flow_panel(underlying='QQQ', window_minutes=15, baseline_days=5):
    now = datetime.now(timezone.utc)
    session = market_session_status(now)
    base = {'schema_version': 'options_flow_v1', 'collected_at_utc': now.isoformat(),
            'underlying': underlying, 'market_session_status': session}

    if session == 'market_closed':
        return {**base, 'panel_status': 'market_closed'}

    window_start = now - timedelta(minutes=window_minutes)
    recent, n_returned = fetch_flow(underlying, start=window_start.strftime('%Y-%m-%dT%H:%M:%S'), min_premium=0)
    flow_censored = n_returned >= FLOW_PRINT_CAP

    if recent.empty:
        return {**base, 'panel_status': 'insufficient_liquid_options_data'}

    call_premium = float(recent.loc[recent['contract_type'] == 'call', 'premium'].sum())
    put_premium = float(recent.loc[recent['contract_type'] == 'put', 'premium'].sum())
    call_put_premium_ratio = float(call_premium / put_premium) if put_premium > 0 else None

    call_vol = int(recent.loc[recent['contract_type'] == 'call', 'volume'].sum())
    put_vol = int(recent.loc[recent['contract_type'] == 'put', 'volume'].sum())
    call_put_vol_ratio = float(call_vol / put_vol) if put_vol > 0 else None

    spot = float(recent['underlying_price'].iloc[-1])
    recent = recent.copy()
    recent['moneyness_dist_pct'] = (recent['strike'] - spot).abs() / spot
    near_spot_share = float((recent['moneyness_dist_pct'] <= 0.02).mean())
    front_share = float((recent['dte'] <= 10).mean())
    traded_delta = float(recent['traded_delta'].sum())
    traded_gamma = float(recent['traded_gamma'].sum())
    total_volume = int(recent['volume'].sum())

    baseline_totals = []
    for d in range(1, baseline_days + 1):
        day_start = (window_start - timedelta(days=d)).strftime('%Y-%m-%dT%H:%M:%S')
        day_end = (now - timedelta(days=d)).strftime('%Y-%m-%dT%H:%M:%S')
        try:
            hist, hist_n = fetch_flow(underlying, start=day_start, end=day_end, min_premium=0)
            if not hist.empty:
                baseline_totals.append(int(hist['volume'].sum()))
        except Exception:
            pass
        time.sleep(0.2)

    volume_zscore = None
    if len(baseline_totals) >= 3:
        mu, sigma = float(np.mean(baseline_totals)), float(np.std(baseline_totals))
        if sigma > 0:
            volume_zscore = (total_volume - mu) / sigma

    label = 'suppressed_censored_flow' if flow_censored else 'normal'
    if not flow_censored and volume_zscore is not None and volume_zscore >= 2.0:
        label = 'unusual_call_dominant_flow' if (call_put_premium_ratio or 1) > 1.3 else (
            'unusual_put_dominant_flow' if (call_put_premium_ratio or 1) < 0.77 else 'unusual_flow')

    return {
        **base,
        'panel_status': 'ok' if total_volume >= 20 else 'degraded',
        'window_minutes': window_minutes,
        'flow_records_returned': n_returned,
        'flow_cap': FLOW_PRINT_CAP,
        'flow_censored': flow_censored,
        'call_premium': call_premium,
        'put_premium': put_premium,
        'call_put_premium_ratio': call_put_premium_ratio,
        'call_put_volume_ratio': call_put_vol_ratio,
        'total_volume': total_volume,
        'volume_zscore_vs_same_time_baseline': volume_zscore,
        'baseline_days_used': len(baseline_totals),
        'near_spot_flow_share': near_spot_share,
        'front_expiry_flow_share': front_share,
        'traded_delta': traded_delta,
        'traded_gamma': traded_gamma,
        'label': label,
        'data_gaps': ['no_open_interest_field - this is CURRENT TRADING ACTIVITY, not positioning'],
    }


if __name__ == '__main__':
    import json
    print("=== STATE PANEL ===")
    print(json.dumps(build_state_panel('QQQ'), indent=2, default=str))
    print("\n=== FLOW PANEL ===")
    print(json.dumps(build_flow_panel('QQQ', window_minutes=15, baseline_days=5), indent=2, default=str))
