"""
Adds a sector concentration cap on top of inverse-vol weighting, directly
targeting the diagnosed root cause: the fundamentals+vector filter
systematically over-selects growth/tech names, and the entire -45.4%
drawdown was one correlated style-factor crash (2022 growth selloff), not
diversified idiosyncratic risk. No sector may exceed CAP_FRACTION of a
given month's basket; when a sector is over-represented, keep the names
with the strongest vector quality read (z-score) and drop the rest -
this doesn't shrink the basket to save the cap, it reallocates within it.
"""
import json
import math

import numpy as np
import pandas as pd
import yfinance as yf

from backtest_pivot_ratchet import run_state_machine, regime_strip_agreement, ATR_LEN, SLOW_LEN, PIVOT_LEN
from vector_screener_sp500_sectors import batch_download_cached, load_sectors
from vector_fundamentals_backtest import fundamentals_asof, load_facts_cache, BACKTEST_START

RECENT_FLIP_WINDOW = 20
MIN_FLIPS_FOR_QUALITY = 5
HORIZON = 20
CAP_FRACTION = 0.30  # the best-model setting (0.20 was tested and found worse)


def build_state(df):
    close = df['close'].values
    regimes, levels, atr, flips = run_state_machine(df)
    warmup = max(ATR_LEN, SLOW_LEN, PIVOT_LEN * 2) + 5
    return {'df': df, 'close': close, 'atr': atr, 'regimes': regimes, 'flips': flips, 'warmup': warmup}


def eligibility_z(stock, idx):
    """Returns quality z-score if eligible, else None."""
    regimes, close, flips, warmup = stock['regimes'], stock['close'], stock['flips'], stock['warmup']
    if idx < warmup + HORIZON + 10:
        return None
    recent = [f for f in flips if idx - RECENT_FLIP_WINDOW <= f[0] <= idx]
    if not recent:
        return None
    last_bar, last_dir = recent[-1]
    if last_dir != 1:
        return None
    flips_so_far = [f for f in flips if f[0] <= idx]
    if len(flips_so_far) < MIN_FLIPS_FOR_QUALITY:
        return None
    strip = regime_strip_agreement(close[:idx + 1], regimes[:idx + 1], [HORIZON], warmup)
    z = float(strip.iloc[0]['z'])
    if z <= 0:
        return None
    return z


def apply_sector_cap(candidates, sectors):
    """candidates: list of dicts with ticker, ret, vol_pct, z. Caps each
    sector's count at CAP_FRACTION of the total, keeping highest-z names
    within an over-represented sector."""
    if not candidates:
        return []
    n_total = len(candidates)
    max_per_sector = max(1, math.ceil(CAP_FRACTION * n_total))

    by_sector = {}
    for c in candidates:
        sec = sectors.get(c['ticker'], '?')
        by_sector.setdefault(sec, []).append(c)

    kept = []
    for sec, group in by_sector.items():
        group.sort(key=lambda c: -c['z'])
        kept.extend(group[:max_per_sector])
    return kept


def stats_line(returns, label):
    r = np.array(returns)
    total = float(np.prod(1 + r) - 1)
    ann = float((1 + total) ** (12 / len(r)) - 1) if len(r) else np.nan
    sharpe = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else np.nan
    equity = np.cumprod(1 + r)
    dd = float((equity / np.maximum.accumulate(equity) - 1).min())
    print(f"  {label:<40} total={total*100:+8.1f}%  ann={ann*100:+6.1f}%  sharpe={sharpe:+.2f}  max_dd={dd*100:6.1f}%")


def invvol_return(picks):
    valid = [c for c in picks if not np.isnan(c['vol_pct']) and c['vol_pct'] > 0]
    if not valid:
        return 0.0
    inv_vol = np.array([1.0 / c['vol_pct'] for c in valid])
    weights = inv_vol / inv_vol.sum()
    rets = np.array([c['ret'] for c in valid])
    return float(np.sum(weights * rets))


def main():
    facts_by_ticker = load_facts_cache()
    sectors = load_sectors()
    with open('sp500_pit_membership.json') as f:
        pit_membership_raw = json.load(f)
    with open('sp500_pit_all_tickers.json') as f:
        all_tickers = json.load(f)
    all_tickers_yf = [t.replace('.', '-') for t in all_tickers]
    raw = batch_download_cached(all_tickers_yf)

    spy_raw = yf.download('SPY', period='max', interval='1d', auto_adjust=True, progress=False)
    spy_raw = spy_raw.reset_index()
    spy_raw.columns = [str(c[0] if isinstance(c, tuple) else c).lower() for c in spy_raw.columns]
    spy_raw = spy_raw.rename(columns={'date': 'datetime'})
    spy_raw['datetime'] = pd.to_datetime(spy_raw['datetime']).dt.tz_localize(None)
    spy = spy_raw[spy_raw['datetime'] >= BACKTEST_START].reset_index(drop=True)
    rebalance_dates = spy.groupby(spy['datetime'].dt.to_period('M'))['datetime'].min().tolist()
    spy_close, spy_dates = spy['close'].values, spy['datetime'].values
    spy_monthly = []
    for i in range(len(rebalance_dates) - 1):
        idx = np.searchsorted(spy_dates, np.datetime64(rebalance_dates[i]), side='right') - 1
        idx_next = np.searchsorted(spy_dates, np.datetime64(rebalance_dates[i + 1]), side='right') - 1
        spy_monthly.append(spy_close[idx_next] / spy_close[idx] - 1)

    stocks = {}
    for t, df in raw.items():
        if len(df) < 500:
            continue
        try:
            stocks[t] = build_state(df)
        except Exception:
            continue
    print(f"Built pivot-ratchet state for {len(stocks)} names\n")

    pit_membership = {mk: set(t.replace('.', '-') for t in ts) for mk, ts in pit_membership_raw.items()}

    fund_cache = {}
    r_invvol, r_capped = [], []
    n_uncapped, n_capped = [], []
    for i in range(len(rebalance_dates) - 1):
        d, d_next = rebalance_dates[i], rebalance_dates[i + 1]
        as_of_str = d.strftime('%Y-%m-%d')
        members = pit_membership.get(d.strftime('%Y-%m'), set())
        candidates = []
        for t, s in stocks.items():
            if t not in members:
                continue
            dates = s['df']['datetime'].values
            idx = np.searchsorted(dates, np.datetime64(d), side='right') - 1
            if idx < 0 or idx >= len(dates):
                continue
            idx_next = np.searchsorted(dates, np.datetime64(d_next), side='right') - 1
            if idx_next < 0 or idx_next >= len(dates) or idx_next <= idx:
                continue
            z = eligibility_z(s, idx)
            if z is None:
                continue
            gaap = facts_by_ticker.get(t)
            if gaap is None:
                continue
            key = (t, as_of_str)
            if key not in fund_cache:
                fund_cache[key] = fundamentals_asof(gaap, as_of_str)
            result = fund_cache[key]
            if result is None or not result['strict_pass']:
                continue
            ret = s['close'][idx_next] / s['close'][idx] - 1
            vol_pct = s['atr'][idx] / s['close'][idx] if not np.isnan(s['atr'][idx]) else np.nan
            candidates.append({'ticker': t, 'ret': ret, 'vol_pct': vol_pct, 'z': z})

        n_uncapped.append(len(candidates))
        r_invvol.append(invvol_return(candidates))

        capped = apply_sector_cap(candidates, sectors)
        n_capped.append(len(capped))
        r_capped.append(invvol_return(capped))

    print(f"{'='*100}\nSECTOR-CAPPED INVERSE-VOL BASKET (cap={CAP_FRACTION*100:.0f}% per sector)\n{'='*100}")
    stats_line(spy_monthly, 'SPY buy & hold')
    stats_line(r_invvol, 'Fundamentals + inverse-vol (no sector cap)')
    stats_line(r_capped, f'Fundamentals + inverse-vol + {CAP_FRACTION*100:.0f}% sector cap')
    print(f"\n  avg picks/month - uncapped: {np.mean(n_uncapped):.2f}   capped: {np.mean(n_capped):.2f}")


if __name__ == '__main__':
    main()
