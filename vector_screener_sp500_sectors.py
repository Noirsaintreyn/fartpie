"""
Sector-concentration diagnostic on the real S&P 500 result: is the
+375.1%/-26.4%dd result already reasonably spread across GICS sectors, or
secretly concentrated (e.g. still mostly tech)? Answers whether the
proposed sector-rotation layer is even needed before building it.

Caches each ticker's price data to parquet on first run (sp500_cache/) so
re-running this diagnostic, or any follow-up test, doesn't re-download
499 tickers again.
"""
import json
import os

import numpy as np
import pandas as pd
import yfinance as yf

from backtest_pivot_ratchet import run_state_machine, regime_strip_agreement, ATR_LEN, SLOW_LEN, PIVOT_LEN

BACKTEST_START = '2018-01-01'
RECENT_FLIP_WINDOW = 20
MIN_FLIPS_FOR_QUALITY = 5
HORIZON = 20
CACHE_DIR = 'sp500_cache_raw2'  # dividend-UNADJUSTED OHLC (matches TradingView's display
                                # exactly - AAPL week of 2019-01-28: $41.63 close, exact match)
                                # PLUS an 'adjclose' column (retained from yfinance's own
                                # 'Adj Close', auto_adjust=False still returns it) - use adjclose
                                # for RETURN computation (it captures dividend income), and
                                # open/high/low/close for flip DETECTION (matches the real
                                # chart's level-crossing behavior). Conflating the two
                                # understates returns for every dividend-paying stock. Old
                                # raw-only cache (no adjclose) preserved at sp500_cache_raw/.


def load_tickers():
    with open('sp500_tickers.json') as f:
        raw = json.load(f)
    return [t.replace('.', '-') for t in raw]


def load_sectors():
    with open('sp500_sectors.json') as f:
        raw = json.load(f)
    return {k.replace('.', '-'): v for k, v in raw.items()}


def batch_download_cached(tickers):
    os.makedirs(CACHE_DIR, exist_ok=True)
    all_data = {}
    to_fetch = []
    for t in tickers:
        path = f"{CACHE_DIR}/{t}.parquet"
        if os.path.exists(path):
            try:
                all_data[t] = pd.read_parquet(path)
            except Exception:
                to_fetch.append(t)
        else:
            to_fetch.append(t)
    print(f"Loaded {len(all_data)} from cache, fetching {len(to_fetch)} fresh...")

    batch_size = 50
    for i in range(0, len(to_fetch), batch_size):
        batch = to_fetch[i:i + batch_size]
        data = yf.download(batch, period='max', interval='1d', group_by='ticker',
                            auto_adjust=False, threads=True, progress=False)
        for t in batch:
            try:
                df = data[t].dropna(how='all') if isinstance(data.columns, pd.MultiIndex) else data
                df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
                if df.empty:
                    continue
                df = df.reset_index()
                df.columns = [str(c).lower() for c in df.columns]
                date_col = 'date' if 'date' in df.columns else 'datetime'
                df = df.rename(columns={date_col: 'datetime'})
                df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
                df = df.rename(columns={'adj close': 'adjclose'})
                df = df[['datetime', 'open', 'high', 'low', 'close', 'adjclose']]
                df.to_parquet(f"{CACHE_DIR}/{t}.parquet", index=False)
                all_data[t] = df
            except Exception:
                continue
        print(f"  fetched batch {i//batch_size+1}/{(len(to_fetch)-1)//batch_size+1}, {len(all_data)} total", flush=True)
    return all_data


def build_state(df):
    close = df['close'].values
    regimes, levels, atr, flips = run_state_machine(df)
    warmup = max(ATR_LEN, SLOW_LEN, PIVOT_LEN * 2) + 5
    return {'df': df, 'close': close, 'regimes': regimes, 'flips': flips, 'warmup': warmup}


def eligibility(stock, idx):
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
    if float(strip.iloc[0]['z']) <= 0:
        return None
    return True


def main():
    tickers = load_tickers()
    sectors = load_sectors()
    raw = batch_download_cached(tickers)
    print(f"\nLoaded {len(raw)}/{len(tickers)} S&P 500 names with usable data")

    spy_raw = yf.download('SPY', period='max', interval='1d', auto_adjust=True, progress=False)
    spy_raw = spy_raw.reset_index()
    spy_raw.columns = [str(c[0] if isinstance(c, tuple) else c).lower() for c in spy_raw.columns]
    spy_raw = spy_raw.rename(columns={'date': 'datetime'})
    spy_raw['datetime'] = pd.to_datetime(spy_raw['datetime']).dt.tz_localize(None)
    spy = spy_raw[spy_raw['datetime'] >= BACKTEST_START].reset_index(drop=True)
    rebalance_dates = spy.groupby(spy['datetime'].dt.to_period('M'))['datetime'].min().tolist()

    stocks = {}
    for t, df in raw.items():
        if len(df) < 500:
            continue
        try:
            stocks[t] = build_state(df)
        except Exception:
            continue
    print(f"Built pivot-ratchet state for {len(stocks)} names\n")

    all_picks = []
    for i in range(len(rebalance_dates) - 1):
        d, d_next = rebalance_dates[i], rebalance_dates[i + 1]
        for t, s in stocks.items():
            dates = s['df']['datetime'].values
            idx = np.searchsorted(dates, np.datetime64(d), side='right') - 1
            if idx < 0 or idx >= len(dates):
                continue
            idx_next = np.searchsorted(dates, np.datetime64(d_next), side='right') - 1
            if idx_next < 0 or idx_next >= len(dates) or idx_next <= idx:
                continue
            if eligibility(s, idx):
                ret = s['close'][idx_next] / s['close'][idx] - 1
                all_picks.append({'month': d.strftime('%Y-%m'), 'symbol': t,
                                   'sector': sectors.get(t, 'UNKNOWN'), 'ret': ret})

    comp = pd.DataFrame(all_picks)
    print(f"{'='*100}\nSECTOR CONCENTRATION DIAGNOSTIC  (total picks: {len(comp)})\n{'='*100}")

    by_sector = comp.groupby('sector').agg(
        n_picks=('ret', 'size'), pct_of_picks=('ret', lambda x: len(x) / len(comp) * 100),
        avg_return=('ret', 'mean'), total_contribution=('ret', 'sum'),
    ).sort_values('n_picks', ascending=False)
    by_sector['pct_of_total_contribution'] = by_sector['total_contribution'] / by_sector['total_contribution'].sum() * 100
    with pd.option_context('display.float_format', '{:.2f}'.format):
        print(by_sector.to_string())

    top1_pct = by_sector['pct_of_picks'].iloc[0]
    top3_pct = by_sector['pct_of_picks'].iloc[:3].sum()
    universe_sector_dist = pd.Series(sectors).value_counts(normalize=True) * 100
    print(f"\ntop-1 sector share of picks: {top1_pct:.1f}%   top-3 sectors share of picks: {top3_pct:.1f}%")
    print(f"\nfor reference, sector composition of the FULL S&P 500 universe (not just picks):")
    print(universe_sector_dist.round(1).to_string())


if __name__ == '__main__':
    main()
