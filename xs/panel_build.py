"""
Phase 0 / step 1 - survivorship-aware daily panel for the S&P 500 PIT universe.

Inputs (all already in fartpie/):
  sp500_pit_membership.json   monthly PIT membership 2018-01..2026-08
  sp500_cache_raw2/*.parquet  daily OHLC + adjclose (preferred)
  sp500_cache_adjusted_OLD/   adjusted OHLC (fallback)
  sp500_sectors.json          GICS sector (current members)
  sp500_subindustries.json    Yahoo sector (fallback)
  lse_ohlcv_cache/*_1h_*      hourly volume (optional, relative use only)

Outputs (xs/data/):
  panel.parquet          long panel: date, ticker, sector, in_univ, open, close,
                         ret (total, close-close), ret_on (overnight), ret_id (intraday), vol
  coverage.csv           per-month: members, members with price data, coverage %
  missing_tickers.csv    PIT members with no usable prices (the survivorship gap)

Conventions:
  - Membership for month M applies to every trading day in M.
  - Renamed tickers are mapped to the successor's price history (same legal entity only).
  - Adjusted open = open * adjclose/close, so ret_on + ret_id compound to ret.
  - A member whose price history ends while still in the index gets in_univ=False
    after its last bar and is listed in missing_tickers.csv (reason='ends_early').
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'xs', 'data')
START, END = '2018-01-01', '2026-08-31'

# Same legal entity continuing under a new ticker. Acquired targets are NOT here.
ALIAS = {
    'ABC': 'COR', 'ANTM': 'ELV', 'BLL': 'BALL', 'BF.B': 'BF-B', 'BRK.B': 'BRK-B',
    'FLT': 'CPAY', 'JEC': 'J', 'PKI': 'RVTY', 'RE': 'EG', 'PEAK': 'DOC', 'HCP': 'DOC',
    'WLTW': 'WTW', 'NLOK': 'GEN', 'SYMC': 'GEN', 'CTL': 'LUMN', 'DISCA': 'WBD', 'UTX': 'RTX', 'TMK': 'GL', 'DWDP': 'DD',
    'FI': 'FISV', 'MMC': 'MRSH', 'BK': 'BNY', 'ARNC': 'HWM', 'MYL': 'VTRS',
    'BHGE': 'BKR', 'HRS': 'LHX', 'KORS': 'CPRI',
}
YAHOO_TO_GICS = {
    'Healthcare': 'Health Care', 'Consumer Cyclical': 'Consumer Discretionary',
    'Consumer Defensive': 'Consumer Staples', 'Technology': 'Information Technology',
    'Financial Services': 'Financials', 'Basic Materials': 'Materials',
}


def load_membership():
    m = json.load(open(os.path.join(ROOT, 'sp500_pit_membership.json')))
    rows = [(k, t) for k, v in m.items() for t in v]
    return pd.DataFrame(rows, columns=['month', 'ticker'])


def load_prices(sym):
    for d in ('sp500_cache_raw2', 'sp500_cache_adjusted_OLD'):
        p = os.path.join(ROOT, d, f'{sym}.parquet')
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p)
        df['date'] = pd.to_datetime(df['datetime']).dt.normalize()
        if 'adjclose' in df:
            f = df['adjclose'] / df['close']
            df['open'], df['close'] = df['open'] * f, df['adjclose']
        df = df[(df['date'] >= pd.Timestamp(START) - pd.Timedelta(days=400)) &
                (df['date'] <= END)]
        df = df[['date', 'open', 'close']].dropna()
        df = df[(df['open'] > 0) & (df['close'] > 0)].drop_duplicates('date')
        if len(df) > 20:
            return df.sort_values('date'), d
    return None, None


def load_volume(sym):
    d = os.path.join(ROOT, 'lse_ohlcv_cache')
    cands = sorted(f for f in os.listdir(d) if f.startswith(f'{sym}_1h_2018-01-01_2026'))
    if not cands:
        return None
    v = pd.read_parquet(os.path.join(d, cands[-1]))
    v['date'] = pd.to_datetime(v['datetime']).dt.normalize()
    return v.groupby('date')['volume'].sum().rename('vol')


def load_sectors():
    gics = json.load(open(os.path.join(ROOT, 'sp500_sectors.json')))
    yh = json.load(open(os.path.join(ROOT, 'sp500_subindustries.json')))
    out = {t: YAHOO_TO_GICS.get(v['sector'], v['sector']) for t, v in yh.items() if v.get('sector')}
    out.update(gics)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    mem = load_membership()
    tickers = sorted(mem['ticker'].unique())
    sectors = load_sectors()
    frames, missing = [], []
    for i, t in enumerate(tickers):
        src_sym = ALIAS.get(t, t)
        px, src = load_prices(src_sym)
        months = mem.loc[mem.ticker == t, 'month']
        if px is None:
            missing.append((t, src_sym, months.min(), months.max(), len(months), 'no_prices'))
            continue
        px = px.copy()
        px['ticker'] = t
        px['ret'] = px['close'].pct_change()
        px['ret_on'] = px['open'] / px['close'].shift(1) - 1
        px['ret_id'] = px['close'] / px['open'] - 1
        vol = load_volume(src_sym)
        px['vol'] = px['date'].map(vol) if vol is not None else np.nan
        # data-quality: spike-and-reverse bars (|ret|>35% then opposite >20%) = corrupt series
        win = (px['date'] >= pd.Period(months.min()).start_time) & \
              (px['date'] <= pd.Period(months.max()).end_time)
        r = px['ret'].where(win)
        spikes = int(((r.abs() > 0.35) & (np.sign(r) != np.sign(r.shift(-1))) &
                      (r.shift(-1).abs() > 0.20)).sum())
        if spikes >= 2:
            missing.append((t, src_sym, months.min(), months.max(), len(months),
                            f'bad_data:{spikes}_spikes'))
            continue
        px['sector'] = sectors.get(t, sectors.get(src_sym, 'Unknown'))
        px['src'] = src
        last_member_end = pd.Period(months.max()).end_time.normalize()
        if px['date'].max() < min(last_member_end, pd.Timestamp(END)) - pd.Timedelta(days=7):
            missing.append((t, src_sym, months.min(), months.max(), len(months),
                            f"ends_early:{px['date'].max().date()}"))
        frames.append(px)
        if (i + 1) % 100 == 0:
            print(f'  {i + 1}/{len(tickers)}', flush=True)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel[(panel['date'] >= START) & (panel['date'] <= END)]
    panel['month'] = panel['date'].dt.strftime('%Y-%m')
    mem['in_univ'] = True
    panel = panel.merge(mem, on=['month', 'ticker'], how='left')
    panel['in_univ'] = panel['in_univ'].eq(True)
    panel = panel.drop(columns='month').sort_values(['date', 'ticker']).reset_index(drop=True)
    panel.to_parquet(os.path.join(OUT, 'panel.parquet'), index=False)

    have = set(panel['ticker'])
    cov = mem.groupby('month')['ticker'].agg(
        members='size', with_prices=lambda s: s.isin(have).sum()).reset_index()
    cov['coverage_pct'] = (100 * cov.with_prices / cov.members).round(1)
    cov.to_csv(os.path.join(OUT, 'coverage.csv'), index=False)
    pd.DataFrame(missing, columns=['ticker', 'price_symbol', 'first_month', 'last_month',
                                   'months_in_index', 'reason']).to_csv(
        os.path.join(OUT, 'missing_tickers.csv'), index=False)

    u = panel[panel.in_univ]
    print(f"\npanel rows {len(panel):,}  tickers {panel.ticker.nunique()}  "
          f"in-universe rows {len(u):,}  dates {u.date.min().date()}..{u.date.max().date()}")
    print(f"coverage: min {cov.coverage_pct.min()}%  median {cov.coverage_pct.median()}%  "
          f"mean {cov.coverage_pct.mean():.1f}%")
    print(f"missing/ends-early tickers: {len(missing)}  "
          f"(member-months lost: {sum(m[4] for m in missing if m[5] != 'ends_early' and not m[5].startswith('ends_early'))})")
    print(f"sector Unknown in-universe rows: {(u.sector == 'Unknown').mean():.2%}")
    print(f"vol available in-universe rows: {u.vol.notna().mean():.1%}")
    bad = u[(u.ret.abs() > 0.5)]
    print(f"|ret|>50% in-universe rows: {len(bad)}")
    if len(bad):
        print(bad[['date', 'ticker', 'ret']].head(15).to_string(index=False))


if __name__ == '__main__':
    sys.exit(main())
