"""
THE real fundamentals backtest: real point-in-time S&P 500 universe,
validated long-only vector (recently flipped + quality gate), now with a
genuine point-in-time fundamentals gate from SEC EDGAR (filed-date
correct, not today's-numbers-applied-backward). Uses annual 10-K filings
as the cleanest, most standardized snapshot - fundamentals don't need
monthly resolution for a monthly-rebalance long-horizon screen anyway.

Same four pre-declared checks as the live screener: positive profit
margin, positive revenue growth (vs prior fiscal year), debt/equity under
a fixed bound, current ratio >= 1 - skipped (not failed) for companies
that don't report a classified current/non-current balance sheet at all
(mainly financials/insurers, a real GAAP convention, not missing data).
"""
import io
import json
import os

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from backtest_pivot_ratchet import run_state_machine, regime_strip_agreement, ATR_LEN, SLOW_LEN, PIVOT_LEN
from vector_screener_sp500_sectors import batch_download_cached

BACKTEST_START = '2018-01-01'
RECENT_FLIP_WINDOW = 20
MIN_FLIPS_FOR_QUALITY = 5
HORIZON = 20
MAX_DEBT_TO_EQUITY = 150.0
CACHE_DIR = 'sec_edgar_cache'

REVENUE_TAGS = ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'SalesRevenueNet']
NET_INCOME_TAGS = ['NetIncomeLoss']
EQUITY_TAGS = ['StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest']
LIABILITIES_TAGS = ['Liabilities']
CURRENT_ASSETS_TAGS = ['AssetsCurrent']
CURRENT_LIAB_TAGS = ['LiabilitiesCurrent']


def load_facts_cache():
    """The full local SEC EDGAR cache (2GB+ of pre-fetched company facts)
    is a local research artifact, never committed - a fresh checkout
    (e.g. the deployed server) has no CACHE_DIR at all. Return whatever
    is actually on disk instead of crashing; get_facts_for_ticker below
    fills in anything missing with a live, on-demand fetch."""
    facts_by_ticker = {}
    if not os.path.isdir(CACHE_DIR):
        return facts_by_ticker
    for fn in os.listdir(CACHE_DIR):
        with open(f"{CACHE_DIR}/{fn}") as f:
            d = json.load(f)
        facts_by_ticker[d['ticker']] = d['facts']['facts'].get('us-gaap', {})
    return facts_by_ticker


_CIK_MAP_CACHE = None


def load_cik_map():
    """Small (~12KB) ticker->CIK mapping, committed to the repo - built
    once from the local SEC cache's filenames, not fetched live, so a
    fresh deploy doesn't need a network round-trip just to know which
    CIK belongs to which ticker."""
    global _CIK_MAP_CACHE
    if _CIK_MAP_CACHE is None:
        with open('sp500_ticker_cik_map.json') as f:
            _CIK_MAP_CACHE = json.load(f)
    return _CIK_MAP_CACHE


def get_facts_for_ticker(ticker, facts_by_ticker):
    """Look up one ticker's GAAP facts, live-fetching from SEC EDGAR's
    free companyfacts API on a cache miss and disk-caching the result -
    the deployed server has no pre-built cache, so this is what actually
    supplies fundamentals data there. facts_by_ticker is mutated in
    place so repeat lookups within the same request/process are free."""
    gaap = facts_by_ticker.get(ticker)
    if gaap is not None:
        return gaap
    cik = load_cik_map().get(ticker)
    if cik is None:
        return None
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = f"{CACHE_DIR}/{ticker}_{cik}.json"
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                d = json.load(f)
            gaap = d['facts']['facts'].get('us-gaap', {})
            facts_by_ticker[ticker] = gaap
            return gaap
        except Exception:
            pass
    try:
        headers = {'User-Agent': 'fartpie-research (personal project) contact@example.com'}
        r = requests.get(f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json', headers=headers, timeout=15)
        if r.status_code != 200:
            return None
        facts = r.json()
    except Exception:
        return None
    try:
        with open(cache_path, 'w') as f:
            json.dump({'ticker': ticker, 'cik': cik, 'facts': facts}, f)
    except Exception:
        pass  # best-effort disk cache - fine if the filesystem is read-only or ephemeral
    gaap = facts.get('facts', {}).get('us-gaap', {})
    facts_by_ticker[ticker] = gaap
    return gaap


def annual_series(gaap, tag_candidates):
    for tag in tag_candidates:
        if tag in gaap:
            entries = gaap[tag]['units'].get('USD', [])
            annual = [e for e in entries if e.get('form') == '10-K' and e.get('fp') == 'FY' and 'end' in e and 'filed' in e]
            if annual:
                return annual
    return []


def value_as_of(annual_entries, as_of_date_str, want_prior_year=False):
    eligible = [e for e in annual_entries if e['filed'] <= as_of_date_str]
    if not eligible:
        return None
    # de-dup by end-period, keep latest filed version of each
    by_end = {}
    for e in eligible:
        if e['end'] not in by_end or e['filed'] > by_end[e['end']]['filed']:
            by_end[e['end']] = e
    ordered = sorted(by_end.values(), key=lambda e: e['end'])
    if not ordered:
        return None
    if want_prior_year:
        return ordered[-2]['val'] if len(ordered) >= 2 else None
    return ordered[-1]['val']


def fundamentals_asof(gaap, as_of_date_str):
    rev_series = annual_series(gaap, REVENUE_TAGS)
    ni_series = annual_series(gaap, NET_INCOME_TAGS)
    eq_series = annual_series(gaap, EQUITY_TAGS)
    liab_series = annual_series(gaap, LIABILITIES_TAGS)
    ca_series = annual_series(gaap, CURRENT_ASSETS_TAGS)
    cl_series = annual_series(gaap, CURRENT_LIAB_TAGS)

    revenue = value_as_of(rev_series, as_of_date_str)
    revenue_prior = value_as_of(rev_series, as_of_date_str, want_prior_year=True)
    net_income = value_as_of(ni_series, as_of_date_str)
    equity = value_as_of(eq_series, as_of_date_str)
    liabilities = value_as_of(liab_series, as_of_date_str)
    curr_assets = value_as_of(ca_series, as_of_date_str)
    curr_liab = value_as_of(cl_series, as_of_date_str)

    if revenue is None or net_income is None:
        return None  # can't evaluate at all

    profit_margin = net_income / revenue if revenue else None
    revenue_growth = (revenue - revenue_prior) / abs(revenue_prior) if revenue_prior else None
    debt_to_equity = (liabilities / equity * 100) if (equity and equity > 0 and liabilities is not None) else None
    current_ratio = (curr_assets / curr_liab) if (curr_liab and curr_liab > 0 and curr_assets is not None) else None

    if profit_margin is None or revenue_growth is None:
        return None

    available_checks = [profit_margin > 0, revenue_growth > 0]
    if debt_to_equity is not None:
        available_checks.append(debt_to_equity < MAX_DEBT_TO_EQUITY)
    if current_ratio is not None:
        available_checks.append(current_ratio >= 1.0)

    n_passed = sum(available_checks)
    n_total = len(available_checks)
    return {
        'strict_pass': n_passed == n_total, 'n_passed': n_passed, 'n_total': n_total,
        # actual metric values, added for narrative reasoning (e.g. the
        # lookup tool) - purely additive, every existing caller that only
        # reads strict_pass/n_passed/n_total is unaffected
        'profit_margin': profit_margin, 'revenue_growth': revenue_growth,
        'debt_to_equity': debt_to_equity, 'current_ratio': current_ratio,
    }


def fetch_fred(series_id, start='2010-01-01'):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
    r = requests.get(url, timeout=30)
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ['date', series_id]
    df['date'] = pd.to_datetime(df['date'])
    df[series_id] = pd.to_numeric(df[series_id], errors='coerce')
    return df.dropna()


def build_state(df):
    close = df['close'].values
    regimes, levels, atr, flips = run_state_machine(df)
    warmup = max(ATR_LEN, SLOW_LEN, PIVOT_LEN * 2) + 5
    return {'df': df, 'close': close, 'atr': atr, 'regimes': regimes, 'flips': flips, 'warmup': warmup}


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


def stats_line(returns, label):
    r = np.array(returns)
    total = float(np.prod(1 + r) - 1)
    ann = float((1 + total) ** (12 / len(r)) - 1) if len(r) else np.nan
    sharpe = float(r.mean() / r.std() * np.sqrt(12)) if r.std() > 0 else np.nan
    equity = np.cumprod(1 + r)
    dd = float((equity / np.maximum.accumulate(equity) - 1).min())
    print(f"  {label:<40} total={total*100:+8.1f}%  ann={ann*100:+6.1f}%  sharpe={sharpe:+.2f}  max_dd={dd*100:6.1f}%")


def main():
    print("Loading SEC EDGAR fundamentals cache...")
    facts_by_ticker = load_facts_cache()
    print(f"  {len(facts_by_ticker)} companies with fundamentals data\n")

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

    r_vector_only, r_with_fundamentals, r_invvol = [], [], []
    n_vec, n_fund = [], []
    fund_cache = {}
    for i in range(len(rebalance_dates) - 1):
        d, d_next = rebalance_dates[i], rebalance_dates[i + 1]
        as_of_str = d.strftime('%Y-%m-%d')
        members = pit_membership.get(d.strftime('%Y-%m'), set())
        picks_vec = []
        strict_picks = []  # (ret, vol_pct) for equal-weight and inverse-vol comparison
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
            if not eligibility(s, idx):
                continue
            ret = s['close'][idx_next] / s['close'][idx] - 1
            picks_vec.append(ret)

            gaap = facts_by_ticker.get(t)
            if gaap is None:
                continue
            key = (t, as_of_str)
            if key not in fund_cache:
                fund_cache[key] = fundamentals_asof(gaap, as_of_str)
            result = fund_cache[key]
            if result is None or not result['strict_pass']:
                continue
            vol_pct = s['atr'][idx] / s['close'][idx] if not np.isnan(s['atr'][idx]) else np.nan
            strict_picks.append((ret, vol_pct))

        r_vector_only.append(np.mean(picks_vec) if picks_vec else 0.0)
        rets_only = [r for r, _ in strict_picks]
        r_with_fundamentals.append(np.mean(rets_only) if rets_only else 0.0)
        n_vec.append(len(picks_vec))
        n_fund.append(len(strict_picks))

        valid = [(r, v) for r, v in strict_picks if not np.isnan(v) and v > 0]
        if valid:
            inv_vol = np.array([1.0 / v for _, v in valid])
            weights = inv_vol / inv_vol.sum()
            rets_arr = np.array([r for r, _ in valid])
            r_invvol.append(float(np.sum(weights * rets_arr)))
        else:
            r_invvol.append(0.0)
    print(f"\n{'='*100}\nVECTOR + FUNDAMENTALS: EQUAL-WEIGHT vs INVERSE-VOLATILITY WEIGHTED\n{'='*100}")
    stats_line(spy_monthly, 'SPY buy & hold')
    stats_line(r_vector_only, 'Vector only (point-in-time universe)')
    stats_line(r_with_fundamentals, 'Vector + fundamentals gate (equal-weight)')
    stats_line(r_invvol, 'Vector + fundamentals gate (inverse-vol weighted)')
    print(f"\n  avg picks/month - vector only: {np.mean(n_vec):.2f}   strict fundamentals: {np.mean(n_fund):.2f}")
    print(f"  months with zero strict-passing picks: {sum(1 for x in n_fund if x == 0)}/{len(n_fund)}")


if __name__ == '__main__':
    main()
