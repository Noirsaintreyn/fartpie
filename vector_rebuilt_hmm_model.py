"""
Rebuild of the saved best model (Long + HMM-gated short) on the
CORRECT engine and CORRECT data: the actual "SVM - Fractals" indicator
(plain EMA seeding, symmetric 0.35 break buffer, no regression filter,
no Path B, no adaptive anything - backtest_pivot_ratchet.run_state_machine,
confirmed against the real Pine source and matching a real chart's flip
sequence, e.g. AAPL 2019-02-25 to 2022-06-13 as one continuous hold) +
raw dividend-unadjusted prices for flip detection (batch_download_cached
now defaults to this) + dividend-adjusted returns for P&L + live-timing
(next trading day's OPEN, not the rebalance day's own close).

Same architecture as the original: real PIT S&P 500 membership, real
SEC EDGAR fundamentals gate, 30% sector cap, top-5 by quality z-score,
inverse-vol weighting, dynamic correlation-based exposure on the long
sleeve, HMM-gated (SPY-based, expanding-window, PIT-safe) short sleeve.
Turnover costs applied to both sleeves.
"""
import json
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM

from backtest_pivot_ratchet import run_state_machine as run_state_machine_full, ATR_LEN, SLOW_LEN, PIVOT_LEN
from backtest_pivot_ratchet import regime_strip_agreement
from vector_fundamentals_sector_cap import apply_sector_cap, stats_line, CAP_FRACTION
from vector_screener_sp500_sectors import batch_download_cached, load_sectors
from vector_fundamentals_backtest import BACKTEST_START, fundamentals_asof, load_facts_cache

warnings.filterwarnings('ignore', category=RuntimeWarning)

RECENT_FLIP_WINDOW = 20
MIN_FLIPS_FOR_QUALITY = 5
HORIZON = 20
TOP_N = 5
HMM_STRESS_THRESHOLD = 0.5
TURNOVER_COST_BPS = 10
CORR_WINDOW = 90
CORR_LOW, CORR_HIGH, FLOOR = 0.30, 0.55, 0.3


def exposure_multiplier(avg_corr):
    if np.isnan(avg_corr):
        return 1.0
    if avg_corr <= CORR_LOW:
        return 1.0
    if avg_corr >= CORR_HIGH:
        return FLOOR
    frac = (avg_corr - CORR_LOW) / (CORR_HIGH - CORR_LOW)
    return 1.0 - frac * (1.0 - FLOOR)


def trailing_returns_matrix(tickers, stocks, as_of_date, window=CORR_WINDOW):
    series = {}
    for t in tickers:
        df = stocks[t]['df']
        mask = df['datetime'] <= as_of_date
        closes = df.loc[mask, ['datetime', 'close']].tail(window + 1)
        if len(closes) < window // 2:
            continue
        rets = closes['close'].pct_change().dropna()
        rets.index = closes['datetime'].iloc[1:]
        series[t] = rets
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series)


def build_state(df):
    close = df['close'].values
    regimes, levels, atr, flips = run_state_machine_full(df)
    warmup = max(ATR_LEN, SLOW_LEN, PIVOT_LEN * 2) + 5
    # dividend-adjustment ratio, applied to open for RETURN computation only
    # (flip detection above already ran on raw close/high/low - unaffected)
    adj_factor = (df['adjclose'] / df['close']).values
    adj_open = df['open'].values * adj_factor
    return {'df': df, 'close': close, 'atr': atr, 'regimes': regimes, 'flips': flips, 'warmup': warmup,
            'adj_open': adj_open}


def eligibility_z(stock, idx, direction):
    regimes, close, flips, warmup = stock['regimes'], stock['close'], stock['flips'], stock['warmup']
    if idx < warmup + HORIZON + 10:
        return None
    recent = [f for f in flips if idx - RECENT_FLIP_WINDOW <= f[0] <= idx and f[1] == direction]
    if not recent:
        return None
    flips_so_far = sum(1 for f in flips if f[0] <= idx)
    if flips_so_far < MIN_FLIPS_FOR_QUALITY:
        return None
    strip = regime_strip_agreement(close[:idx + 1], regimes[:idx + 1], [HORIZON], warmup)
    z = float(strip.iloc[0]['z'])
    return z if z > 0 else None


def invvol_return(picks, sign=1):
    valid = [c for c in picks if not np.isnan(c['vol_pct']) and c['vol_pct'] > 0]
    if not valid:
        return 0.0
    inv_vol = np.array([1.0 / c['vol_pct'] for c in valid])
    weights = inv_vol / inv_vol.sum()
    rets = np.array([c['ret'] for c in valid])
    return float(np.sum(weights * rets)) * sign


def fit_hmm_stress_prob(weekly_log_rets):
    if len(weekly_log_rets) < 52:
        return np.nan
    X = weekly_log_rets.reshape(-1, 1)
    try:
        model = GaussianHMM(n_components=2, covariance_type='diag', n_iter=50, random_state=42)
        model.fit(X)
        probs = model.predict_proba(X)
        means = model.means_[:, 0]
        stress_state = int(np.argmin(means))
        return float(probs[-1, stress_state])
    except Exception:
        return np.nan


def turnover_costs(ticker_sets, top_n):
    counts = [len(ticker_sets[0])] + [
        len(ticker_sets[i] - ticker_sets[i - 1]) for i in range(1, len(ticker_sets))
    ]
    return np.array([n * (1.0 / top_n) * (TURNOVER_COST_BPS / 10000.0) for n in counts])


def main():
    facts_by_ticker = load_facts_cache()
    sectors = load_sectors()
    with open('sp500_pit_membership.json') as f:
        pit_membership_raw = json.load(f)
    with open('sp500_pit_all_tickers.json') as f:
        all_tickers = json.load(f)
    all_tickers_yf = [t.replace('.', '-') for t in all_tickers]
    raw = batch_download_cached(all_tickers_yf)

    # SPY benchmark - raw prices too, for consistency with the rest of the engine
    spy_raw = yf.download('SPY', period='max', interval='1d', auto_adjust=False, progress=False)
    spy_raw = spy_raw.reset_index()
    spy_raw.columns = [str(c[0] if isinstance(c, tuple) else c).lower() for c in spy_raw.columns]
    spy_raw = spy_raw.rename(columns={'date': 'datetime'})
    spy_raw['datetime'] = pd.to_datetime(spy_raw['datetime']).dt.tz_localize(None)
    spy_full = spy_raw.copy()
    spy = spy_raw[spy_raw['datetime'] >= BACKTEST_START].reset_index(drop=True)
    rebalance_dates = spy.groupby(spy['datetime'].dt.to_period('M'))['datetime'].min().tolist()
    spy_close, spy_open, spy_dates = spy['close'].values, spy['open'].values, spy['datetime'].values
    spy_adj_open = spy_open * (spy['adj close'].values / spy_close)
    spy_monthly = []
    for i in range(len(rebalance_dates) - 1):
        idx = np.searchsorted(spy_dates, np.datetime64(rebalance_dates[i]), side='right') - 1
        idx_next = np.searchsorted(spy_dates, np.datetime64(rebalance_dates[i + 1]), side='right') - 1
        # live-timing: enter/exit at the NEXT day's open after the rebalance date;
        # dividend-adjusted for a fair total-return comparison
        idx_fill = min(idx + 1, len(spy_adj_open) - 1)
        idx_next_fill = min(idx_next + 1, len(spy_adj_open) - 1)
        spy_monthly.append(spy_adj_open[idx_next_fill] / spy_adj_open[idx_fill] - 1)

    spy_weekly = spy_full.set_index('datetime')['close'].resample('W-FRI').last().dropna()
    spy_weekly_logret = np.log(spy_weekly / spy_weekly.shift(1)).dropna()
    print("Fitting HMM regime probability at each rebalance date (expanding window, PIT-safe)...")
    hmm_stress_prob = np.array([
        fit_hmm_stress_prob(spy_weekly_logret[spy_weekly_logret.index <= d].values)
        for d in rebalance_dates[:-1]
    ])
    print("  done\n")

    pit_membership = {mk: set(t.replace('.', '-') for t in ts) for mk, ts in pit_membership_raw.items()}
    stocks = {}
    for t, df in raw.items():
        if len(df) < 500:
            continue
        try:
            stocks[t] = build_state(df)
        except Exception:
            continue
    print(f"Built state for {len(stocks)} names\n")

    fund_cache = {}
    r_long, r_short_hmm_gated = [], []
    long_ticker_sets, short_ticker_sets = [], []

    for i in range(len(rebalance_dates) - 1):
        d, d_next = rebalance_dates[i], rebalance_dates[i + 1]
        as_of_str = d.strftime('%Y-%m-%d')
        members = pit_membership.get(d.strftime('%Y-%m'), set())
        long_candidates, short_candidates = [], []
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
            # LIVE-TIMING: fill at the NEXT bar's open (idx+1), not the
            # rebalance/eligibility bar's own close
            idx_fill = idx + 1
            idx_next_fill = idx_next + 1
            if idx_fill >= len(dates) or idx_next_fill >= len(dates):
                continue

            zl = eligibility_z(s, idx, direction=1)
            zs = eligibility_z(s, idx, direction=-1)
            if zl is None and zs is None:
                continue
            gaap = facts_by_ticker.get(t)
            fund_pass = None
            if gaap is not None:
                key = (t, as_of_str)
                if key not in fund_cache:
                    fund_cache[key] = fundamentals_asof(gaap, as_of_str)
                result = fund_cache[key]
                fund_pass = result is not None and result['strict_pass']

            adj_open = s['adj_open']
            price_ret = adj_open[idx_next_fill] / adj_open[idx_fill] - 1
            vol_pct = s['atr'][idx] / s['close'][idx] if not np.isnan(s['atr'][idx]) else np.nan
            if zl is not None and fund_pass:
                long_candidates.append({'ticker': t, 'ret': price_ret, 'vol_pct': vol_pct, 'z': zl})
            if zs is not None:
                short_candidates.append({'ticker': t, 'ret': -price_ret, 'vol_pct': vol_pct, 'z': zs})

        capped_long = apply_sector_cap(long_candidates, sectors)
        top5_long = sorted(capped_long, key=lambda c: -c['z'])[:TOP_N]
        long_ticker_sets.append(set(c['ticker'] for c in top5_long))
        r_long_month = invvol_return(top5_long)

        if len(capped_long) >= 2:
            ret_matrix = trailing_returns_matrix([c['ticker'] for c in capped_long], stocks, d)
            if not ret_matrix.empty and len(ret_matrix.columns) >= 2:
                corr_matrix = ret_matrix.corr()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                avg_corr = float(np.nanmean(upper.values))
            else:
                avg_corr = np.nan
        else:
            avg_corr = np.nan
        mult = exposure_multiplier(avg_corr)
        r_long.append(r_long_month * mult)

        capped_short = apply_sector_cap(short_candidates, sectors)
        top5_short = sorted(capped_short, key=lambda c: -c['z'])[:TOP_N]
        r_short_month = invvol_return(top5_short)
        stress_gate = 1.0 if (not np.isnan(hmm_stress_prob[i]) and hmm_stress_prob[i] > HMM_STRESS_THRESHOLD) else 0.0
        r_short_hmm_gated.append(r_short_month * stress_gate)
        short_ticker_sets.append(set(c['ticker'] for c in top5_short) if stress_gate == 1.0 else set())

    r_long = np.array(r_long)
    r_short_hmm_gated = np.array(r_short_hmm_gated)
    long_costs = turnover_costs(long_ticker_sets, TOP_N)
    short_costs = turnover_costs(short_ticker_sets, TOP_N)
    r_combined_before_costs = r_long + r_short_hmm_gated
    r_combined_after_costs = (r_long - long_costs) + (r_short_hmm_gated - short_costs)

    dates_arr = np.array(rebalance_dates[:-1])
    period_2020 = (dates_arr >= pd.Timestamp('2020-01-01')) & (dates_arr < pd.Timestamp('2021-01-01'))
    period_2022 = (dates_arr >= pd.Timestamp('2022-01-01')) & (dates_arr < pd.Timestamp('2023-01-01'))
    period_2023plus = dates_arr >= pd.Timestamp('2023-01-01')
    spy_arr = np.array(spy_monthly)

    print(f"{'='*100}\nREBUILT MODEL ON CORRECTED ENGINE (raw prices, Path A+B, regression filter, live-timing)\n{'='*100}")
    stats_line(spy_monthly, 'SPY buy & hold')
    stats_line(r_long, 'Long-only')
    stats_line(r_short_hmm_gated, 'Short-only, HMM-gated')
    stats_line(r_combined_before_costs, 'Combined, before costs')
    stats_line(r_combined_after_costs, 'Combined, after costs')

    print(f"\n{'='*100}\nLONG-ONLY, FLAT EXPOSURE TRIM SWEEP (on top of the existing dynamic correlation dial)\n{'='*100}")
    for trim in [1.0, 0.9, 0.8, 0.7, 0.6]:
        stats_line(r_long * trim, f'Long-only x {int(trim*100)}% flat trim')

    print(f"\n  2020:")
    stats_line(spy_arr[period_2020], '    SPY')
    stats_line(r_combined_after_costs[period_2020], '    Combined, after costs')
    print(f"\n  2022:")
    stats_line(spy_arr[period_2022], '    SPY')
    stats_line(r_combined_after_costs[period_2022], '    Combined, after costs')
    print(f"\n  2023-2026 (untouched period from the original rigor test):")
    stats_line(spy_arr[period_2023plus], '    SPY')
    stats_line(r_combined_after_costs[period_2023plus], '    Combined, after costs')


if __name__ == '__main__':
    main()
