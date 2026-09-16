"""
Fixes the R:R mismatch flagged in review: evaluate_level()'s 'bounced'
label is defined against a FIXED, arbitrary 2:1 hurdle (bounce_atr_mult=1.0
vs break_atr_mult=0.5), while realized_rr (mfe/mae, computed separately in
backtest_ml_filter_v3_events.py) measures actual excursion under a
DIFFERENT, unrelated definition. Multiplying one by the other doesn't give
real expectancy - they're not the same trade.

This runs ONE coherent trade simulation instead: pick an explicit risk unit
(1.0x ATR stop) and an explicit reward multiple (R = 1.0 / 1.5 / 2.0x ATR
target), and score win/loss under THAT SAME rule - so win rate and R are
finally consistent and expectancy is real.

Stop/target resolution is intrabar-touch based (a real stop-loss order
fills on touch, not on a confirmed close) - a deliberate change from
evaluate_level's close-based+confirm-bars approach, which was built for a
different question ("did this look like a real S/R defense") not "would
this trade have made money". If both stop and target are touched within
the same bar, conservatively scored as a loss (can't know which happened
first intrabar).

Expectancy is reported both gross (in R units) and after a stated cost
assumption (COST_FRAC_OF_RISK = 0.10, i.e. 10% of the 1R stop distance
round-trip - a deliberately conservative placeholder for spread+slippage
on a liquid instrument, not a measured number; call it out as an
assumption, not fact, when reading the results).
"""
import contextlib
import io
import time

import numpy as np
import pandas as pd

from backtest_levels import load_csv, compute_atr, METHODS
from fetch_lse_ohlcv import fetch_ohlcv
from datetime import datetime

LOOKBACK = 150
STEP = 10
MAX_HOLD_BARS = 40
RISK_ATR_MULT = 1.0
REWARD_R_MULTIPLES = [1.0, 1.5, 2.0]
COST_FRAC_OF_RISK = 0.10  # stated assumption, not measured - see module docstring


def simulate_trade_rr(price, side, fwd_high, fwd_low, atr, risk_atr_mult, reward_r):
    touch_idx = None
    for i in range(len(fwd_high)):
        if fwd_low[i] <= price <= fwd_high[i]:
            touch_idx = i
            break
    if touch_idx is None:
        return None

    risk = risk_atr_mult * atr
    reward = reward_r * risk_atr_mult * atr
    if side == 'support':
        stop_level = price - risk
        target_level = price + reward
    else:
        stop_level = price + risk
        target_level = price - reward

    end = min(touch_idx + MAX_HOLD_BARS, len(fwd_high))
    for i in range(touch_idx, end):
        hi, lo = fwd_high[i], fwd_low[i]
        stop_hit = (lo <= stop_level) if side == 'support' else (hi >= stop_level)
        target_hit = (hi >= target_level) if side == 'support' else (lo <= target_level)
        if stop_hit and target_hit:
            return {'outcome': 'loss', 'bars_held': i - touch_idx}  # conservative: can't resolve intrabar order
        if stop_hit:
            return {'outcome': 'loss', 'bars_held': i - touch_idx}
        if target_hit:
            return {'outcome': 'win', 'bars_held': i - touch_idx}
    return {'outcome': 'unresolved', 'bars_held': end - touch_idx}


def run_coherent_backtest(df, instrument, timeframe, reward_r_list):
    """Runs level detection ONCE per window, then scores every detected
    level against every reward_r in reward_r_list - the expensive part
    (7-8 clustering/anomaly methods per window) doesn't get repeated per
    R multiple, only the cheap stop/target check does."""
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n = len(df)
    starts = list(range(LOOKBACK, n - MAX_HOLD_BARS, STEP))

    rows = []
    for t in starts:
        win_h, win_l, win_c = highs[t - LOOKBACK:t], lows[t - LOOKBACK:t], closes[t - LOOKBACK:t]
        current_price = closes[t - 1]
        atr = compute_atr(win_h, win_l, win_c)
        if atr <= 0:
            continue
        fwd_h, fwd_l = highs[t:t + MAX_HOLD_BARS], lows[t:t + MAX_HOLD_BARS]

        for method_name, fn in METHODS.items():
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    levels = fn(win_h, win_l, win_c, timeframe)
            except Exception:
                continue
            for lvl in levels or []:
                price = lvl.get('price')
                if price is None or price <= 0:
                    continue
                side = 'support' if price < current_price else 'resistance'
                for reward_r in reward_r_list:
                    outcome = simulate_trade_rr(price, side, fwd_h, fwd_l, atr, RISK_ATR_MULT, reward_r)
                    if outcome is None:
                        continue
                    rows.append({'instrument': instrument, 'timeframe': timeframe, 'method': method_name,
                                 'reward_r': reward_r, 'outcome': outcome['outcome']})
    return pd.DataFrame(rows)


def summarize_rr(results):
    def agg(g):
        resolved = g[g['outcome'] != 'unresolved']
        n_resolved = len(resolved)
        n_wins = (resolved['outcome'] == 'win').sum()
        win_rate = n_wins / n_resolved if n_resolved else np.nan
        R = g['reward_r'].iloc[0]
        expectancy_gross = win_rate * R - (1 - win_rate) * 1 if n_resolved else np.nan
        cost_R = COST_FRAC_OF_RISK  # cost expressed directly in R units (fraction of the 1R risk unit)
        expectancy_net = expectancy_gross - cost_R if n_resolved else np.nan
        breakeven_win_rate = 1 / (1 + R)
        return pd.Series({
            'n_trades_resolved': n_resolved,
            'n_unresolved': len(g) - n_resolved,
            'win_rate': win_rate,
            'breakeven_win_rate_needed': breakeven_win_rate,
            'edge_vs_breakeven_pp': (win_rate - breakeven_win_rate) * 100 if n_resolved else np.nan,
            'expectancy_R_gross': expectancy_gross,
            'expectancy_R_net_of_cost': expectancy_net,
        })
    return results.groupby(['instrument', 'timeframe', 'reward_r', 'method']).apply(agg).reset_index()


def load_csv_semicolon(path, skiprows):
    """Same export family as load_csv, but the 5Min/15Min/1Min NQ files use
    ';' delimiters (no comma-in-number quoting needed) and the 1Min file
    additionally has no title row above the header."""
    from backtest_levels import _parse_euro_number
    df = pd.read_csv(path, sep=';', skiprows=skiprows)
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].apply(_parse_euro_number)
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    df['datetime'] = pd.to_datetime(df['date'], format='%m/%d/%Y %I:%M %p', errors='coerce')
    df = df.dropna(subset=['datetime', 'open', 'high', 'low', 'close'])
    return df.sort_values('datetime').reset_index(drop=True)[['datetime', 'open', 'high', 'low', 'close', 'volume']]


def load_source(instrument, timeframe):
    lse_map = {'AAPL': 'AAPL', 'AMD': 'AMD'}
    csv_map = {
        ('NQ', '1h'): '/Users/rey/Downloads/1H_NQ.csv', ('NQ', '4h'): '/Users/rey/Downloads/4H_NQ.csv',
        ('GC', '1h'): '/Users/rey/fartpie/data/1H_GC.csv', ('GC', '4h'): '/Users/rey/fartpie/data/4H_GC.csv',
    }
    # ';'-delimited exports, different header/skiprows convention than the
    # comma-delimited csv_map files above.
    semicolon_csv_map = {
        ('NQ', '15m'): ('/Users/rey/Downloads/15Min_NQ.csv', 1),
        ('NQ', '5m'): ('/Users/rey/Downloads/NQ_5Min.csv', 1),
        ('NQ', '1m'): ('/Users/rey/Downloads/NQ/1Min/1Min_NQ.csv', 0),
    }
    if instrument in lse_map:
        tf_api = {'1d': '1d', '4h': '4h', '1h': '1h', '15m': '15m', '5m': '5m', '1m': '1m'}[timeframe]
        # Sub-hourly bars over the full 2010+ history would be tens of
        # millions of rows and impractical to fetch/cache - cap the lookback
        # per granularity instead, same tradeoff data_loader.py already
        # makes for intraday yfinance pulls.
        intraday_start = {'1m': datetime(2024, 1, 1), '5m': datetime(2021, 1, 1), '15m': datetime(2018, 1, 1)}
        start = intraday_start.get(timeframe, datetime(2010, 1, 1))
        df = fetch_ohlcv(lse_map[instrument], tf_api, start, datetime(2026, 8, 15))
        return df
    if (instrument, timeframe) in semicolon_csv_map:
        path, skiprows = semicolon_csv_map[(instrument, timeframe)]
        return load_csv_semicolon(path, skiprows)
    path = csv_map.get((instrument, timeframe))
    if path is None:
        return None
    return load_csv(path)


def main():
    targets = [
        ('AAPL', '1d'), ('AAPL', '4h'), ('AAPL', '1h'),
        ('AMD', '1d'), ('AMD', '4h'), ('AMD', '1h'),
        ('NQ', '1h'), ('NQ', '4h'),
        ('GC', '1h'), ('GC', '4h'),
    ]
    all_results = []
    for instrument, timeframe in targets:
        print(f"\n{'='*80}\n{instrument} {timeframe}\n{'='*80}")
        df = load_source(instrument, timeframe)
        if df is None or len(df) < LOOKBACK + MAX_HOLD_BARS + STEP:
            print("  no/insufficient data, skipping")
            continue
        print(f"  {len(df)} bars, {df['datetime'].min()} -> {df['datetime'].max()}")
        t0 = time.time()
        res = run_coherent_backtest(df, instrument, timeframe, REWARD_R_MULTIPLES)
        print(f"  {len(res)} trades simulated across R={REWARD_R_MULTIPLES} ({time.time()-t0:.1f}s)")
        all_results.append(res)

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv('backtest_levels_coherent_rr_results.csv', index=False)
    summary = summarize_rr(results)
    summary.to_csv('backtest_levels_coherent_rr_summary.csv', index=False)

    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.max_rows', 300)
    print(f"\n\n=== Coherent R:R summary (cost assumption: {COST_FRAC_OF_RISK}R round-trip) ===")
    print(summary.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
