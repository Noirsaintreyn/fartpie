"""
Backtest reference-point levels (prior day/week/month high-low, session
open, round numbers, VWAP) against the same random-baseline discipline
used to disprove the clustering-based methods in backtest_levels.py.

Unlike those methods, these need the FULL history up to each point (not
just a fixed lookback window) - a prior month's high/low isn't visible in
150 bars. Reuses evaluate_level/compute_atr so results are directly
comparable to everything already backtested.
"""
import argparse
import contextlib
import io
import os
import re
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from backtest_levels import load_csv, infer_timeframe, compute_atr, evaluate_level
import reference_levels as rl

with contextlib.redirect_stdout(io.StringIO()):
    import backend

_RANDOM_RNG = np.random.default_rng(42)


def build_method_levels(method, df_arrays, t, win_h, win_l, win_c):
    dates, week_keys, month_keys, opens, highs, lows, closes, volumes, datetimes = df_arrays
    current_price = closes[t - 1]

    if method == 'Prior-Day-HL':
        return rl.prior_day_levels(dates, highs, lows, closes, t)
    elif method == 'Prior-Week-HL':
        return rl.prior_week_levels(week_keys, highs, lows, t)
    elif method == 'Prior-Month-HL':
        return rl.prior_month_levels(month_keys, highs, lows, t)
    elif method == 'Session-Open':
        return rl.session_open_level(dates, opens, t)
    elif method == 'Round-Numbers':
        return rl.round_number_levels(current_price)
    elif method == 'VWAP':
        return rl.vwap_levels(backend, win_h, win_l, win_c, volumes[t - len(win_c):t], datetimes[t - len(win_c):t])
    elif method == 'Random':
        lo, hi = float(win_l.min()), float(win_h.max())
        if hi <= lo:
            return []
        prices = _RANDOM_RNG.uniform(lo, hi, size=8)
        return [{'price': float(p), 'category': 'Random', 'type': 'Random Baseline'} for p in prices]
    return []


METHOD_NAMES = ['Prior-Day-HL', 'Prior-Week-HL', 'Prior-Month-HL', 'Session-Open',
                'Round-Numbers', 'VWAP', 'Random']


def run_file(path, lookback, horizon, step,
             bounce_atr_mult, break_atr_mult, break_confirm_bars, reaction_bars, recovery_bars,
             verbose=False):
    instrument = re.sub(r'^\d+[a-zA-Z]+_', '', os.path.splitext(os.path.basename(path))[0])
    timeframe = infer_timeframe(os.path.basename(path))
    print(f"Loading {path} (instrument={instrument}, timeframe={timeframe})...")
    df = load_csv(path)
    n = len(df)

    dates = df['datetime'].dt.date.values
    iso = df['datetime'].dt.isocalendar()
    week_keys = (iso['year'].astype('int64') * 100 + iso['week'].astype('int64')).to_numpy()
    month_keys = (df['datetime'].dt.year.astype('int64') * 100 + df['datetime'].dt.month.astype('int64')).to_numpy()
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    volumes = df['volume'].values
    datetimes = df['datetime'].values
    df_arrays = (dates, week_keys, month_keys, opens, highs, lows, closes, volumes, datetimes)

    starts = list(range(lookback, n - horizon, step))
    rows = []
    t0 = time.time()
    for wi, t in enumerate(starts):
        win_h = highs[t - lookback:t]
        win_l = lows[t - lookback:t]
        win_c = closes[t - lookback:t]
        current_price = closes[t - 1]
        atr = compute_atr(win_h, win_l, win_c)
        if atr <= 0:
            continue

        fwd_h = highs[t:t + horizon]
        fwd_l = lows[t:t + horizon]
        fwd_c = closes[t:t + horizon]

        for method in METHOD_NAMES:
            try:
                levels = build_method_levels(method, df_arrays, t, win_h, win_l, win_c)
            except Exception as e:
                if verbose:
                    print(f"  [{method}] error at t={t}: {e}")
                continue

            for lvl in levels or []:
                price = lvl.get('price')
                if price is None or price <= 0:
                    continue
                side = 'support' if price < current_price else 'resistance'
                outcome = evaluate_level(price, side, fwd_h, fwd_l, fwd_c, atr,
                                          bounce_atr_mult, break_atr_mult, break_confirm_bars,
                                          reaction_bars, recovery_bars)
                rows.append({
                    'instrument': instrument, 'timeframe': timeframe, 'method': method,
                    'type': lvl.get('type'), 't': t, 'price': price, 'side': side,
                    'touched': outcome is not None,
                    'bounced': outcome['bounced'] if outcome else None,
                    'broken': outcome['broken'] if outcome else None,
                    'hold_duration': outcome['hold_duration'] if outcome else None,
                    'recovered': outcome['recovered'] if outcome else None,
                })
        if verbose and wi % 200 == 0:
            print(f"  window {wi}/{len(starts)} ({time.time()-t0:.1f}s elapsed)")

    return pd.DataFrame(rows)


def summarize(results):
    def agg(g):
        touched = g[g['touched']]
        n_levels = len(g)
        n_touched = len(touched)
        n_bounced = touched['bounced'].sum() if n_touched else 0
        n_broken = touched['broken'].sum() if n_touched else 0
        recovered_subset = touched[touched['broken'] == True]  # noqa: E712
        n_recovered = recovered_subset['recovered'].sum() if len(recovered_subset) else 0
        return pd.Series({
            'n_levels_generated': n_levels, 'n_touched': n_touched,
            'touch_rate': n_touched / n_levels if n_levels else np.nan,
            'support_accuracy': n_bounced / n_touched if n_touched else np.nan,
            'break_rate': n_broken / n_touched if n_touched else np.nan,
            'false_breakout_recovery': n_recovered / n_broken if n_broken else np.nan,
            'avg_hold_duration_bars': touched['hold_duration'].mean() if n_touched else np.nan,
        })
    per_file = results.groupby(['instrument', 'timeframe', 'method']).apply(agg).reset_index()
    overall = results.groupby('method').apply(agg).reset_index()
    return per_file, overall


def zscore_test(p1, n1, p2, n2):
    x1, x2 = p1 * n1, p2 * n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return np.nan, np.nan
    z = (p1 - p2) / se
    pval = 2 * (1 - norm.cdf(abs(z)))
    return z, pval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', required=True)
    ap.add_argument('--lookback', type=int, default=150)
    ap.add_argument('--horizon', type=int, default=40)
    ap.add_argument('--step', type=int, default=20)
    ap.add_argument('--bounce-atr-mult', type=float, default=1.0)
    ap.add_argument('--break-atr-mult', type=float, default=0.5)
    ap.add_argument('--break-confirm-bars', type=int, default=2)
    ap.add_argument('--reaction-bars', type=int, default=10)
    ap.add_argument('--recovery-bars', type=int, default=10)
    ap.add_argument('--out', default='backtest_reference_results.csv')
    ap.add_argument('--summary-out', default='backtest_reference_summary.csv')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    all_results = []
    for path in args.files:
        res = run_file(path, args.lookback, args.horizon, args.step,
                        args.bounce_atr_mult, args.break_atr_mult, args.break_confirm_bars,
                        args.reaction_bars, args.recovery_bars, verbose=args.verbose)
        print(f"  -> {len(res)} candidate levels evaluated")
        all_results.append(res)

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(args.out, index=False)
    per_file, overall = summarize(results)
    per_file.to_csv(args.summary_out, index=False)

    pd.set_option('display.width', 160)
    print("\n=== Per instrument/timeframe ===")
    print(per_file.round(3).to_string(index=False))

    print("\n=== Overall, with significance vs Random (Bonferroni-corrected for", len(METHOD_NAMES) - 1, "comparisons) ===")
    rand_row = overall[overall['method'] == 'Random'].iloc[0]
    p_rand, n_rand = rand_row['support_accuracy'], rand_row['n_touched']
    bonferroni_alpha = 0.05 / (len(METHOD_NAMES) - 1)
    print(f"Random: accuracy={p_rand:.4f}, n_touched={n_rand:.0f}")
    print(f"Bonferroni-corrected significance threshold: p < {bonferroni_alpha:.4f}\n")
    for _, row in overall.iterrows():
        if row['method'] == 'Random':
            continue
        z, pval = zscore_test(row['support_accuracy'], row['n_touched'], p_rand, n_rand)
        sig = 'YES' if pval < bonferroni_alpha else ('borderline' if pval < 0.05 else 'no')
        print(f"{row['method']:<16} accuracy={row['support_accuracy']:.4f}  n={row['n_touched']:.0f}  "
              f"z={z:.2f}  p={pval:.4f}  significant={sig}")


if __name__ == '__main__':
    main()
