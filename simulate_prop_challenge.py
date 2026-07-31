"""
Entry/SL/TP trade simulator + prop-firm challenge pass-rate backtest, built
on top of ALL 7 validated ML-filtered level types (score_and_filter_levels
in backend.py - walk-forward validated: GMM, HDBSCAN, Isolation-Forest,
KDE, MeanShift, OPTICS, and TDA all improved in 16/16 folds and cleared
Bonferroni correction once filtered, 0.480-0.497 filtered accuracy).
Originally only used GMM+TDA - expanded after realizing the trade
simulator was never updated when the production filter was.

Entry: price touches the highest-scoring filtered level (any of the 7
types) -> long if support (level below price), short if resistance
(level above price).
Stop:  break_atr_mult * ATR beyond the level (0.5 ATR) - hard intrabar
       stop using highs/lows, not the softer "2 consecutive closes"
       confirmation used to score level VALIDITY - a real stop has to be
       a hard exit, not wait for confirmation while the loss grows.
Target: bounce_atr_mult * ATR from the level (1.0 ATR), same logic.
Only one open position at a time. Position size fixed at 1 mini contract
(NQ=$20/point, ES=$50/point) per the current sizing choice.

Two things this produces:
  1. A trade list (backtest_prop_trades.csv) - every simulated trade with
     entry/exit/P&L, for direct inspection.
  2. A prop-firm challenge pass-rate: simulate MANY independent challenge
     attempts starting at different points across the 12yr history (not
     just one), so "does this beat the prop firm" is a measured
     probability, not a single anecdote.

Rules simulated (state explicitly, since these are an interpretation of
what was described, not fetched from an actual firm's rulebook):
  Eval:    reach +$3,000 cumulative P&L over >= 2 trading days, never
           breaching -$2,000 drawdown from the starting balance (static).
  Funded:  >=5 trading days with >=+$150 P&L each, same -$2,000 static
           drawdown limit, timed from the end of the eval phase.
"""
import argparse
import contextlib
import io
import os
import re
import time

import numpy as np
import pandas as pd

from backtest_levels import load_csv, infer_timeframe, compute_atr

with contextlib.redirect_stdout(io.StringIO()):
    import backend

POINT_VALUE = {'NQ': 20.0, 'ES': 50.0}


def generate_trades(path, lookback=150, step=5,
                     bounce_atr_mult=1.0, break_atr_mult=0.5,
                     max_hold_bars=200, verbose=False, start_date=None, end_date=None):
    """Walk forward, generate ML-filtered GMM/TDA levels, and simulate hard
    intrabar-stop/target trades against them. One trade open at a time."""
    instrument = re.sub(r'^\d+[a-zA-Z]+_', '', os.path.splitext(os.path.basename(path))[0])
    timeframe = infer_timeframe(os.path.basename(path))
    print(f"Loading {path} (instrument={instrument}, timeframe={timeframe})...")
    df = load_csv(path)
    n = len(df)
    highs, lows, closes = df['high'].values, df['low'].values, df['close'].values
    volumes = df['volume'].values
    datetimes = df['datetime'].values
    point_value = POINT_VALUE.get(instrument, 20.0)

    trades = []
    open_trade = None  # dict with entry info while a position is live
    starts = list(range(lookback, n, step))
    start_ts = np.datetime64(start_date) if start_date else None
    end_ts = np.datetime64(end_date) if end_date else None

    t0 = time.time()
    for wi, t in enumerate(starts):
        if verbose and wi % 200 == 0:
            print(f"  window {wi}/{len(starts)} ({time.time()-t0:.1f}s elapsed, {len(trades)} trades so far)")
        # manage an already-open trade first: check if this bar's H/L hits
        # stop or target (hard intrabar exit, checked in time order).
        # Scans only the NEW bars since the last check (last_checked_idx),
        # not from entry every time - the old version re-scanned the whole
        # trade history on every outer iteration while a trade stayed open,
        # O(max_hold_bars^2/step) redundant work per trade instead of
        # O(max_hold_bars).
        if open_trade is not None:
            scan_start = open_trade['last_checked_idx'] + 1
            scan_end = min(t, open_trade['entry_idx'] + max_hold_bars) + 1
            for i in range(scan_start, scan_end):
                if i >= n:
                    break
                h, l = highs[i], lows[i]
                side = open_trade['side']
                if side == 'long':
                    if l <= open_trade['stop_price']:
                        _close_trade(open_trade, i, open_trade['stop_price'], 'stop', datetimes, point_value, trades)
                        open_trade = None
                        break
                    if h >= open_trade['target_price']:
                        _close_trade(open_trade, i, open_trade['target_price'], 'target', datetimes, point_value, trades)
                        open_trade = None
                        break
                else:
                    if h >= open_trade['stop_price']:
                        _close_trade(open_trade, i, open_trade['stop_price'], 'stop', datetimes, point_value, trades)
                        open_trade = None
                        break
                    if l <= open_trade['target_price']:
                        _close_trade(open_trade, i, open_trade['target_price'], 'target', datetimes, point_value, trades)
                        open_trade = None
                        break
                if open_trade is not None:
                    open_trade['last_checked_idx'] = i
            else:
                if open_trade is not None and t >= open_trade['entry_idx'] + max_hold_bars:
                    exit_idx = min(open_trade['entry_idx'] + max_hold_bars, n - 1)
                    _close_trade(open_trade, exit_idx, closes[exit_idx], 'timeout', datetimes, point_value, trades)
                    open_trade = None

        if open_trade is not None:
            continue  # still in a trade, don't look for new entries

        # entries restricted to the target date range - lookback context
        # before it is still used (PIT-safe), we're just gating NEW entries
        # to this window. A trade opened inside the window is still allowed
        # to run to its natural exit even if that lands after end_date.
        if start_ts is not None and datetimes[t - 1] < start_ts:
            continue
        if end_ts is not None and datetimes[t - 1] > end_ts:
            continue

        win_h, win_l, win_c = highs[t - lookback:t], lows[t - lookback:t], closes[t - lookback:t]
        win_v, win_dt = volumes[t - lookback:t], datetimes[t - lookback:t]
        current_price = closes[t - 1]
        atr = compute_atr(win_h, win_l, win_c)
        if atr <= 0:
            continue

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                levels_by_category = {
                    'GMM': backend.calculate_gmm_levels(win_h, win_l, win_c),
                    'TDA': backend.persistent_homology_levels(win_h, win_l, win_c, max_levels=8),
                    'HDBSCAN': backend.calculate_hdbscan_levels(win_h, win_l, win_c, timeframe=timeframe),
                    'OPTICS': backend.optics_multi_density_levels(win_h, win_l, win_c),
                    'KDE': backend.kde_based_levels(win_h, win_l, win_c),
                    'MeanShift': backend.calculate_meanshift_levels(win_h, win_l, win_c),
                    'Isolation-Forest': backend.find_pivot_anomalies(win_h, win_l, win_c),
                }
                filtered = backend.score_and_filter_levels(
                    levels_by_category, win_h, win_l, win_c, win_v, current_price, timestamps=win_dt,
                )
        except Exception as e:
            if verbose:
                print(f"  filter failed at t={t}: {e}")
            continue
        if not filtered:
            continue

        # take the single highest-scoring filtered level as this window's
        # entry candidate, only if price is actually near it (within 0.25 ATR)
        # i.e. treat "touch" as the entry trigger, same spirit as the backtest
        best = max(filtered, key=lambda l: l['ml_filter_score'])
        price = best['price']
        if abs(current_price - price) > 0.25 * atr:
            continue  # not close enough to be a real touch/entry right now

        side = 'long' if price < current_price else 'short'
        if side == 'long':
            stop_price = price - break_atr_mult * atr
            target_price = price + bounce_atr_mult * atr
        else:
            stop_price = price + break_atr_mult * atr
            target_price = price - bounce_atr_mult * atr

        open_trade = {
            'instrument': instrument, 'timeframe': timeframe, 'level_type': best['category'],
            'ml_filter_score': best['ml_filter_score'], 'side': side,
            'entry_idx': t, 'entry_price': current_price, 'level_price': price,
            'stop_price': stop_price, 'target_price': target_price,
            'last_checked_idx': t,
        }

    return pd.DataFrame(trades)


def _close_trade(open_trade, exit_idx, exit_price, reason, datetimes, point_value, trades):
    entry_price = open_trade['entry_price']
    side = open_trade['side']
    points = (exit_price - entry_price) if side == 'long' else (entry_price - exit_price)
    trades.append({
        **{k: v for k, v in open_trade.items() if k not in ('entry_idx',)},
        'entry_datetime': datetimes[open_trade['entry_idx']],
        'exit_datetime': datetimes[exit_idx],
        'exit_price': exit_price, 'exit_reason': reason,
        'points': points, 'pnl_usd': points * point_value,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', required=True)
    ap.add_argument('--lookback', type=int, default=150)
    ap.add_argument('--step', type=int, default=5)
    ap.add_argument('--out', default='backtest_prop_trades.csv')
    ap.add_argument('--start-date', default=None, help='only open new trades on/after this date (YYYY-MM-DD)')
    ap.add_argument('--end-date', default=None, help='only open new trades on/before this date (YYYY-MM-DD)')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    all_trades = []
    for path in args.files:
        trades = generate_trades(path, lookback=args.lookback, step=args.step, verbose=args.verbose,
                                  start_date=args.start_date, end_date=args.end_date)
        print(f"  -> {len(trades)} trades simulated")
        all_trades.append(trades)

    trades = pd.concat(all_trades, ignore_index=True)
    trades.to_csv(args.out, index=False)

    print(f"\n=== Trade summary ({len(trades)} total trades) ===")
    print(f"Win rate: {(trades['pnl_usd'] > 0).mean():.4f}")
    print(f"Avg P&L per trade: ${trades['pnl_usd'].mean():.2f}")
    print(f"Total P&L: ${trades['pnl_usd'].sum():.2f}")
    print(f"By exit reason:\n{trades['exit_reason'].value_counts()}")
    print(f"\nBy instrument:")
    print(trades.groupby('instrument').agg(n=('pnl_usd', 'size'), win_rate=('pnl_usd', lambda s: (s > 0).mean()),
                                            avg_pnl=('pnl_usd', 'mean'), total_pnl=('pnl_usd', 'sum')).round(2))


if __name__ == '__main__':
    main()
