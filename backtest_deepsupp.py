"""
Backtest DeepSupp against the other 9 methods, on a proper chronological
train/holdout split (unlike HDBSCAN/OPTICS/GMM/etc., DeepSupp needs a
persistently trained model, so it can't be refit fresh per window - see
deepsupp.py's docstring).

For each file: train the autoencoder on the first `--train-frac` of history,
then walk-forward evaluate DeepSupp AND all 9 other methods only on windows
in the remaining holdout period, using the identical evaluate_level/ATR
logic as backtest_levels.py. Restricting every method to the same holdout
windows keeps the comparison apples-to-apples - DeepSupp cannot be
legitimately scored on the training period, so nothing else is either.
"""
import argparse
import contextlib
import io
import os
import re
import time

import numpy as np
import pandas as pd

from backtest_levels import (
    METHODS, load_csv, infer_timeframe, compute_atr, evaluate_level,
)
import deepsupp as ds


def run_file(path, lookback, horizon, step, train_frac,
             bounce_atr_mult, break_atr_mult, break_confirm_bars,
             reaction_bars, recovery_bars, train_stride, infer_stride,
             epochs, verbose=False):
    instrument = re.sub(r'^\d+[a-zA-Z]+_', '', os.path.splitext(os.path.basename(path))[0])
    timeframe = infer_timeframe(os.path.basename(path))
    print(f"Loading {path} (instrument={instrument}, timeframe={timeframe})...")
    df = load_csv(path)
    n = len(df)
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    volumes = df['volume'].values

    train_end = int(n * train_frac)
    print(f"  {n} bars total, training DeepSupp on first {train_end} "
          f"({df['datetime'].iloc[0]} -> {df['datetime'].iloc[train_end]})")

    t0 = time.time()
    train_feats = ds.compute_features(closes[:train_end], volumes[:train_end])
    snapshots, _ = ds.build_correlation_snapshots(train_feats, stride=train_stride)
    print(f"  {len(snapshots)} training snapshots built in {time.time()-t0:.1f}s")

    t0 = time.time()
    model = ds.train_autoencoder(snapshots, epochs=epochs, verbose=verbose)
    print(f"  autoencoder trained in {time.time()-t0:.1f}s")

    # holdout-only walk-forward starts: every start must be > train_end so
    # no evaluation window's detection data overlaps what DeepSupp trained
    # on being "the future" relative to that point - the model's weights
    # were fit only on data strictly before train_end, and every evaluated
    # t here is strictly after it.
    starts = [t for t in range(lookback, n - horizon, step) if t > train_end]
    print(f"  {len(starts)} holdout windows to evaluate")

    rows = []
    t0 = time.time()
    for wi, t in enumerate(starts):
        win_h = highs[t - lookback:t]
        win_l = lows[t - lookback:t]
        win_c = closes[t - lookback:t]
        win_v = volumes[t - lookback:t]
        current_price = closes[t - 1]
        atr = compute_atr(win_h, win_l, win_c)
        if atr <= 0:
            continue

        fwd_h = highs[t:t + horizon]
        fwd_l = lows[t:t + horizon]
        fwd_c = closes[t:t + horizon]

        all_methods = dict(METHODS)
        all_methods['DeepSupp'] = None  # handled specially below

        for method_name in list(METHODS.keys()) + ['DeepSupp']:
            try:
                if method_name == 'DeepSupp':
                    levels = ds.deepsupp_levels(model, win_h, win_l, win_c, win_v, stride=infer_stride)
                else:
                    with contextlib.redirect_stdout(io.StringIO()):
                        levels = METHODS[method_name](win_h, win_l, win_c, timeframe)
            except Exception as e:
                if verbose:
                    print(f"  [{method_name}] error at t={t}: {e}")
                continue

            for lvl in levels or []:
                price = lvl.get('price')
                if price is None or price <= 0:
                    continue
                side = 'support' if price < current_price else 'resistance'
                outcome = evaluate_level(
                    price, side, fwd_h, fwd_l, fwd_c, atr,
                    bounce_atr_mult, break_atr_mult, break_confirm_bars,
                    reaction_bars, recovery_bars
                )
                rows.append({
                    'instrument': instrument, 'timeframe': timeframe, 'method': method_name,
                    't': t, 'price': price, 'side': side,
                    'touched': outcome is not None,
                    'bounced': outcome['bounced'] if outcome else None,
                    'broken': outcome['broken'] if outcome else None,
                    'hold_duration': outcome['hold_duration'] if outcome else None,
                    'recovered': outcome['recovered'] if outcome else None,
                })

        if verbose and wi % 50 == 0:
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
            'n_levels_generated': n_levels,
            'n_touched': n_touched,
            'touch_rate': n_touched / n_levels if n_levels else np.nan,
            'support_accuracy': n_bounced / n_touched if n_touched else np.nan,
            'break_rate': n_broken / n_touched if n_touched else np.nan,
            'false_breakout_recovery': n_recovered / n_broken if n_broken else np.nan,
            'avg_hold_duration_bars': touched['hold_duration'].mean() if n_touched else np.nan,
        })
    per_method = results.groupby(['instrument', 'timeframe', 'method']).apply(agg).reset_index()
    overall = results.groupby('method').apply(agg).reset_index()
    return per_method, overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', required=True)
    ap.add_argument('--lookback', type=int, default=150)
    ap.add_argument('--horizon', type=int, default=40)
    ap.add_argument('--step', type=int, default=20)
    ap.add_argument('--train-frac', type=float, default=0.6)
    ap.add_argument('--bounce-atr-mult', type=float, default=1.0)
    ap.add_argument('--break-atr-mult', type=float, default=0.5)
    ap.add_argument('--break-confirm-bars', type=int, default=2)
    ap.add_argument('--reaction-bars', type=int, default=10)
    ap.add_argument('--recovery-bars', type=int, default=10)
    ap.add_argument('--train-stride', type=int, default=2)
    ap.add_argument('--infer-stride', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--out', default='backtest_deepsupp_results.csv')
    ap.add_argument('--summary-out', default='backtest_deepsupp_summary.csv')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    all_results = []
    for path in args.files:
        res = run_file(
            path, args.lookback, args.horizon, args.step, args.train_frac,
            args.bounce_atr_mult, args.break_atr_mult, args.break_confirm_bars,
            args.reaction_bars, args.recovery_bars, args.train_stride, args.infer_stride,
            args.epochs, verbose=args.verbose,
        )
        print(f"  -> {len(res)} candidate levels evaluated")
        all_results.append(res)

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(args.out, index=False)
    per_method, overall = summarize(results)
    per_method.to_csv(args.summary_out, index=False)

    pd.set_option('display.width', 160)
    pd.set_option('display.max_columns', 20)
    print("\n=== Per instrument/timeframe (holdout period only) ===")
    print(per_method.round(3).to_string(index=False))
    print("\n=== Overall (holdout period only, all files combined) ===")
    print(overall.sort_values('support_accuracy', ascending=False).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
