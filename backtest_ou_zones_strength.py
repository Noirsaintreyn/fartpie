"""
Rejection STRENGTH/DURABILITY analysis for the standalone OU zone model -
answers the question: of the touches classified "rejected" under the
narrow 2-bar confirmation rule (backtest_ou_zones.py), how many are a
genuine, decisive reversal vs. a shallow pause that later gets run over
anyway ("price rejects 50 points then returns to destroy the level")?

Same zone construction and touch definition as backtest_ou_zones.py
(k=0.25, price extends out to the OU zone edge). For each touch, adds:

  mfe_atr: max favorable excursion - how far price moved AWAY from the
    edge (back toward fair value) within strength_window bars after
    touch, in ATR units. A real, strong rejection should show a
    meaningful move (>=0.5-1 ATR), not just a 2-bar non-break.
  later_failure: even among touches classified "rejected" by the 2-bar
    rule, did price come back within the REST of the horizon and close
    beyond the edge by touch_confirm_bars anyway - a delayed break the
    narrow rule missed.
  strong_reject: rejected AND mfe_atr >= min_mfe_atr AND NOT later_failure
    - the actual "reliable, noticeable rejection" the zone should be
    trusted for, as opposed to randomness that happens to pass a
    2-bar check.
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
from backtest_ou_zones import fit_ou_process

with contextlib.redirect_stdout(io.StringIO()):
    import backend


def backtest_ou_zone_strength(path, lookback=150, horizon=40, step=20, k=0.25,
                               touch_confirm_bars=2, strength_window=10, min_mfe_atr=0.5,
                               verbose=False):
    instrument = re.sub(r'^\d+[a-zA-Z]+_', '', os.path.splitext(os.path.basename(path))[0])
    timeframe = infer_timeframe(os.path.basename(path))
    print(f"Loading {path} (instrument={instrument}, timeframe={timeframe})...")
    df = load_csv(path)
    n = len(df)
    highs, lows, closes = df['high'].values, df['low'].values, df['close'].values
    volumes = df['volume'].values
    datetimes = df['datetime'].values

    starts = list(range(lookback, n - horizon, step))
    rows = []
    t0 = time.time()
    for wi, t in enumerate(starts):
        if verbose and wi % 500 == 0:
            print(f"  window {wi}/{len(starts)} ({time.time()-t0:.1f}s elapsed, {len(rows)} touch-events so far)")

        win_h, win_l, win_c = highs[t - lookback:t], lows[t - lookback:t], closes[t - lookback:t]
        win_v, win_dt = volumes[t - lookback:t], datetimes[t - lookback:t]
        current_price = closes[t - 1]
        atr = compute_atr(win_h, win_l, win_c)
        if atr <= 0:
            continue

        with contextlib.redirect_stdout(io.StringIO()):
            vwap_result = backend.calculate_vwap(win_h, win_l, win_c, win_v, timestamps=win_dt)
        if vwap_result is None:
            continue
        vwap_series = vwap_result['vwap_series']
        deviation = win_c - vwap_series

        fit = fit_ou_process(deviation)
        if fit is None:
            continue
        theta, mu, stationary_std = fit
        current_vwap = vwap_series[-1]
        zone_low = current_vwap + mu - k * stationary_std
        zone_high = current_vwap + mu + k * stationary_std

        fwd_h, fwd_l, fwd_c = highs[t:t + horizon], lows[t:t + horizon], closes[t:t + horizon]

        for edge, side in [(zone_high, 'resistance'), (zone_low, 'support')]:
            if side == 'resistance' and current_price >= edge:
                continue
            if side == 'support' and current_price <= edge:
                continue

            touched, touch_idx = False, None
            for i in range(len(fwd_c)):
                if side == 'resistance' and fwd_h[i] >= edge:
                    touched, touch_idx = True, i
                    break
                if side == 'support' and fwd_l[i] <= edge:
                    touched, touch_idx = True, i
                    break
            if not touched:
                continue

            confirm_slice_c = fwd_c[touch_idx:touch_idx + touch_confirm_bars]
            if side == 'resistance':
                broke = np.sum(confirm_slice_c > edge) >= touch_confirm_bars
            else:
                broke = np.sum(confirm_slice_c < edge) >= touch_confirm_bars
            rejected = not broke

            # max favorable excursion: furthest price got AWAY from the
            # edge, back toward fair value, within strength_window bars
            # after touch (capped by the available horizon)
            window_end = min(touch_idx + strength_window, len(fwd_c))
            if side == 'resistance':
                mfe = edge - np.min(fwd_l[touch_idx:window_end]) if window_end > touch_idx else 0.0
            else:
                mfe = np.max(fwd_h[touch_idx:window_end]) - edge if window_end > touch_idx else 0.0
            mfe_atr = mfe / atr

            # later failure: among "rejected" touches, does price still
            # come back and close beyond the edge by touch_confirm_bars
            # ANYWHERE in the rest of the horizon (a delayed break the
            # narrow 2-bar rule missed)?
            later_failure = False
            if rejected:
                rest_c = fwd_c[touch_idx + touch_confirm_bars:]
                for j in range(len(rest_c) - touch_confirm_bars + 1):
                    seg = rest_c[j:j + touch_confirm_bars]
                    if side == 'resistance' and np.sum(seg > edge) >= touch_confirm_bars:
                        later_failure = True
                        break
                    if side == 'support' and np.sum(seg < edge) >= touch_confirm_bars:
                        later_failure = True
                        break

            strong_reject = bool(rejected and mfe_atr >= min_mfe_atr and not later_failure)

            rows.append({
                'instrument': instrument, 'timeframe': timeframe, 't': t, 'side': side,
                'zone_width_atr': (zone_high - zone_low) / atr,
                'rejected': rejected, 'mfe_atr': mfe_atr,
                'later_failure': later_failure, 'strong_reject': strong_reject,
            })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', required=True)
    ap.add_argument('--lookback', type=int, default=150)
    ap.add_argument('--horizon', type=int, default=40)
    ap.add_argument('--step', type=int, default=20)
    ap.add_argument('--k', type=float, default=0.25)
    ap.add_argument('--strength-window', type=int, default=10)
    ap.add_argument('--min-mfe-atr', type=float, default=0.5)
    ap.add_argument('--out', default='backtest_ou_zones_strength_events.csv')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    all_events = []
    for path in args.files:
        ev = backtest_ou_zone_strength(path, args.lookback, args.horizon, args.step, args.k,
                                        strength_window=args.strength_window, min_mfe_atr=args.min_mfe_atr,
                                        verbose=args.verbose)
        print(f"  -> {len(ev)} touch-events collected")
        all_events.append(ev)
        pd.concat(all_events, ignore_index=True).to_csv(args.out, index=False)

    events = pd.concat(all_events, ignore_index=True)
    events.to_csv(args.out, index=False)

    print(f"\n=== Rejection strength/durability (k={args.k}, strength_window={args.strength_window} bars, "
          f"min_mfe={args.min_mfe_atr} ATR) ===")
    for (inst, tf), g in events.groupby(['instrument', 'timeframe']):
        n_total = len(g)
        n_rejected = g['rejected'].sum()
        rejected = g[g['rejected']]
        n_later_failure = rejected['later_failure'].sum()
        n_strong = g['strong_reject'].sum()
        print(f"\n{inst} {tf} (n={n_total} touches):")
        print(f"  classified 'rejected' (2-bar rule): {n_rejected} ({n_rejected/n_total:.1%})")
        print(f"  of those, later come back and break anyway: {n_later_failure} ({n_later_failure/max(n_rejected,1):.1%} of rejected)")
        print(f"  median MFE on rejected touches: {rejected['mfe_atr'].median():.2f} ATR")
        print(f"  STRONG reject (real move + holds, not just 2-bar pass): {n_strong} ({n_strong/n_total:.1%} of all touches, "
              f"{n_strong/max(n_rejected,1):.1%} of nominally-rejected touches)")


if __name__ == '__main__':
    main()
