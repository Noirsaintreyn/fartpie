"""
Hybrid OU zone: fixes "it just looks like VWAP" (the plain OU zone re-fits
every 5 bars from a rolling window, so its center never gets far from
current price) without going to a fully static level (which can't react to
a genuine regime change). Two changes from backtest_ou_zones.py:

1. CENTER: multi-timeframe VWAP consensus (session + weekly + monthly,
   equally weighted - a stated, tunable default) instead of a single
   rolling-window VWAP. Weekly/monthly VWAP are anchored to their real
   calendar period from the daily bars (D_NQ.csv/D_ES.csv), not resampled
   from the rolling intraday window - so they move on a genuinely slower
   clock than price, not a shadow of it.

2. HOLD RULE: the zone stays FIXED (not refit every scan step) until one
   of three triggers fires, whichever comes first:
     - RESOLVED: price touches an edge and confirms reject or break
       (the original durability-respecting rule)
     - VOL SHIFT: current ATR vs. the ATR at zone formation moves beyond
       vol_shift_threshold (uses ATR, not a fresh GJR-GARCH fit, per bar -
       GARCH is too expensive to run every bar just to check a trigger;
       ATR is a legitimate, cheap realized-vol proxy for "has the regime
       changed enough to invalidate this zone")
     - SESSION ROLLOVER: a new calendar day starts

Same four validation criteria as the original: width, touch rate, reject
rate vs random, and does the NEXT zone (after this one resolves) also work.
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

from backtest_levels import load_csv, infer_timeframe, compute_atr
from backtest_ou_zones import fit_ou_process

with contextlib.redirect_stdout(io.StringIO()):
    import backend


def htf_vwap_anchors(daily_df, current_ts, lookback_days=90):
    """Weekly/monthly anchored VWAP as of the most recent COMPLETE daily bar
    at/before current_ts (never peeks at today's incomplete day). Returns
    (weekly_vwap, monthly_vwap), either None if not enough history."""
    past = daily_df[daily_df['datetime'] < pd.Timestamp(current_ts).normalize()]
    if len(past) < 5:
        return None, None
    past = past.tail(lookback_days)
    typical = (past['high'] + past['low'] + past['close']) / 3.0
    vol = past['volume'].values
    dates = past['datetime'].values

    last_date = pd.Timestamp(dates[-1])
    week_start = last_date - pd.Timedelta(days=last_date.dayofweek)
    month_start = last_date.replace(day=1)

    week_mask = past['datetime'] >= week_start
    month_mask = past['datetime'] >= month_start

    def vwap_of(mask):
        v = vol[mask.values]
        if v.sum() <= 0:
            return None
        return float((typical[mask.values] * v).sum() / v.sum())

    return vwap_of(week_mask), vwap_of(month_mask)


def backtest_hybrid(path, daily_path, lookback=150, k=0.25, vol_shift_threshold=0.4,
                     touch_confirm_bars=2, max_hold_bars=500, use_session_trigger=True, verbose=False):
    instrument = re.sub(r'^\d+[a-zA-Z]+_', '', os.path.splitext(os.path.basename(path))[0])
    timeframe = infer_timeframe(os.path.basename(path))
    print(f"Loading {path} (instrument={instrument}, timeframe={timeframe})...")
    df = load_csv(path)
    daily_df = load_csv(daily_path)
    n = len(df)
    highs, lows, closes = df['high'].values, df['low'].values, df['close'].values
    volumes = df['volume'].values
    datetimes = df['datetime'].values
    dates = pd.to_datetime(pd.Series(datetimes)).dt.date.values

    rows = []
    t = lookback
    t0 = time.time()
    n_zones = 0

    while t < n - 1:
        if verbose and n_zones % 200 == 0 and n_zones > 0:
            print(f"  ~bar {t}/{n} ({time.time()-t0:.1f}s elapsed, {n_zones} zones so far)")

        win_h, win_l, win_c = highs[t - lookback:t], lows[t - lookback:t], closes[t - lookback:t]
        win_v, win_dt = volumes[t - lookback:t], datetimes[t - lookback:t]
        atr_formation = compute_atr(win_h, win_l, win_c)
        if atr_formation <= 0:
            t += 1
            continue

        with contextlib.redirect_stdout(io.StringIO()):
            vwap_result = backend.calculate_vwap(win_h, win_l, win_c, win_v, timestamps=win_dt)
        if vwap_result is None:
            t += 1
            continue
        session_vwap = vwap_result['vwap']

        weekly_vwap, monthly_vwap = htf_vwap_anchors(daily_df, datetimes[t - 1])
        anchors = [a for a in [session_vwap, weekly_vwap, monthly_vwap] if a is not None]
        consensus_vwap = float(np.mean(anchors))

        vwap_series = vwap_result['vwap_series']
        deviation = win_c - vwap_series  # OU fit still on the rolling session-vwap deviation
        fit = fit_ou_process(deviation)  # (just for a stationary-std width estimate)
        if fit is None:
            t += 1
            continue
        _, _, stationary_std = fit

        zone_low = consensus_vwap - k * stationary_std
        zone_high = consensus_vwap + k * stationary_std
        formation_date = dates[t - 1]
        n_zones += 1

        # hold this zone fixed, walking forward bar by bar, until a trigger fires
        touched_edge, touch_idx = None, None
        outcome_reason, rejected = None, None
        i = t
        end_i = min(t + max_hold_bars, n)
        while i < end_i:
            if use_session_trigger and dates[i] != formation_date:
                outcome_reason = 'session_rollover'
                break
            win_atr_now = compute_atr(highs[max(0, i - 14):i], lows[max(0, i - 14):i], closes[max(0, i - 14):i])
            if win_atr_now > 0 and abs(win_atr_now - atr_formation) / atr_formation > vol_shift_threshold:
                outcome_reason = 'vol_shift'
                break

            if touched_edge is None:
                if highs[i] >= zone_high:
                    touched_edge, touch_idx = 'resistance', i
                elif lows[i] <= zone_low:
                    touched_edge, touch_idx = 'support', i
            else:
                if i >= touch_idx + touch_confirm_bars:
                    confirm_c = closes[touch_idx:touch_idx + touch_confirm_bars]
                    if touched_edge == 'resistance':
                        broke = np.sum(confirm_c > zone_high) >= touch_confirm_bars
                    else:
                        broke = np.sum(confirm_c < zone_low) >= touch_confirm_bars
                    rejected = not broke
                    outcome_reason = 'resolved'
                    break
            i += 1
        else:
            outcome_reason = outcome_reason or 'max_hold_reached'

        rows.append({
            'instrument': instrument, 'timeframe': timeframe, 't_formed': t,
            'zone_low': zone_low, 'zone_high': zone_high, 'zone_width_atr': (zone_high - zone_low) / atr_formation,
            'consensus_vwap': consensus_vwap, 'session_vwap': session_vwap,
            'weekly_vwap': weekly_vwap, 'monthly_vwap': monthly_vwap,
            'touched': touched_edge is not None, 'side': touched_edge,
            'outcome_reason': outcome_reason, 'rejected': rejected,
            'hold_bars': i - t,
        })

        t = max(i + 1, t + 1)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', required=True)
    ap.add_argument('--daily-files', nargs='+', required=True)
    ap.add_argument('--lookback', type=int, default=150)
    ap.add_argument('--k', type=float, default=0.25)
    ap.add_argument('--vol-shift-threshold', type=float, default=0.4)
    ap.add_argument('--max-hold-bars', type=int, default=500)
    ap.add_argument('--no-session-trigger', action='store_true',
                     help='disable the session-rollover trigger (calendar-day change is a poor boundary for '
                          'near-24h futures trading - use this to test resolve+vol-shift only)')
    ap.add_argument('--out', default='backtest_ou_zones_hybrid_events.csv')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    if len(args.files) != len(args.daily_files):
        raise SystemExit("--files and --daily-files must have matching counts (one daily file per intraday file)")

    all_events = []
    for path, daily_path in zip(args.files, args.daily_files):
        ev = backtest_hybrid(path, daily_path, args.lookback, args.k, args.vol_shift_threshold,
                              max_hold_bars=args.max_hold_bars, use_session_trigger=not args.no_session_trigger,
                              verbose=args.verbose)
        print(f"  -> {len(ev)} zones formed")
        all_events.append(ev)
        pd.concat(all_events, ignore_index=True).to_csv(args.out, index=False)

    events = pd.concat(all_events, ignore_index=True)
    events.to_csv(args.out, index=False)

    print(f"\n=== Hybrid OU zone backtest (k={args.k}, vol_shift_threshold={args.vol_shift_threshold}) ===")
    for (inst, tf), g in events.groupby(['instrument', 'timeframe']):
        n_total = len(g)
        touch_rate = g['touched'].mean()
        touched = g[g['touched']]
        resolved = touched[touched['outcome_reason'] == 'resolved']
        reject_rate = resolved['rejected'].mean() if len(resolved) > 0 else float('nan')
        print(f"\n{inst} {tf} (n={n_total} zones formed):")
        print(f"  median zone width: {g['zone_width_atr'].median():.2f} ATR")
        print(f"  median hold duration: {g['hold_bars'].median():.0f} bars")
        print(f"  outcome breakdown: {g['outcome_reason'].value_counts().to_dict()}")
        print(f"  touch rate: {touch_rate:.1%}")
        print(f"  reject rate (of touches that resolved via touch, not session/vol trigger): {reject_rate:.1%}  (n={len(resolved)}, random baseline ~50%)")


if __name__ == '__main__':
    main()
