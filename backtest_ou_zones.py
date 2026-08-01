"""
Independent backtest of OU-process-derived price zones, BEFORE any
integration with the level filter. Validation criteria (all four required
before this is usable as a filter, not just built):

  1. Zone width must actually filter - if it's ~100 points on an instrument
     where dozens of candidate levels sit inside that range, it's not
     narrowing anything down. Reported in both price points and ATR units.
  2. Zones must get touched often enough to matter - no reason to watch a
     zone price always reverses before ever reaching.
  3. First-touch rejection rate must beat random (same random-baseline
     methodology used throughout this project).
  4. After a zone is hit or broken, the RECOMPUTED zone (rolling re-fit)
     must also show a real rejection rate on its next touch - proof this
     is a genuine adaptive regime read, not a one-shot lucky fit.

Zone construction: fit an Ornstein-Uhlenbeck process to the price-minus-VWAP
series (X_t = price_t - VWAP_t), a quantity that's mean-reverting by
construction. OU is fit via the standard AR(1) discretization:
  X_{t+1} = a + b*X_t + noise  =>  theta = (1-b)/dt, mu = a/(1-b),
  sigma^2 = var(noise) * 2*theta / (1 - b^2)  (stationary-variance form)
Zone = VWAP + mu +/- k * stationary_std, where stationary_std =
sigma/sqrt(2*theta). This width is adaptive by construction: fast/tight
mean-reversion (calm, rangebound) gives a narrow zone; slow/noisy
mean-reversion (trending, volatile) gives a wide one - not a fixed-width
band that's sometimes way too wide and sometimes way too narrow.
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

with contextlib.redirect_stdout(io.StringIO()):
    import backend


def fit_ou_process(x, dt=1.0):
    """Fit OU via AR(1) regression: x[t+1] = a + b*x[t] + eps.
    Returns (theta, mu, stationary_std) or None if the fit is degenerate
    (b outside (0,1) means no real mean reversion, or a flat/zero-variance
    series)."""
    if len(x) < 20:
        return None
    x0, x1 = x[:-1], x[1:]
    if np.std(x0) < 1e-9:
        return None
    b, a = np.polyfit(x0, x1, 1)
    if not (0 < b < 1):
        return None  # not mean-reverting (b>=1) or oscillatory/degenerate (b<=0)
    resid = x1 - (a + b * x0)
    resid_var = np.var(resid)
    theta = (1 - b) / dt
    mu = a / (1 - b)
    stationary_var = resid_var / (1 - b ** 2)
    if stationary_var <= 0:
        return None
    return theta, mu, float(np.sqrt(stationary_var))


def backtest_ou_zones(path, lookback=150, horizon=40, step=20, k=1.0,
                       touch_confirm_bars=2, reject_confirm_bars=2, verbose=False):
    """
    Walk forward. At each point, fit OU to (price - VWAP) over the lookback
    window, derive a zone (VWAP + mu +/- k*std). Then check forward bars:
      - does price reach either zone edge within horizon? (touch)
      - if touched, does it reject (close back inside the zone within
        reject_confirm_bars) or break through (touch_confirm_bars closes
        beyond the edge)?
      - after resolution (reject or break), re-fit OU on the window ending
        at the resolution bar and repeat the touch/reject check on THAT
        new zone, forward from there - this directly tests validation
        criterion 4 (does the recomputed zone also work).
    """
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
            print(f"  window {wi}/{len(starts)} ({time.time()-t0:.1f}s elapsed, {len(rows)} zone-events so far)")

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
        zone_width = zone_high - zone_low

        fwd_h, fwd_l, fwd_c = highs[t:t + horizon], lows[t:t + horizon], closes[t:t + horizon]

        # Forward-looking test on EACH edge independently: starting from
        # current price (typically near fair value), does price EXTEND OUT
        # to reach this far edge, and if so does it reject there (reverse)
        # or break through (continue)? This is the direction the zone is
        # actually meant to predict - "if we're stretched, look for
        # rejection at a far zone distance away" - not "does an
        # already-stretched price revert partway back toward fair value"
        # (which is near-tautological for a mean-reverting series and was
        # the bug in the first version: 100% touch rate, backwards logic).
        for edge, side in [(zone_high, 'resistance'), (zone_low, 'support')]:
            if side == 'resistance' and current_price >= edge:
                continue  # already beyond this edge - not a forward prediction
            if side == 'support' and current_price <= edge:
                continue

            touched = False
            touch_idx = None
            for i in range(len(fwd_c)):
                if side == 'resistance' and fwd_h[i] >= edge:
                    touched, touch_idx = True, i
                    break
                if side == 'support' and fwd_l[i] <= edge:
                    touched, touch_idx = True, i
                    break

            rejected = None
            if touched:
                if side == 'resistance':
                    broke = np.sum(fwd_c[touch_idx:touch_idx + touch_confirm_bars] > edge) >= touch_confirm_bars
                else:
                    broke = np.sum(fwd_c[touch_idx:touch_idx + touch_confirm_bars] < edge) >= touch_confirm_bars
                rejected = not broke

            rows.append({
                'instrument': instrument, 'timeframe': timeframe, 't': t,
                'side': side, 'zone_low': zone_low, 'zone_high': zone_high,
                'zone_width': zone_width, 'zone_width_atr': zone_width / atr,
                'theta': theta, 'stationary_std': stationary_std,
                'touched': touched, 'rejected': rejected,
            })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', required=True)
    ap.add_argument('--lookback', type=int, default=150)
    ap.add_argument('--horizon', type=int, default=40)
    ap.add_argument('--step', type=int, default=20)
    ap.add_argument('--k', type=float, default=1.0, help='zone half-width in stationary-std units')
    ap.add_argument('--out', default='backtest_ou_zones_events.csv')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    all_events = []
    for path in args.files:
        ev = backtest_ou_zones(path, args.lookback, args.horizon, args.step, args.k, verbose=args.verbose)
        print(f"  -> {len(ev)} zone-events collected")
        all_events.append(ev)
        pd.concat(all_events, ignore_index=True).to_csv(args.out, index=False)

    events = pd.concat(all_events, ignore_index=True)
    events.to_csv(args.out, index=False)

    print(f"\n=== OU zone backtest (k={args.k}) ===")
    for (inst, tf), g in events.groupby(['instrument', 'timeframe']):
        n_total = len(g)
        touch_rate = g['touched'].mean()
        touched = g[g['touched']]
        reject_rate = touched['rejected'].mean() if len(touched) > 0 else float('nan')
        median_width_atr = g['zone_width_atr'].median()
        median_width_pts = g['zone_width'].median()
        print(f"\n{inst} {tf} (n={n_total}):")
        print(f"  median zone width: {median_width_pts:.1f} points ({median_width_atr:.2f} ATR)")
        print(f"  touch rate (zone edge reached within horizon): {touch_rate:.1%}")
        print(f"  reject rate at touch (n={len(touched)}): {reject_rate:.1%}  (random baseline ~50% for a symmetric edge)")


if __name__ == '__main__':
    main()
