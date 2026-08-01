"""
Context-aware extension of the standalone OU zone model (backtest_ou_zones.py)
- NOT part of the v2/v3 level-detector filter (dropped). Same zone
construction (OU fit on price-minus-VWAP, k=0.25 stationary-std half-width,
same touch/reject definition), but tests whether adding regime context -
Hurst, HMM state confidence/flip, GJR-GARCH forecast, EGARCH forecast
(new - never implemented before, log-variance model, asymmetric like GJR
but via a different functional form), Garman-Klass realized vol - lets a
logistic regression discriminate WHICH touches reject vs break, beyond the
flat ~61-64% rate the context-free zone already shows on its own.

This is a genuinely separate question from "does the zone work at all"
(already answered yes, independently). It's "does knowing the regime at
zone-formation time predict whether THIS PARTICULAR touch will reject."
Walk-forward validated the same way as everything else in this project:
expanding-window folds, pooled OOS predictions, compare discrimination
against the flat baseline.
"""
import argparse
import contextlib
import io
import os
import re
import time

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from backtest_levels import load_csv, infer_timeframe, compute_atr
from backtest_ou_zones import fit_ou_process

with contextlib.redirect_stdout(io.StringIO()):
    import backend


def fit_egarch_vol_forecast_pct(returns_pct):
    """EGARCH(1,1) 1-step vol forecast - models log-variance directly (no
    positivity constraint needed, unlike GARCH/GJR), asymmetry via a
    different functional form than GJR's o=1 term. Genuine alternative to
    compare against GJR-GARCH, not assumed to be better a priori."""
    try:
        if len(returns_pct) < 50:
            return None
        model = arch_model(returns_pct, vol='EGARCH', p=1, o=1, q=1, rescale=False)
        result = model.fit(disp='off', show_warning=False)
        forecast = result.forecast(horizon=1)
        vol_pct = float(np.sqrt(forecast.variance.values[-1, 0]))
        # EGARCH forecasts log-variance then exponentiates - an unstable
        # fit can blow this up to nonsensical values (observed up to 1e154
        # in practice). Treat anything beyond a sane vol range as a failed
        # fit rather than feeding garbage into the model.
        if not np.isfinite(vol_pct) or vol_pct > 50 or vol_pct <= 0:
            return None
        return vol_pct
    except Exception:
        return None


def hmm_regime_change_features(closes, n_states=3, flip_lookback=5):
    if not getattr(backend, 'HMMLEARN_AVAILABLE', False) or len(closes) < 60:
        return None, None
    returns = np.diff(np.log(closes)).reshape(-1, 1)
    try:
        model = backend.GaussianHMM(n_components=n_states, covariance_type='diag',
                                     n_iter=50, random_state=42)
        model.fit(returns)
        states = model.predict(returns)
        post = model.predict_proba(returns)
        last_conf = float(post[-1, states[-1]])
        recent_flip = int(states[-1] != states[max(0, len(states) - 1 - flip_lookback)])
        return last_conf, recent_flip
    except Exception:
        return None, None


def collect_ou_zone_context_events(path, lookback=150, horizon=40, step=20, k=0.25,
                                    touch_confirm_bars=2, verbose=False):
    instrument = re.sub(r'^\d+[a-zA-Z]+_', '', os.path.splitext(os.path.basename(path))[0])
    timeframe = infer_timeframe(os.path.basename(path))
    print(f"Loading {path} (instrument={instrument}, timeframe={timeframe})...")
    df = load_csv(path)
    n = len(df)
    opens, highs, lows, closes = df['open'].values, df['high'].values, df['low'].values, df['close'].values
    volumes = df['volume'].values
    datetimes = df['datetime'].values

    starts = list(range(lookback, n - horizon, step))
    rows = []
    t0 = time.time()
    for wi, t in enumerate(starts):
        if verbose and wi % 200 == 0:
            print(f"  window {wi}/{len(starts)} ({time.time()-t0:.1f}s elapsed, {len(rows)} touch-events so far)")

        win_o = opens[t - lookback:t]
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

        # which edge(s), if any, are ahead of current price
        edges = []
        if current_price < zone_high:
            edges.append((zone_high, 'resistance'))
        if current_price > zone_low:
            edges.append((zone_low, 'support'))
        if not edges:
            continue

        fwd_h, fwd_l, fwd_c = highs[t:t + horizon], lows[t:t + horizon], closes[t:t + horizon]
        any_touch = False
        touch_results = []
        for edge, side in edges:
            touched, touch_idx = False, None
            for i in range(len(fwd_c)):
                if side == 'resistance' and fwd_h[i] >= edge:
                    touched, touch_idx = True, i
                    break
                if side == 'support' and fwd_l[i] <= edge:
                    touched, touch_idx = True, i
                    break
            if touched:
                any_touch = True
                if side == 'resistance':
                    broke = np.sum(fwd_c[touch_idx:touch_idx + touch_confirm_bars] > edge) >= touch_confirm_bars
                else:
                    broke = np.sum(fwd_c[touch_idx:touch_idx + touch_confirm_bars] < edge) >= touch_confirm_bars
                touch_results.append((side, not broke))

        if not any_touch:
            continue

        # context features computed once per window (at zone-formation
        # time t, PIT-safe - never using bars beyond t)
        returns_pct = np.diff(np.log(win_c)) * 100
        with contextlib.redirect_stdout(io.StringIO()):
            gjr_vol_pct = backend.fit_gjr_garch_vol_forecast_pct(returns_pct)
            egarch_vol_pct = fit_egarch_vol_forecast_pct(returns_pct)
        try:
            gk_vol_pct = backend.garman_klass_daily_volatility(win_o, win_h, win_l, win_c) * 100
        except Exception:
            gk_vol_pct = np.nan
        try:
            hurst = backend.calculate_hurst_exponent(win_c)['hurst']
        except Exception:
            hurst = np.nan
        with contextlib.redirect_stdout(io.StringIO()):
            hmm_conf, hmm_flip = hmm_regime_change_features(win_c)

        for side, rejected in touch_results:
            rows.append({
                'instrument': instrument, 'timeframe': timeframe, 't': t, 'side': side,
                'theta': theta, 'stationary_std': stationary_std,
                'hurst': hurst,
                'hmm_state_confidence': hmm_conf if hmm_conf is not None else 0.5,
                'hmm_recent_flip': hmm_flip if hmm_flip is not None else 0,
                'gjr_vol_pct': gjr_vol_pct, 'egarch_vol_pct': egarch_vol_pct,
                'garman_klass_vol_pct': gk_vol_pct,
                'rejected': int(rejected),
            })

    return pd.DataFrame(rows)


def zscore_test(p1, n1, p2, n2):
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan
    x1, x2 = p1 * n1, p2 * n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return np.nan, np.nan
    z = (p1 - p2) / se
    return z, 2 * (1 - norm.cdf(abs(z)))


FEATURE_COLS = ['theta', 'stationary_std', 'hurst', 'hmm_state_confidence',
                'hmm_recent_flip', 'gjr_vol_pct', 'egarch_vol_pct', 'garman_klass_vol_pct']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', required=True)
    ap.add_argument('--lookback', type=int, default=150)
    ap.add_argument('--horizon', type=int, default=40)
    ap.add_argument('--step', type=int, default=20)
    ap.add_argument('--k', type=float, default=0.25)
    ap.add_argument('--k-folds', type=int, default=5)
    ap.add_argument('--out', default='backtest_ou_zones_context_events.csv')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    all_events = []
    for path in args.files:
        ev = collect_ou_zone_context_events(path, args.lookback, args.horizon, args.step, args.k, verbose=args.verbose)
        print(f"  -> {len(ev)} touch-events collected")
        all_events.append(ev)
        pd.concat(all_events, ignore_index=True).to_csv(args.out, index=False)

    events = pd.concat(all_events, ignore_index=True)
    events = events.dropna(subset=FEATURE_COLS)
    events.to_csv(args.out, index=False)
    print(f"\nTotal touch-events with complete context: {len(events)}")

    print("\n=== Flat (context-free) reject rate, for reference ===")
    for (inst, tf), g in events.groupby(['instrument', 'timeframe']):
        print(f"  {inst} {tf}: {g['rejected'].mean():.1%}  (n={len(g)})")

    print(f"\n=== Walk-forward: does context predict WHICH touches reject? ({args.k_folds-1} folds/file) ===")
    oos_rows = []
    for (inst, tf), g in events.groupby(['instrument', 'timeframe']):
        g = g.sort_values('t').reset_index(drop=True)
        fold_edges = np.quantile(g['t'], np.linspace(0, 1, args.k_folds + 1))
        fold_ids = np.digitize(g['t'], fold_edges[1:-1], right=True)
        g = g.assign(fold=fold_ids)
        for k in range(1, args.k_folds):
            train = g[g['fold'] < k]
            test = g[g['fold'] == k]
            if len(train) < 100 or len(test) < 30:
                continue
            # theta (~0.004-0.37) and stationary_std (~2-800, raw price
            # units) differ from the other features and each other by
            # orders of magnitude - standardize before fitting, or the
            # regularization penalty hits large-scale features far harder
            # than small-scale ones and can degrade to a near-constant fit
            scaler = StandardScaler()
            X_train = scaler.fit_transform(train[FEATURE_COLS])
            X_test = scaler.transform(test[FEATURE_COLS])
            model = LogisticRegression(max_iter=2000)
            model.fit(X_train, train['rejected'])
            test = test.copy()
            test['pred_proba'] = model.predict_proba(X_test)[:, 1]
            test['instrument'], test['timeframe'] = inst, tf
            oos_rows.append(test)

    if not oos_rows:
        print("Not enough data for walk-forward folds.")
        return
    oos = pd.concat(oos_rows, ignore_index=True)
    oos.to_csv('backtest_ou_zones_context_oos.csv', index=False)

    for (inst, tf), g in oos.groupby(['instrument', 'timeframe']):
        flat_rate = g['rejected'].mean()
        median_score = g['pred_proba'].median()
        top_half = g[g['pred_proba'] >= median_score]
        bottom_half = g[g['pred_proba'] < median_score]
        z, p = zscore_test(top_half['rejected'].mean(), len(top_half), bottom_half['rejected'].mean(), len(bottom_half))
        print(f"\n{inst} {tf} (pooled OOS n={len(g)}):")
        print(f"  flat reject rate: {flat_rate:.1%}")
        print(f"  top-half-by-context reject rate: {top_half['rejected'].mean():.1%} (n={len(top_half)})")
        print(f"  bottom-half-by-context reject rate: {bottom_half['rejected'].mean():.1%} (n={len(bottom_half)})")
        print(f"  spread significance: z={z:.2f}, p={p:.4f}")


if __name__ == '__main__':
    main()
