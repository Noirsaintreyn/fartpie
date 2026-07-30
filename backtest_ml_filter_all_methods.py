"""
Extends backtest_ml_filter.py to ALL currently-kept level methods (GMM, TDA,
HDBSCAN, OPTICS, KDE, MeanShift, Isolation-Forest), not just GMM/TDA/VWAP -
to check whether the same ML filter (VWAP-distance normalized by a GARCH
volatility forecast, plus ATR-distance) is worth adding to the ones it
hasn't been validated on yet, before wiring anything else into production.

Same design as backtest_ml_filter.py: walk-forward feature/label collection,
chronological 70/30 split, logistic regression filter, compared against the
random baseline with the same z-test/Bonferroni discipline.
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
from sklearn.linear_model import LogisticRegression

from backtest_levels import load_csv, infer_timeframe, compute_atr, evaluate_level

with contextlib.redirect_stdout(io.StringIO()):
    import backend

_RANDOM_RNG = np.random.default_rng(42)


def collect_events(path, lookback, horizon, step,
                    bounce_atr_mult, break_atr_mult, break_confirm_bars,
                    reaction_bars, recovery_bars, verbose=False):
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
        win_h, win_l, win_c = highs[t - lookback:t], lows[t - lookback:t], closes[t - lookback:t]
        win_v, win_dt = volumes[t - lookback:t], datetimes[t - lookback:t]
        current_price = closes[t - 1]
        atr = compute_atr(win_h, win_l, win_c)
        if atr <= 0:
            continue

        returns_pct = np.diff(np.log(win_c)) * 100
        with contextlib.redirect_stdout(io.StringIO()):
            garch = backend.fit_garch_model(returns_pct)
        if garch is None:
            continue
        vol_forecast = garch['forecast_vol'][0] / 100.0 * current_price  # back to price units, 1-step ahead

        with contextlib.redirect_stdout(io.StringIO()):
            vwap_result = backend.calculate_vwap(win_h, win_l, win_c, win_v, timestamps=win_dt)
        if vwap_result is None:
            continue
        vwap = vwap_result['vwap']

        candidates = [(vwap, 'VWAP')]
        for level_name, fn, kwargs in [
            ('GMM', backend.calculate_gmm_levels, {}),
            ('TDA', backend.persistent_homology_levels, {'max_levels': 8}),
            ('HDBSCAN', backend.calculate_hdbscan_levels, {'timeframe': timeframe}),
            ('OPTICS', backend.optics_multi_density_levels, {}),
            ('KDE', backend.kde_based_levels, {}),
            ('MeanShift', backend.calculate_meanshift_levels, {}),
            ('Isolation-Forest', backend.find_pivot_anomalies, {}),
        ]:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    lvls = fn(win_h, win_l, win_c, **kwargs)
            except Exception:
                lvls = []
            candidates += [(lvl['price'], level_name) for lvl in (lvls or [])]

        fwd_h, fwd_l, fwd_c = highs[t:t + horizon], lows[t:t + horizon], closes[t:t + horizon]

        for price, level_type in candidates:
            if price is None or price <= 0:
                continue
            side = 'support' if price < current_price else 'resistance'
            outcome = evaluate_level(price, side, fwd_h, fwd_l, fwd_c, atr,
                                      bounce_atr_mult, break_atr_mult, break_confirm_bars,
                                      reaction_bars, recovery_bars)
            if outcome is None:
                continue  # only touched events are useful for the filter model
            rows.append({
                'instrument': instrument, 'timeframe': timeframe, 'level_type': level_type,
                't': t, 'price': price,
                'vwap_distance_norm': (price - vwap) / (vol_forecast + 1e-9),
                'vol_forecast_pct_of_price': vol_forecast / current_price,
                'atr_distance': (price - current_price) / atr,
                'bounced': int(outcome['bounced']),
            })

        if verbose and wi % 500 == 0:
            print(f"  window {wi}/{len(starts)} ({time.time()-t0:.1f}s elapsed)")

    return pd.DataFrame(rows)


def zscore_test(p1, n1, p2, n2):
    x1, x2 = p1 * n1, p2 * n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return np.nan, np.nan
    z = (p1 - p2) / se
    return z, 2 * (1 - norm.cdf(abs(z)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', required=True)
    ap.add_argument('--lookback', type=int, default=150)
    ap.add_argument('--horizon', type=int, default=40)
    ap.add_argument('--step', type=int, default=20)
    ap.add_argument('--train-frac', type=float, default=0.7)
    ap.add_argument('--bounce-atr-mult', type=float, default=1.0)
    ap.add_argument('--break-atr-mult', type=float, default=0.5)
    ap.add_argument('--break-confirm-bars', type=int, default=2)
    ap.add_argument('--reaction-bars', type=int, default=10)
    ap.add_argument('--recovery-bars', type=int, default=10)
    ap.add_argument('--random-accuracy', type=float, default=0.4227)
    ap.add_argument('--random-n', type=int, default=29257)
    ap.add_argument('--out', default='backtest_ml_filter_all_methods_events.csv')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    all_events = []
    for path in args.files:
        ev = collect_events(path, args.lookback, args.horizon, args.step,
                             args.bounce_atr_mult, args.break_atr_mult, args.break_confirm_bars,
                             args.reaction_bars, args.recovery_bars, verbose=args.verbose)
        print(f"  -> {len(ev)} touched candidate events collected")
        all_events.append(ev)

    events = pd.concat(all_events, ignore_index=True)
    events.to_csv(args.out, index=False)

    # chronological split per instrument+timeframe file (not globally, so each
    # file's train/holdout boundary is its own timeline)
    events['is_train'] = False
    for (inst, tf), g in events.groupby(['instrument', 'timeframe']):
        cutoff_t = g['t'].quantile(args.train_frac)
        events.loc[g.index, 'is_train'] = g['t'] <= cutoff_t

    train = events[events['is_train']]
    holdout = events[~events['is_train']]
    print(f"\nTrain events: {len(train)}, Holdout events: {len(holdout)}")

    feature_cols = ['vwap_distance_norm', 'vol_forecast_pct_of_price', 'atr_distance']
    X_train = pd.get_dummies(train[feature_cols + ['level_type']], columns=['level_type'])
    X_holdout = pd.get_dummies(holdout[feature_cols + ['level_type']], columns=['level_type'])
    X_holdout = X_holdout.reindex(columns=X_train.columns, fill_value=0)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train['bounced'])
    holdout = holdout.copy()
    holdout['pred_proba'] = model.predict_proba(X_holdout)[:, 1]

    print("\nLogistic regression coefficients:")
    for name, coef in zip(X_train.columns, model.coef_[0]):
        print(f"  {name:<28} {coef:+.4f}")

    print(f"\n=== Per level type: unconditional vs ML-filtered (top 50% by predicted P(bounce)) ===")
    p_rand, n_rand = args.random_accuracy, args.random_n
    n_comparisons = 8  # VWAP, GMM, TDA, HDBSCAN, OPTICS, KDE, MeanShift, Isolation-Forest
    alpha = 0.05 / n_comparisons

    for level_type, g in holdout.groupby('level_type'):
        n_total = len(g)
        unconditional_acc = g['bounced'].mean()
        median_score = g['pred_proba'].median()
        filtered = g[g['pred_proba'] >= median_score]
        filtered_acc = filtered['bounced'].mean()
        n_filtered = len(filtered)

        z_uncond, p_uncond = zscore_test(unconditional_acc, n_total, p_rand, n_rand)
        z_filt, p_filt = zscore_test(filtered_acc, n_filtered, p_rand, n_rand)

        print(f"\n{level_type} (holdout n={n_total}):")
        print(f"  Unconditional accuracy: {unconditional_acc:.4f}  (z={z_uncond:.2f}, p={p_uncond:.5f} vs random)")
        print(f"  ML-filtered accuracy:   {filtered_acc:.4f}  n={n_filtered}  (z={z_filt:.2f}, p={p_filt:.5f} vs random, "
              f"Bonferroni threshold={alpha:.5f})")
        print(f"  Filter lift over unconditional: {filtered_acc - unconditional_acc:+.4f}")


if __name__ == '__main__':
    main()
