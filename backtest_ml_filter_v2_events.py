"""
v2 of the ML filter: expands the feature set based on the trade-level
diagnosis that the v1 filter (vwap_distance_norm, vol_forecast_pct_of_price,
atr_distance) has no resolving power WITHIN the already-filtered population
(winners and losers had nearly identical scores in the actual trade sim -
see conversation history) and the entry rule ("take the single highest-
scoring candidate") ignores everything else the ensemble already knows.

New features:
  - hurst: Hurst exponent of the window (persistence/mean-reversion regime)
  - hmm_state_confidence: posterior probability of the HMM's assigned
    state for the last bar (low confidence = regime is ambiguous/possibly
    transitioning)
  - hmm_recent_flip: 1 if the HMM-assigned state differs from 5 bars ago,
    0 otherwise (a direct "did the regime just change" signal, not just
    "what regime are we in")
  - gjr_vol_forecast_pct: GJR-GARCH (asymmetric, captures the leverage
    effect that symmetric GARCH misses) 1-step vol forecast, replacing
    plain GARCH for the vwap_distance_norm normalization
  - garman_klass_vol_pct: Garman-Klass realized vol over the window, as an
    independent realized-vol signal alongside the GJR-GARCH forecast
    (kept as a full Realized-GARCH fusion was judged not worth the added
    complexity for this pass - noted honestly, not silently simplified)
  - confluence: number of DISTINCT other level_types with a candidate
    within 0.5*ATR of this candidate's price - "every level printed" used
    as a variable instead of only the single top-scoring candidate
  - gjr_vol_regime_ratio: GJR-GARCH forecast vol / Garman-Klass realized
    vol (both % of price). >1 = forecast vol expanding relative to what
    actually just happened, <1 = contracting - "GARCH action" as a
    regime signal, not just a normalizer
  - vwap_bias_alignment: signed distance (in GJR-GARCH vol units) of
    current price from VWAP, positive if the trade direction agrees with
    the VWAP tilt (long above VWAP / short below) and negative if it
    fades the tilt (long below VWAP / short above - a dip-buy/rip-sell).
    Encodes the requested "bias filter" + "context zones" (stretched vs
    compressed) as one continuous, signed feature instead of a hard gate
    or hand-picked weights, so the model - not an assertion - learns
    whether trend-aligned-stretched or counter-trend-stretched setups
    actually have edge here

vwap_distance_norm now uses the GJR-GARCH forecast (not plain GARCH) as
its normalizer.

Note on "confidence-scaled position sizing" (also requested): that is
implemented downstream in simulate_prop_challenge.py using this model's
predicted P(bounce) plus ATR-based stop distance, not here - this script
only produces the features/labels used to fit that probability.
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

from backtest_levels import load_csv, infer_timeframe, compute_atr, evaluate_level

with contextlib.redirect_stdout(io.StringIO()):
    import backend

CANDIDATE_METHODS = [
    ('GMM', backend.calculate_gmm_levels, {}),
    ('TDA', backend.persistent_homology_levels, {'max_levels': 8}),
    ('HDBSCAN', backend.calculate_hdbscan_levels, {'timeframe_kw': True}),
    ('OPTICS', backend.optics_multi_density_levels, {}),
    ('KDE', backend.kde_based_levels, {}),
    ('MeanShift', backend.calculate_meanshift_levels, {}),
    ('Isolation-Forest', backend.find_pivot_anomalies, {}),
]


def fit_gjr_garch_forecast(returns_pct):
    """GJR-GARCH(1,1,1) - the o=1 term captures the asymmetric/leverage
    effect plain GARCH misses. Returns 1-step-ahead vol forecast in the
    same units as returns_pct (percent)."""
    try:
        model = arch_model(returns_pct, vol='GARCH', p=1, o=1, q=1, rescale=False)
        result = model.fit(disp='off', show_warning=False)
        forecast = result.forecast(horizon=1)
        return float(np.sqrt(forecast.variance.values[-1, 0]))
    except Exception:
        return None


def hmm_regime_change_features(closes, n_states=3, flip_lookback=5):
    """Returns (state_confidence, recent_flip) for the LAST bar, or
    (None, None) if the fit fails / not enough data."""
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


def collect_events(path, lookback, horizon, step,
                    bounce_atr_mult, break_atr_mult, break_confirm_bars,
                    reaction_bars, recovery_bars, confluence_atr_mult=0.5, verbose=False):
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
        if verbose and wi % 200 == 0:
            print(f"  window {wi}/{len(starts)} ({time.time()-t0:.1f}s elapsed, {len(rows)} events so far)")

        win_h, win_l, win_c = highs[t - lookback:t], lows[t - lookback:t], closes[t - lookback:t]
        win_v, win_dt = volumes[t - lookback:t], datetimes[t - lookback:t]
        current_price = closes[t - 1]
        atr = compute_atr(win_h, win_l, win_c)
        if atr <= 0:
            continue

        returns_pct = np.diff(np.log(win_c)) * 100
        with contextlib.redirect_stdout(io.StringIO()):
            gjr_vol = fit_gjr_garch_forecast(returns_pct)
        if gjr_vol is None:
            continue
        vol_forecast = gjr_vol / 100.0 * current_price

        try:
            gk_vol_pct = backend.garman_klass_daily_volatility(
                df['open'].values[t - lookback:t], win_h, win_l, win_c) * 100
        except Exception:
            gk_vol_pct = np.nan

        try:
            hurst = backend.calculate_hurst_exponent(win_c)['hurst']
        except Exception:
            hurst = np.nan

        with contextlib.redirect_stdout(io.StringIO()):
            hmm_conf, hmm_flip = hmm_regime_change_features(win_c)

        with contextlib.redirect_stdout(io.StringIO()):
            vwap_result = backend.calculate_vwap(win_h, win_l, win_c, win_v, timestamps=win_dt)
        if vwap_result is None:
            continue
        vwap = vwap_result['vwap']

        candidates = []
        for level_name, fn, kwargs in CANDIDATE_METHODS:
            call_kwargs = {'timeframe': timeframe} if kwargs.get('timeframe_kw') else {k: v for k, v in kwargs.items() if k != 'timeframe_kw'}
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    lvls = fn(win_h, win_l, win_c, **call_kwargs)
            except Exception:
                lvls = []
            candidates += [(lvl['price'], level_name) for lvl in (lvls or [])]
        candidates.append((vwap, 'VWAP'))

        # confluence: for each candidate, how many DISTINCT other level
        # types have a candidate within confluence_atr_mult*ATR of it
        tol = confluence_atr_mult * atr
        by_type = {}
        for price, lt in candidates:
            by_type.setdefault(lt, []).append(price)

        def confluence_count(price, own_type):
            count = 0
            for lt, prices in by_type.items():
                if lt == own_type:
                    continue
                if any(abs(p - price) <= tol for p in prices):
                    count += 1
            return count

        fwd_h, fwd_l, fwd_c = highs[t:t + horizon], lows[t:t + horizon], closes[t:t + horizon]

        for price, level_type in candidates:
            if price is None or price <= 0:
                continue
            side = 'support' if price < current_price else 'resistance'
            outcome = evaluate_level(price, side, fwd_h, fwd_l, fwd_c, atr,
                                      bounce_atr_mult, break_atr_mult, break_confirm_bars,
                                      reaction_bars, recovery_bars)
            if outcome is None:
                continue

            # signed VWAP bias/stretch: positive = trade direction agrees with
            # the VWAP tilt (long above VWAP / short below - "trend" side) and
            # magnitude grows with how stretched price is from VWAP; negative =
            # trade fades the tilt (long below VWAP / short above - a dip-buy/
            # rip-sell) with magnitude growing the same way. One signed
            # continuous feature instead of a hand-asserted hard filter, so the
            # model - not a guess - decides whether aligned-and-stretched or
            # counter-and-stretched actually predicts bounces here.
            trade_direction = 'long' if side == 'support' else 'short'
            price_vs_vwap = current_price - vwap
            aligned_with_bias = (trade_direction == 'long' and price_vs_vwap > 0) or \
                                 (trade_direction == 'short' and price_vs_vwap < 0)
            vwap_stretch = abs(price_vs_vwap) / (vol_forecast + 1e-9)
            vwap_bias_alignment = vwap_stretch if aligned_with_bias else -vwap_stretch
            gjr_vol_pct = vol_forecast / current_price * 100
            gjr_vol_regime_ratio = gjr_vol_pct / (gk_vol_pct + 1e-9)

            rows.append({
                'instrument': instrument, 'timeframe': timeframe, 'level_type': level_type,
                't': t, 'price': price,
                'vwap_distance_norm': (price - vwap) / (vol_forecast + 1e-9),
                'vol_forecast_pct_of_price': vol_forecast / current_price,
                'atr_distance': (price - current_price) / atr,
                'confluence': confluence_count(price, level_type),
                'hurst': hurst,
                'hmm_state_confidence': hmm_conf if hmm_conf is not None else 0.5,
                'hmm_recent_flip': hmm_flip if hmm_flip is not None else 0,
                'garman_klass_vol_pct': gk_vol_pct,
                'gjr_vol_regime_ratio': gjr_vol_regime_ratio,
                'vwap_bias_alignment': vwap_bias_alignment,
                'bounced': int(outcome['bounced']),
            })

    return pd.DataFrame(rows)


def zscore_test(p1, n1, p2, n2):
    x1, x2 = p1 * n1, p2 * n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0 or n1 == 0 or n2 == 0:
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
    ap.add_argument('--out', default='backtest_ml_filter_v2_events.csv')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    all_events = []
    for path in args.files:
        ev = collect_events(path, args.lookback, args.horizon, args.step,
                             args.bounce_atr_mult, args.break_atr_mult, args.break_confirm_bars,
                             args.reaction_bars, args.recovery_bars, verbose=args.verbose)
        print(f"  -> {len(ev)} touched candidate events collected")
        all_events.append(ev)
        # checkpoint after every file - a multi-hour, multi-file run
        # shouldn't lose everything if it's interrupted or crashes partway
        # through a later file
        pd.concat(all_events, ignore_index=True).to_csv(args.out, index=False)
        print(f"  -> checkpointed {sum(len(e) for e in all_events)} events so far to {args.out}")

    events = pd.concat(all_events, ignore_index=True)
    events = events.dropna(subset=['hurst', 'garman_klass_vol_pct'])
    events.to_csv(args.out, index=False)

    events['is_train'] = False
    for (inst, tf), g in events.groupby(['instrument', 'timeframe']):
        cutoff_t = g['t'].quantile(args.train_frac)
        events.loc[g.index, 'is_train'] = g['t'] <= cutoff_t

    train = events[events['is_train']]
    holdout = events[~events['is_train']]
    print(f"\nTrain events: {len(train)}, Holdout events: {len(holdout)}")

    feature_cols = ['vwap_distance_norm', 'vol_forecast_pct_of_price', 'atr_distance',
                     'confluence', 'hurst', 'hmm_state_confidence', 'hmm_recent_flip',
                     'garman_klass_vol_pct', 'gjr_vol_regime_ratio', 'vwap_bias_alignment']
    X_train = pd.get_dummies(train[feature_cols + ['level_type']], columns=['level_type'])
    X_holdout = pd.get_dummies(holdout[feature_cols + ['level_type']], columns=['level_type'])
    X_holdout = X_holdout.reindex(columns=X_train.columns, fill_value=0)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, train['bounced'])
    holdout = holdout.copy()
    holdout['pred_proba'] = model.predict_proba(X_holdout)[:, 1]

    print("\nLogistic regression coefficients:")
    for name, coef in zip(X_train.columns, model.coef_[0]):
        print(f"  {name:<28} {coef:+.4f}")

    print(f"\n=== Per level type: unconditional vs ML-filtered (top 50% by predicted P(bounce)) ===")
    p_rand, n_rand = args.random_accuracy, args.random_n
    n_comparisons = 8
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

    # correlation check: does the score now discriminate within the top
    # half itself (the exact thing v1 failed at)?
    print("\n=== Does pred_proba discriminate WITHIN the filtered (top-half) population? ===")
    for level_type, g in holdout.groupby('level_type'):
        median_score = g['pred_proba'].median()
        top_half = g[g['pred_proba'] >= median_score]
        if len(top_half) < 50:
            continue
        top_quarter_score = top_half['pred_proba'].quantile(0.75)
        very_top = top_half[top_half['pred_proba'] >= top_quarter_score]
        rest = top_half[top_half['pred_proba'] < top_quarter_score]
        print(f"  {level_type:<18} top-25%-of-filtered acc={very_top['bounced'].mean():.3f} (n={len(very_top)})  "
              f"vs rest-of-filtered acc={rest['bounced'].mean():.3f} (n={len(rest)})")


if __name__ == '__main__':
    main()
