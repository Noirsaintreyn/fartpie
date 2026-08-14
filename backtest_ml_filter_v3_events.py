"""
v3 of the ML filter: v2's exact feature set plus ONE new feature -
ou_zone_distance_atr - derived from the independently-validated OU-process
zone model (backtest_ou_zones.py). That standalone backtest showed a real,
statistically significant (z=5.7-14.6 across all 4 instrument/timeframe
combos), temporally stable, 12-year-consistent ~61-64% rejection rate at
the OU zone's edges (vs 50% random for a symmetric edge) - see conversation
history. This wires that into the level filter as intended: NOT a
multi-timeframe/HTF-VWAP expansion (that was tried, showed near-zero
coefficients and mostly noise-level accuracy changes on the full dataset,
and was discarded), but the one addition that's independently earned its
place before touching this model.

ou_zone_distance_atr: signed distance (in ATR units) from a candidate's
price to the nearest OU zone edge:
  0            if the candidate sits INSIDE the zone (between the edges)
  negative     if the candidate is BELOW zone_low (magnitude = how far below)
  positive     if the candidate is ABOVE zone_high (magnitude = how far above)
This lets the model learn whether candidates positioned beyond the
zone's edges - exactly where the OU backtest found real rejection edge -
are more likely to bounce, rather than hand-asserting it as a hard rule.
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
from backtest_ou_zones import fit_ou_process
from backtest_hvn_lvn_context import build_volume_profile, nearest_hvn_lvn_distance

with contextlib.redirect_stdout(io.StringIO()):
    import backend

CANDIDATE_METHODS = [
    ('GMM', backend.calculate_gmm_levels, {'volumes_kw': True}),
    ('TDA', backend.persistent_homology_levels, {'max_levels': 8}),
    ('HDBSCAN', backend.calculate_hdbscan_levels, {'timeframe_kw': True, 'volumes_kw': True}),
    # OPTICS: was backend.optics_multi_density_levels, which had zero live
    # callers and was removed as dead code in today's system review - the
    # function actually wired into every live endpoint is enhanced_optics_levels
    ('OPTICS', backend.enhanced_optics_levels, {'timeframe_kw': True, 'volumes_kw': True}),
    ('KDE', backend.kde_based_levels, {'volumes_kw': True}),
    ('MeanShift', backend.calculate_meanshift_levels, {'volumes_kw': True}),
    ('Isolation-Forest', backend.find_pivot_anomalies, {}),
    # Added to match today's unified algorithm set across the 4 main
    # endpoints (previously this v3 script predated that unification and
    # was never actually run to completion - no output CSV existed).
    # Wyckoff and time_weighted_hdbscan are NOT included here despite also
    # being part of today's unified set - both take a different call
    # signature (DataFrame / timestamps array, not just highs/lows/closes)
    # that this script's generic call loop doesn't support - would need
    # separate handling, scoped out of this pass rather than rushed.
    ('Multiscale-HDBSCAN', backend.multiscale_hdbscan_levels, {'timeframe_kw': True, 'volumes_kw': True}),
    # Three new candidate generators (mechanistically different from the
    # 8 above - all of those find dense historical price clusters; these
    # find structural breaks, an adaptive fair-value estimate, and
    # multi-scale-persistent pivots instead) - see backend.py docstrings
    # for each. Not yet validated, being tested exactly like every other
    # candidate: same events pipeline, same walk-forward accuracy check.
    # Changepoint validated as a real edge on 4H, but only on NQ (+13.3pp
    # over its own realized-R:R breakeven) and ES (+6.3pp) once broken out
    # per instrument - GC/SI were noise-level (+1.4pp/+0.7pp) and CL was
    # net negative (-8.7pp, n=134). Gated to 4H + NQ/ES only rather than
    # run everywhere.
    ('Changepoint', backend.changepoint_levels, {'timeframe_only': '4h', 'instrument_allowlist': {'NQ', 'ES'}}),
    # Kalman (neutral/inconclusive) and Wavelet (moderate, not yet decided)
    # left out of the active set for now - not rejected like volume-weighted
    # clustering or 3:1 RR were, just not formalized yet. Uncomment to re-test.
    # ('Kalman', backend.kalman_equilibrium_levels, {}),
    # ('Wavelet', backend.wavelet_pivot_levels, {}),
]


def fit_gjr_garch_forecast(returns_pct):
    try:
        model = arch_model(returns_pct, vol='GARCH', p=1, o=1, q=1, rescale=False)
        result = model.fit(disp='off', show_warning=False)
        forecast = result.forecast(horizon=1)
        return float(np.sqrt(forecast.variance.values[-1, 0]))
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


def collect_events(path, lookback, horizon, step,
                    bounce_atr_mult, break_atr_mult, break_confirm_bars,
                    reaction_bars, recovery_bars, confluence_atr_mult=0.5,
                    ou_k=0.25, verbose=False, volume_weighted_clustering=False):
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

        # OHLCV-approximated volume-by-price for this window (not true
        # tick-level VP, which we don't have - see backtest_hvn_lvn_context.py
        # for the approximation and the validated incremental lift over the
        # existing feature set). Computed once per window here (candidate
        # loop below re-slices win_h/win_l for the forward reaction window,
        # so this must happen before that reassignment).
        hvn_bin_centers, hvn_vol_at_price = build_volume_profile(win_h, win_l, win_v)

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

        # OU zone: fit on price-minus-VWAP deviation, same as
        # backtest_ou_zones.py's independently-validated methodology
        vwap_series = vwap_result['vwap_series']
        deviation = win_c - vwap_series
        ou_fit = fit_ou_process(deviation)
        if ou_fit is not None:
            _, ou_mu, ou_std = ou_fit
            ou_zone_low = vwap_series[-1] + ou_mu - ou_k * ou_std
            ou_zone_high = vwap_series[-1] + ou_mu + ou_k * ou_std
        else:
            ou_zone_low = ou_zone_high = None

        candidates = []
        for level_name, fn, kwargs in CANDIDATE_METHODS:
            if kwargs.get('timeframe_only') and kwargs['timeframe_only'] != timeframe:
                continue
            if kwargs.get('instrument_allowlist') and instrument not in kwargs['instrument_allowlist']:
                continue
            call_kwargs = {k: v for k, v in kwargs.items() if k not in ('timeframe_kw', 'volumes_kw', 'timeframe_only', 'instrument_allowlist')}
            if kwargs.get('timeframe_kw'):
                call_kwargs['timeframe'] = timeframe
            if volume_weighted_clustering and kwargs.get('volumes_kw'):
                call_kwargs['volumes'] = win_v
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    lvls = fn(win_h, win_l, win_c, **call_kwargs)
            except Exception:
                lvls = []
            candidates += [(lvl['price'], level_name) for lvl in (lvls or [])]
        candidates.append((vwap, 'VWAP'))

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

            # Realized MFE/MAE (max favorable/adverse excursion, in ATR
            # units) within the same reaction window evaluate_level uses -
            # duplicates its touch-detection scan rather than modifying the
            # shared function (evaluate_level is imported/used elsewhere in
            # the codebase, don't want to risk changing its return contract).
            # This is the ACTUAL price movement, not the fixed 1.0/0.5 ATR
            # target/stop evaluate_level's bounce/break labels are defined
            # against - answers "how far did it really go", not just
            # "did it clear the fixed target before the fixed stop".
            touch_idx = None
            for i in range(len(fwd_h)):
                if fwd_l[i] <= price <= fwd_h[i]:
                    touch_idx = i
                    break
            mfe_atr = mae_atr = np.nan
            if touch_idx is not None:
                react_end = min(touch_idx + reaction_bars, len(fwd_c))
                win_h, win_l = fwd_h[touch_idx:react_end], fwd_l[touch_idx:react_end]
                if len(win_h):
                    if side == 'support':
                        mfe_atr = float((win_h - price).max() / atr)
                        mae_atr = float((price - win_l).max() / atr)
                    else:
                        mfe_atr = float((price - win_l).max() / atr)
                        mae_atr = float((win_h - price).max() / atr)

            trade_direction = 'long' if side == 'support' else 'short'
            price_vs_vwap = current_price - vwap
            aligned_with_bias = (trade_direction == 'long' and price_vs_vwap > 0) or \
                                 (trade_direction == 'short' and price_vs_vwap < 0)
            vwap_stretch = abs(price_vs_vwap) / (vol_forecast + 1e-9)
            vwap_bias_alignment = vwap_stretch if aligned_with_bias else -vwap_stretch
            gjr_vol_pct = vol_forecast / current_price * 100
            gjr_vol_regime_ratio = gjr_vol_pct / (gk_vol_pct + 1e-9)

            if ou_zone_low is not None:
                if price < ou_zone_low:
                    ou_zone_distance_atr = (price - ou_zone_low) / atr
                elif price > ou_zone_high:
                    ou_zone_distance_atr = (price - ou_zone_high) / atr
                else:
                    ou_zone_distance_atr = 0.0
            else:
                ou_zone_distance_atr = np.nan

            dist_hvn_atr, dist_lvn_atr = nearest_hvn_lvn_distance(price, hvn_bin_centers, hvn_vol_at_price, atr)

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
                'ou_zone_distance_atr': ou_zone_distance_atr,
                'bounced': int(outcome['bounced']),
                'mfe_atr': mfe_atr, 'mae_atr': mae_atr,
                'dist_hvn_atr': dist_hvn_atr, 'dist_lvn_atr': dist_lvn_atr,
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
    ap.add_argument('--ou-k', type=float, default=0.25, help='OU zone half-width in stationary-std units')
    ap.add_argument('--random-accuracy', type=float, default=0.4227)
    ap.add_argument('--random-n', type=int, default=29257)
    ap.add_argument('--out', default='backtest_ml_filter_v3_events.csv')
    ap.add_argument('-v', '--verbose', action='store_true')
    ap.add_argument('--volume-weighted-clustering', action='store_true',
                     help='Pass volumes into HDBSCAN/OPTICS/Multiscale-HDBSCAN/GMM/MeanShift/KDE so the '
                          'clustering itself is volume-weighted, not just an added filter feature')
    args = ap.parse_args()

    all_events = []
    for path in args.files:
        ev = collect_events(path, args.lookback, args.horizon, args.step,
                             args.bounce_atr_mult, args.break_atr_mult, args.break_confirm_bars,
                             args.reaction_bars, args.recovery_bars, ou_k=args.ou_k, verbose=args.verbose,
                             volume_weighted_clustering=args.volume_weighted_clustering)
        print(f"  -> {len(ev)} touched candidate events collected")
        all_events.append(ev)
        pd.concat(all_events, ignore_index=True).to_csv(args.out, index=False)
        print(f"  -> checkpointed {sum(len(e) for e in all_events)} events so far to {args.out}")

    events = pd.concat(all_events, ignore_index=True)
    events = events.dropna(subset=['hurst', 'garman_klass_vol_pct', 'ou_zone_distance_atr'])
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
                     'garman_klass_vol_pct', 'gjr_vol_regime_ratio', 'vwap_bias_alignment',
                     'ou_zone_distance_atr']
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
