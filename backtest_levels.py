"""
Walk-forward backtest comparing support/resistance level-detection methods
against realized price behavior, on real OHLCV data.

Methodology (mirrors the metric definitions used in the DeepSupp paper,
arXiv:2507.01971, since that's the standard we've been comparing against):

  For each instrument/timeframe file, step through history. At each step t:
    1. Take the last `--lookback` bars as the detection window.
    2. Run every registered method on that window -> candidate levels.
    3. Look forward up to `--horizon` bars and, for each level:
       - Touch:      did price trade through the level at all?
       - Bounce:     on first touch, did price reverse >= bounce_threshold
                     away from the level within `reaction_bars`, without a
                     decisive close through it first? (-> "Support Accuracy")
       - Hold time:  bars from first touch until the level is decisively
                     broken (close beyond it by `break_buffer`), censored
                     at the horizon if never broken. (-> "Support Hold Duration")
       - Recovery:   if broken, did price close back on the original side
                     within `recovery_bars`? (-> "False Breakout Recovery")
  Levels are only scored on their first touch, to avoid double-counting the
  same level across the window it persists in.

No look-ahead: a level generated from bars [t-lookback, t] is only ever
evaluated against bars (t, t+horizon].

Usage:
  python backtest_levels.py --files /path/1H_NQ.csv /path/4H_NQ.csv ... \
      --lookback 150 --horizon 40 --step 10 --out results.csv
"""
import argparse
import contextlib
import io
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import pywt
from sklearn.mixture import GaussianMixture

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

with contextlib.redirect_stdout(io.StringIO()):
    import backend  # noqa: E402  (reuse the production level-detection implementations)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_euro_number(x):
    """'25.677,75' (thousands='.', decimal=',') -> 25677.75"""
    if isinstance(x, (int, float)):
        return float(x)
    return float(str(x).replace('.', '').replace(',', '.'))


def load_csv(path):
    """Loads one of the exported CSVs: title row, then header, then data
    (most-recent-first, one row per contract-month, European number format).
    Intraday exports have a date+time column ('%m/%d/%Y %I:%M %p'); daily
    exports (D_NQ.csv/D_ES.csv) are date-only ('%m/%d/%Y') - try intraday
    format first, fall back to date-only rather than dropping every row."""
    df = pd.read_csv(path, skiprows=1)
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].apply(_parse_euro_number)
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    date_col = df.columns[0]
    df['datetime'] = pd.to_datetime(df[date_col], format='%m/%d/%Y %I:%M %p', errors='coerce')
    if df['datetime'].isna().all():
        df['datetime'] = pd.to_datetime(df[date_col], format='%m/%d/%Y', errors='coerce')
    df = df.dropna(subset=['datetime', 'open', 'high', 'low', 'close'])
    df = df.sort_values('datetime').reset_index(drop=True)
    return df[['datetime', 'open', 'high', 'low', 'close', 'volume']]


def infer_timeframe(filename):
    m = re.match(r'(\d+)([a-zA-Z]+)', os.path.basename(filename))
    if not m:
        return '1h'
    return (m.group(1) + m.group(2)).lower()


# ---------------------------------------------------------------------------
# Method registry - thin wrappers around backend.py's existing, self-contained
# level-detection functions. Add new candidates here (e.g. wavelet, GMM) as
# they're built, and they get scored by the same harness for free.
# ---------------------------------------------------------------------------

def _run_hdbscan(h, l, c, tf):
    return backend.calculate_hdbscan_levels(h, l, c, timeframe=tf)


def _run_optics(h, l, c, tf):
    return backend.optics_multi_density_levels(h, l, c)


def _run_isolation_forest(h, l, c, tf):
    return backend.find_pivot_anomalies(h, l, c)


def _run_meanshift(h, l, c, tf):
    return backend.calculate_meanshift_levels(h, l, c)


def _run_kde(h, l, c, tf):
    return backend.kde_based_levels(h, l, c)


_RANDOM_RNG = np.random.default_rng(42)


def _run_random(h, l, c, tf, n_levels=8):
    """Baseline sanity check: n_levels drawn uniformly from the window's own
    observed price range, no structure at all. If this scores anywhere near
    the real methods, that says something important about what the metric
    is actually measuring (e.g. generic mean-reversion at this ATR scale)
    rather than method-specific level quality."""
    lo, hi = float(l.min()), float(h.max())
    if hi <= lo:
        return []
    prices = _RANDOM_RNG.uniform(lo, hi, size=n_levels)
    return [{'price': float(p), 'category': 'Random', 'type': 'Random Baseline'} for p in prices]


def _run_tda(h, l, c, tf):
    return backend.persistent_homology_levels(h, l, c, max_levels=8)


# --- New candidates: no persistent training, refit fresh on each window
# (same pattern as HDBSCAN/OPTICS/KDE above), so no extra PIT risk. ---

def gmm_levels(highs, lows, closes, min_components=3, max_components=10, min_frac=0.03):
    """Soft/probabilistic alternative to HDBSCAN: cluster raw prices with a
    Gaussian Mixture, select component count by BIC, use each component's
    mean as a level and mean posterior probability as its confidence."""
    if len(closes) < 20:
        return []
    all_prices = np.concatenate([highs, lows, closes])
    X = all_prices.reshape(-1, 1)
    n = len(X)

    best_gmm, best_bic = None, np.inf
    for k in range(min_components, min(max_components, n // 5) + 1):
        try:
            gmm = GaussianMixture(n_components=k, random_state=42, max_iter=200, n_init=1)
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < best_bic:
                best_bic, best_gmm = bic, gmm
        except Exception:
            continue
    if best_gmm is None:
        return []

    labels = best_gmm.predict(X)
    probs = best_gmm.predict_proba(X)
    levels = []
    for k in range(best_gmm.n_components):
        mask = labels == k
        count = int(mask.sum())
        if count < max(5, n * min_frac):
            continue
        center = float(best_gmm.means_[k][0])
        if center <= 0:
            continue
        confidence = float(np.clip(probs[mask, k].mean(), 0, 0.95))
        levels.append({
            'price': center, 'type': 'GMM Cluster', 'touches': count,
            'strength': confidence, 'breakoutProb': float(1 - confidence),
            'reversionProb': confidence, 'category': 'GMM',
        })
    return levels


def wavelet_levels(highs, lows, closes, wavelet='db4', max_level=3, prominence=0.02):
    """Multi-resolution peak detection: low-pass reconstruct the price series
    at increasing wavelet decomposition levels (coarser = major levels, finer
    = minor levels), find extrema at each scale, then merge duplicates."""
    n = len(closes)
    if n < 32:
        return []
    dec_len = pywt.Wavelet(wavelet).dec_len
    max_possible = pywt.dwt_max_level(n, dec_len)
    top_level = max(1, min(max_level, max_possible))
    if top_level < 1:
        return []

    price_range = highs.max() - lows.min()
    if price_range <= 0:
        return []
    min_prominence = price_range * prominence

    candidates = []
    for lvl in range(1, top_level + 1):
        try:
            coeffs = pywt.wavedec(closes, wavelet, level=lvl)
        except Exception:
            continue
        zeroed = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        try:
            smoothed = pywt.waverec(zeroed, wavelet)[:n]
        except Exception:
            continue
        if len(smoothed) < n:
            continue

        from scipy.signal import find_peaks
        peaks, peak_props = find_peaks(smoothed, prominence=min_prominence, distance=5)
        valleys, valley_props = find_peaks(-smoothed, prominence=min_prominence, distance=5)

        scale_strength = 0.5 + 0.4 * (lvl / top_level)  # coarser scale -> stronger "major" level
        for i, idx in enumerate(peaks):
            if idx >= len(highs):
                continue
            prom_strength = min(peak_props['prominences'][i] / min_prominence / 3, 0.9)
            strength = float(np.clip(prom_strength * 0.5 + scale_strength * 0.5, 0, 0.95))
            candidates.append({
                'price': float(highs[idx]), 'type': f'Wavelet-L{lvl} Resistance',
                'touches': 1, 'strength': strength, 'breakoutProb': float(1 - strength),
                'reversionProb': strength, 'category': 'Wavelet', 'scale': lvl,
            })
        for i, idx in enumerate(valleys):
            if idx >= len(lows):
                continue
            prom_strength = min(valley_props['prominences'][i] / min_prominence / 3, 0.9)
            strength = float(np.clip(prom_strength * 0.5 + scale_strength * 0.5, 0, 0.95))
            candidates.append({
                'price': float(lows[idx]), 'type': f'Wavelet-L{lvl} Support',
                'touches': 1, 'strength': strength, 'breakoutProb': float(1 - strength),
                'reversionProb': strength, 'category': 'Wavelet', 'scale': lvl,
            })

    if not candidates:
        return []
    # merge near-duplicate levels across scales (keep the strongest)
    candidates.sort(key=lambda x: -x['strength'])
    merged = []
    tol = price_range * 0.0025
    for cand in candidates:
        if any(abs(cand['price'] - m['price']) < tol for m in merged):
            continue
        merged.append(cand)
    return merged


def hmm_levels(highs, lows, closes, n_states=4, min_frac=0.05):
    """Regime segmentation via Gaussian HMM (states fit fresh per window, same
    as backend.detect_market_regime_hmm) -> for each state, take the median
    price of bars assigned to it plus the state's high/low bracket, weighted
    by state-posterior confidence. Unlike detect_market_regime_hmm, this
    actually emits price levels instead of just a regime label."""
    if not getattr(backend, 'HMMLEARN_AVAILABLE', False) or len(closes) < 60:
        return []
    returns = np.diff(np.log(closes)).reshape(-1, 1)
    try:
        model = backend.GaussianHMM(n_components=n_states, covariance_type='diag',
                                     n_iter=100, random_state=42)
        model.fit(returns)
        states = model.predict(returns)
        post = model.predict_proba(returns)
    except Exception:
        return []

    price_idx = np.arange(1, len(closes))  # returns[i] = transition into closes[i+1]
    levels = []
    for s in range(n_states):
        mask = states == s
        idx = price_idx[mask]
        if len(idx) < max(5, len(closes) * min_frac):
            continue
        confidence = float(np.clip(post[mask, s].mean(), 0, 0.95))
        median_price = float(np.median(closes[idx]))
        hi_b = float(np.percentile(highs[idx], 90))
        lo_b = float(np.percentile(lows[idx], 10))
        levels.append({
            'price': median_price, 'type': 'HMM Regime Median', 'touches': int(len(idx)),
            'strength': confidence, 'breakoutProb': float(1 - confidence),
            'reversionProb': confidence, 'category': 'HMM',
        })
        edge_strength = float(confidence * 0.8)
        levels.append({
            'price': hi_b, 'type': 'HMM Regime High', 'touches': int(len(idx)),
            'strength': edge_strength, 'breakoutProb': float(1 - edge_strength),
            'reversionProb': edge_strength, 'category': 'HMM',
        })
        levels.append({
            'price': lo_b, 'type': 'HMM Regime Low', 'touches': int(len(idx)),
            'strength': edge_strength, 'breakoutProb': float(1 - edge_strength),
            'reversionProb': edge_strength, 'category': 'HMM',
        })
    return levels


def _run_gmm(h, l, c, tf):
    return gmm_levels(h, l, c)


def _run_wavelet(h, l, c, tf):
    return wavelet_levels(h, l, c)


def _run_hmm(h, l, c, tf):
    return hmm_levels(h, l, c)


METHODS = {
    'HDBSCAN': _run_hdbscan,
    'OPTICS': _run_optics,
    'Isolation-Forest': _run_isolation_forest,
    'MeanShift': _run_meanshift,
    'KDE': _run_kde,
    'GMM': _run_gmm,
    'TDA': _run_tda,
    'Random': _run_random,
    # Wavelet/HMM-Levels function defs kept above for reference but not
    # registered - both backtested weaker than the rest, see prior findings.
}


# ---------------------------------------------------------------------------
# Evaluation of a single level against forward price action
# ---------------------------------------------------------------------------

def evaluate_level(price, side, fwd_high, fwd_low, fwd_close, atr,
                    bounce_atr_mult, break_atr_mult, break_confirm_bars,
                    reaction_bars, recovery_bars):
    """
    side: 'support' if price approached from above (level below current price
          at detection time), 'resistance' if approached from below.
    Returns None if never touched within the forward window.

    break_level/target_level are sized in ATR units (computed from the
    detection window, not the future) instead of a fixed % of price, so the
    thresholds scale with realized volatility instead of penalizing every
    level equally regardless of regime.

    A break requires `break_confirm_bars` CONSECUTIVE closes beyond the
    break level, not one noisy close. Without this, a level that gets
    touched (by definition trading through it) very often has its very own
    touch bar close past a tight fixed-% buffer just from ordinary bar
    range, which was killing the bounce case before it had any chance to
    develop. A brief poke-through-and-reclaim is exactly what a real
    support/resistance defense often looks like, and should still be able
    to resolve as a bounce.
    """
    touch_idx = None
    for i in range(len(fwd_high)):
        if fwd_low[i] <= price <= fwd_high[i]:
            touch_idx = i
            break
    if touch_idx is None:
        return None

    break_level = price - break_atr_mult * atr if side == 'support' else price + break_atr_mult * atr
    target_level = price + bounce_atr_mult * atr if side == 'support' else price - bounce_atr_mult * atr

    reaction_end = min(touch_idx + reaction_bars, len(fwd_close))
    broken = False
    break_idx = None
    bounced = False
    confirm_count = 0
    for i in range(touch_idx, reaction_end):
        c = fwd_close[i]
        beyond_break = (c < break_level) if side == 'support' else (c > break_level)
        reached_target = (c >= target_level) if side == 'support' else (c <= target_level)

        confirm_count = confirm_count + 1 if beyond_break else 0
        if confirm_count >= break_confirm_bars:
            broken = True
            break_idx = i - break_confirm_bars + 1
            break
        if reached_target:
            bounced = True
            break

    hold_duration = (break_idx - touch_idx) if broken else (reaction_end - touch_idx)
    censored = not broken

    recovered = None
    if broken:
        recovery_end = min(break_idx + recovery_bars, len(fwd_close))
        recovered = False
        for i in range(break_idx, recovery_end):
            c = fwd_close[i]
            if side == 'support' and c >= price:
                recovered = True
                break
            if side == 'resistance' and c <= price:
                recovered = True
                break

    return {
        'touched': True,
        'bounced': bounced,
        'broken': broken,
        'hold_duration': hold_duration,
        'censored': censored,
        'recovered': recovered,
    }


# ---------------------------------------------------------------------------
# Walk-forward driver
# ---------------------------------------------------------------------------

def compute_atr(highs, lows, closes, period=14):
    """Average True Range over the last `period` bars of the given (past-only)
    window. Used to size break/bounce thresholds in volatility-relative
    terms instead of a fixed % of price."""
    prev_close = np.concatenate([[closes[0]], closes[:-1]])
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    return float(np.mean(tr[-period:]))


def run_backtest(df, instrument, timeframe, lookback, horizon, step,
                  bounce_atr_mult, break_atr_mult, break_confirm_bars,
                  reaction_bars, recovery_bars,
                  max_windows=None, verbose=False, start_date=None, end_date=None):
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    datetimes = df['datetime'].values

    rows = []
    n = len(df)
    starts = list(range(lookback, n - horizon, step))
    if start_date or end_date:
        # Filter by the EVALUATION point's date (last bar of the detection
        # window), not the underlying data - lookback context before the
        # period start is still fair game (PIT-safe), we're just
        # restricting which dates get scored.
        start_ts = np.datetime64(start_date) if start_date else None
        end_ts = np.datetime64(end_date) if end_date else None
        starts = [t for t in starts
                  if (start_ts is None or datetimes[t - 1] >= start_ts)
                  and (end_ts is None or datetimes[t - 1] <= end_ts)]
    if max_windows:
        starts = starts[:max_windows]

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

        for method_name, fn in METHODS.items():
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    levels = fn(win_h, win_l, win_c, timeframe)
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
                    'instrument': instrument,
                    'timeframe': timeframe,
                    'method': method_name,
                    't': t,
                    'price': price,
                    'side': side,
                    'touched': outcome is not None,
                    'bounced': outcome['bounced'] if outcome else None,
                    'broken': outcome['broken'] if outcome else None,
                    'hold_duration': outcome['hold_duration'] if outcome else None,
                    'censored': outcome['censored'] if outcome else None,
                    'recovered': outcome['recovered'] if outcome else None,
                })

        if verbose and wi % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{instrument} {timeframe}] window {wi}/{len(starts)}  ({elapsed:.1f}s elapsed)")

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
        avg_hold = touched['hold_duration'].mean() if n_touched else np.nan
        return pd.Series({
            'n_levels_generated': n_levels,
            'n_touched': n_touched,
            'touch_rate': n_touched / n_levels if n_levels else np.nan,
            'support_accuracy': n_bounced / n_touched if n_touched else np.nan,
            'break_rate': n_broken / n_touched if n_touched else np.nan,
            'false_breakout_recovery': n_recovered / n_broken if n_broken else np.nan,
            'avg_hold_duration_bars': avg_hold,
        })

    per_method = results.groupby(['instrument', 'timeframe', 'method']).apply(agg).reset_index()
    overall = results.groupby('method').apply(agg).reset_index()
    return per_method, overall


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--files', nargs='+', required=True)
    ap.add_argument('--lookback', type=int, default=150)
    ap.add_argument('--horizon', type=int, default=40)
    ap.add_argument('--step', type=int, default=10)
    ap.add_argument('--bounce-atr-mult', type=float, default=1.0, help='ATR multiples of favorable move counted as a bounce')
    ap.add_argument('--break-atr-mult', type=float, default=0.5, help='ATR multiples of close-through counted as a break')
    ap.add_argument('--break-confirm-bars', type=int, default=2, help='consecutive closes beyond break level required to confirm a break')
    ap.add_argument('--reaction-bars', type=int, default=10, help='bars after touch to resolve bounce vs. break')
    ap.add_argument('--recovery-bars', type=int, default=10, help='bars after a break to check for recovery')
    ap.add_argument('--max-windows', type=int, default=None, help='cap windows per file (for quick smoke tests)')
    ap.add_argument('--start-date', default=None, help='only score evaluation points on/after this date (YYYY-MM-DD) - lookback context before it is still used')
    ap.add_argument('--end-date', default=None, help='only score evaluation points on/before this date (YYYY-MM-DD)')
    ap.add_argument('--out', default='backtest_results.csv')
    ap.add_argument('--summary-out', default='backtest_summary.csv')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    all_results = []
    for path in args.files:
        instrument = re.sub(r'^\d+[a-zA-Z]+_', '', os.path.splitext(os.path.basename(path))[0])
        timeframe = infer_timeframe(os.path.basename(path))
        print(f"Loading {path} (instrument={instrument}, timeframe={timeframe})...")
        df = load_csv(path)
        print(f"  {len(df)} bars, {df['datetime'].min()} -> {df['datetime'].max()}")

        res = run_backtest(
            df, instrument, timeframe,
            lookback=args.lookback, horizon=args.horizon, step=args.step,
            bounce_atr_mult=args.bounce_atr_mult, break_atr_mult=args.break_atr_mult,
            break_confirm_bars=args.break_confirm_bars,
            reaction_bars=args.reaction_bars, recovery_bars=args.recovery_bars,
            max_windows=args.max_windows, verbose=args.verbose,
            start_date=args.start_date, end_date=args.end_date,
        )
        print(f"  -> {len(res)} candidate levels evaluated")
        all_results.append(res)

    results = pd.concat(all_results, ignore_index=True)
    results.to_csv(args.out, index=False)

    per_method, overall = summarize(results)
    per_method.to_csv(args.summary_out, index=False)

    pd.set_option('display.width', 160)
    pd.set_option('display.max_columns', 20)
    print("\n=== Per instrument/timeframe ===")
    print(per_method.round(3).to_string(index=False))
    print("\n=== Overall (all files combined) ===")
    print(overall.round(3).to_string(index=False))
    print(f"\nRaw results -> {args.out}")
    print(f"Summary     -> {args.summary_out}")


if __name__ == '__main__':
    main()
