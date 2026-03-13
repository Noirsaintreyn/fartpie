"""
Backtest Engine for Level Detection Algorithms and HOD/LOD Predictions

Walk-forward backtesting framework that evaluates:
1. Level detection accuracy (per algorithm)
2. Per-level detail (each individual level's outcome)
3. HOD/LOD prediction accuracy (statistical vs level-constrained vs state-conditioned)

No lookahead bias: at each evaluation point, only past data is used.
"""

import numpy as np
import pandas as pd
import traceback
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Level Evaluation Helpers
# ---------------------------------------------------------------------------

def _level_touched(level_price, future_highs, future_lows, tolerance_pct=0.15):
    """Check if price touched a level within tolerance (% of price)."""
    tol = level_price * (tolerance_pct / 100.0)
    for h, l in zip(future_highs, future_lows):
        if l - tol <= level_price <= h + tol:
            return True
    return False


def _level_bounced(level_price, future_closes, future_highs, future_lows, tolerance_pct=0.15):
    """
    Check if price touched a level and then reversed (bounced).
    A bounce = price touches the level, then moves away by at least 1x tolerance.
    """
    tol = level_price * (tolerance_pct / 100.0)
    touch_idx = None

    for i, (h, l) in enumerate(zip(future_highs, future_lows)):
        if l - tol <= level_price <= h + tol:
            touch_idx = i
            break

    if touch_idx is None:
        return False  # Never touched

    # After touch, did price move away?
    for j in range(touch_idx + 1, len(future_closes)):
        dist = abs(future_closes[j] - level_price)
        if dist > tol * 2:  # Moved 2x tolerance away = bounce
            return True

    return False


def _level_broke_through(level_price, current_price, future_closes, tolerance_pct=0.15):
    """
    Check if price broke through a level (closed beyond it).
    Direction depends on whether level is above or below current price.
    """
    tol = level_price * (tolerance_pct / 100.0)
    is_resistance = level_price > current_price

    for c in future_closes:
        if is_resistance and c > level_price + tol:
            return True
        elif not is_resistance and c < level_price - tol:
            return True

    return False


# ---------------------------------------------------------------------------
# Level Detection Runner (uses functions from backend.py)
# ---------------------------------------------------------------------------

def run_level_detection_algorithms(highs, lows, closes, volumes, hist_subset,
                                   current_price, sigma_price, timeframe='1d',
                                   timestamps=None):
    """
    Run all level detection algorithms and return results keyed by algorithm name.
    Each value is a list of level dicts with at least {'price': float, 'strength': float}.
    """
    # Lazy import to avoid circular imports at module level
    import backend as be

    results = {}

    # HDBSCAN
    try:
        results['HDBSCAN'] = be.calculate_hdbscan_levels(highs, lows, closes, timeframe=timeframe) or []
    except Exception:
        results['HDBSCAN'] = []

    # Enhanced OPTICS
    try:
        results['OPTICS'] = be.enhanced_optics_levels(highs, lows, closes, timeframe=timeframe) or []
    except Exception:
        results['OPTICS'] = []

    # KDE
    try:
        results['KDE'] = be.kde_based_levels(highs, lows, closes) or []
    except Exception:
        results['KDE'] = []

    # Multiscale HDBSCAN
    try:
        results['Multiscale'] = be.multiscale_hdbscan_levels(highs, lows, closes, timeframe=timeframe) or []
    except Exception:
        results['Multiscale'] = []

    # Time-Weighted HDBSCAN
    try:
        if timestamps is not None and len(timestamps) == len(closes):
            results['TimeWeighted'] = be.time_weighted_hdbscan(highs, lows, closes, timestamps) or []
        else:
            results['TimeWeighted'] = []
    except Exception:
        results['TimeWeighted'] = []

    # Wyckoff
    try:
        results['Wyckoff'] = be.detect_wyckoff_zones(hist_subset, lookback=50) or []
    except Exception:
        results['Wyckoff'] = []

    # Persistent Homology (TDA)
    try:
        if hasattr(be, 'RIPSER_AVAILABLE') and be.RIPSER_AVAILABLE:
            results['TDA'] = be.persistent_homology_levels(highs, lows, closes, max_levels=8) or []
        else:
            results['TDA'] = []
    except Exception:
        results['TDA'] = []

    # Neural Network
    try:
        if hasattr(be, 'TORCH_AVAILABLE') and be.TORCH_AVAILABLE:
            results['NeuralNetwork'] = be.detect_levels_with_neural_network(
                hist_subset, lookback=100, threshold=0.7
            ) or []
        else:
            results['NeuralNetwork'] = []
    except Exception:
        results['NeuralNetwork'] = []

    # DeepSupp
    try:
        if hasattr(be, 'TORCH_AVAILABLE') and be.TORCH_AVAILABLE:
            results['DeepSupp'] = be.detect_levels_with_deepsupp(
                hist_subset, model_path='deepsupp_v4.pt', device='cpu'
            ) or []
        else:
            results['DeepSupp'] = []
    except Exception:
        results['DeepSupp'] = []

    # Isolation Forest
    try:
        results['IsolationForest'] = be.find_pivot_anomalies(highs, lows, closes) or []
    except Exception:
        results['IsolationForest'] = []

    # Peak/Valley
    try:
        results['PeakValley'] = be.find_peaks_valleys_scipy(highs, lows, closes) or []
    except Exception:
        results['PeakValley'] = []

    # Local Interaction
    try:
        results['Interaction'] = be.calculate_local_interaction_levels(
            closes, current_price, sigma_price, lookback=200, bins=30, max_levels=5
        ) or []
    except Exception:
        results['Interaction'] = []

    # Pivot Points
    try:
        results['PivotPoints'] = be.calculate_pivot_points(hist_subset, timeframe) or []
    except Exception:
        results['PivotPoints'] = []

    # Fibonacci
    try:
        results['Fibonacci'] = be.calculate_fibonacci_levels(highs, lows) or []
    except Exception:
        results['Fibonacci'] = []

    return results


# ---------------------------------------------------------------------------
# HOD / LOD Prediction Helpers
# ---------------------------------------------------------------------------

def predict_hodlod_statistical(closes, current_price, sigma_price):
    """
    Simple statistical HOD/LOD: current_price +/- N * sigma.
    Returns dict with 1std, 2std, 3std bounds.
    """
    return {
        'hod_1std': current_price + 1.0 * sigma_price,
        'hod_2std': current_price + 2.0 * sigma_price,
        'hod_3std': current_price + 3.0 * sigma_price,
        'lod_1std': current_price - 1.0 * sigma_price,
        'lod_2std': current_price - 2.0 * sigma_price,
        'lod_3std': current_price - 3.0 * sigma_price,
    }


def predict_hodlod_level_constrained(current_price, sigma_price, levels, closes,
                                      volumes, highs, lows, timeframe='1d'):
    """
    Level-constrained HOD/LOD: refine sigma bounds using detected levels.
    Uses refine_extrema_with_levels from backend.
    """
    import backend as be

    base_hod = current_price + 2.0 * sigma_price
    base_lod = current_price - 2.0 * sigma_price

    returns = np.log(closes[1:] / closes[:-1]) * 100 if len(closes) > 1 else np.array([0.0])
    micro_state = be.detect_market_microstructure_state(closes, volumes, returns, highs, lows)

    try:
        refined_hod, refined_lod, _ = be.refine_extrema_with_levels(
            spot=current_price,
            hod_th=base_hod,
            lod_th=base_lod,
            levels=levels,
            state=micro_state,
            timeframe=timeframe,
        )
        return {
            'hod': refined_hod,
            'lod': refined_lod,
        }
    except Exception:
        return {
            'hod': base_hod,
            'lod': base_lod,
        }


# ---------------------------------------------------------------------------
# Main Backtest Functions
# ---------------------------------------------------------------------------

def backtest_levels(
    ticker: str = 'SPY',
    timeframe: str = '1d',
    lookback_bars: int = 200,
    eval_bars: int = 5,
    step_bars: int = 1,
    tolerance_pct: float = 0.15,
    max_eval_points: int = 100,
    progress_callback=None,
):
    """
    Walk-forward backtest of all level detection algorithms.

    Returns per-algorithm aggregate metrics AND per-level detail rows
    so the frontend can display individual level outcomes.
    """
    import yfinance as yf
    import backend as be

    # Fetch data
    stock = yf.Ticker(ticker)
    period_map = {
        '1m': '7d', '5m': '1mo', '15m': '3mo',
        '1h': '6mo', '4h': '1y', '1d': '2y',
    }
    interval_map = {
        '1m': '1m', '5m': '5m', '15m': '15m',
        '1h': '1h', '4h': '1h', '1d': '1d',
    }
    period = period_map.get(timeframe, '2y')
    interval = interval_map.get(timeframe, '1d')

    hist = stock.history(period=period, interval=interval)
    if hist is None or len(hist) < lookback_bars + eval_bars + 10:
        return {'success': False, 'error': f'Insufficient data for {ticker} at {timeframe}'}

    # If 4h, resample from 1h
    if timeframe == '4h':
        hist = hist.resample('4h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min',
            'Close': 'last', 'Volume': 'sum',
        }).dropna()
        if len(hist) < lookback_bars + eval_bars + 10:
            return {'success': False, 'error': f'Insufficient data after 4h resampling for {ticker} (got {len(hist)} bars, need {lookback_bars + eval_bars + 10})'}

    total_bars = len(hist)
    start_idx = lookback_bars
    end_idx = total_bars - eval_bars
    eval_indices = list(range(start_idx, end_idx, step_bars))

    if len(eval_indices) > max_eval_points:
        step = max(1, len(eval_indices) // max_eval_points)
        eval_indices = eval_indices[::step][:max_eval_points]

    total_steps = len(eval_indices)

    # Initialize per-algorithm accumulators
    algo_names = [
        'HDBSCAN', 'OPTICS', 'KDE', 'Multiscale', 'TimeWeighted',
        'Wyckoff', 'TDA', 'NeuralNetwork', 'DeepSupp', 'IsolationForest',
        'PeakValley', 'Interaction', 'PivotPoints', 'Fibonacci',
    ]
    algo_stats = {name: {
        'total_levels': 0,
        'touched': 0,
        'bounced': 0,
        'broke': 0,
        'not_touched': 0,
        'avg_distances': [],
        'eval_points': 0,
    } for name in algo_names}

    step_details = []
    # NEW: per-level detail rows for the "each level" view
    level_details = []

    for step_num, idx in enumerate(eval_indices):
        if progress_callback:
            progress_callback(step_num, total_steps)

        hist_window = hist.iloc[idx - lookback_bars:idx]
        future_window = hist.iloc[idx:idx + eval_bars]

        if len(hist_window) < 60 or len(future_window) < 1:
            continue

        closes = hist_window['Close'].values
        highs = hist_window['High'].values
        lows = hist_window['Low'].values
        volumes = hist_window['Volume'].values
        timestamps = hist_window.index

        current_price = closes[-1]
        sigma_price = np.std(np.diff(np.log(closes))) * 100 * current_price if len(closes) > 1 else current_price * 0.01

        future_closes = future_window['Close'].values
        future_highs = future_window['High'].values
        future_lows = future_window['Low'].values

        # Run all level detection algorithms
        try:
            algo_results = run_level_detection_algorithms(
                highs, lows, closes, volumes,
                hist_window, current_price, sigma_price,
                timeframe=timeframe,
                timestamps=timestamps,
            )
        except Exception as e:
            print(f"Level detection failed at step {step_num}: {e}")
            continue

        step_detail = {
            'bar_index': int(idx),
            'date': str(hist.index[idx]),
            'current_price': float(current_price),
            'algorithms': {},
        }

        for algo_name, levels in algo_results.items():
            if algo_name not in algo_stats:
                continue

            n_levels = len(levels)
            algo_stats[algo_name]['total_levels'] += n_levels
            algo_stats[algo_name]['eval_points'] += 1

            touched_count = 0
            bounced_count = 0
            broke_count = 0
            distances = []

            for level in levels:
                price = level.get('price', None)
                if price is None or not np.isfinite(price):
                    continue

                dist_pct = abs(price - current_price) / current_price * 100
                distances.append(dist_pct)

                touched = _level_touched(price, future_highs, future_lows, tolerance_pct)
                bounced = False
                broke = False

                if touched:
                    touched_count += 1
                    algo_stats[algo_name]['touched'] += 1

                    bounced = _level_bounced(price, future_closes, future_highs, future_lows, tolerance_pct)
                    if bounced:
                        bounced_count += 1
                        algo_stats[algo_name]['bounced'] += 1

                    broke = _level_broke_through(price, current_price, future_closes, tolerance_pct)
                    if broke:
                        broke_count += 1
                        algo_stats[algo_name]['broke'] += 1
                else:
                    algo_stats[algo_name]['not_touched'] += 1

                # Determine level type (support vs resistance)
                level_type = 'resistance' if price > current_price else 'support'

                # Determine outcome
                if not touched:
                    outcome = 'not_touched'
                elif bounced:
                    outcome = 'bounced'
                elif broke:
                    outcome = 'broke'
                else:
                    outcome = 'touched'

                # Per-level detail row
                level_details.append({
                    'date': str(hist.index[idx]),
                    'algorithm': algo_name,
                    'level_price': round(float(price), 4),
                    'current_price': round(float(current_price), 4),
                    'distance_pct': round(dist_pct, 4),
                    'level_type': level_type,
                    'touched': touched,
                    'bounced': bounced,
                    'broke': broke,
                    'outcome': outcome,
                    'strength': round(float(level.get('strength', 0)), 4),
                })

            algo_stats[algo_name]['avg_distances'].extend(distances)

            step_detail['algorithms'][algo_name] = {
                'n_levels': n_levels,
                'touched': touched_count,
                'bounced': bounced_count,
                'broke': broke_count,
            }

        step_details.append(step_detail)

    # Compute final metrics per algorithm
    algo_metrics = {}
    for name, stats in algo_stats.items():
        total = stats['total_levels']
        touched = stats['touched']
        bounced = stats['bounced']
        broke = stats['broke']
        not_touched = stats['not_touched']
        avg_dists = stats['avg_distances']

        algo_metrics[name] = {
            'total_levels_generated': total,
            'eval_points': stats['eval_points'],
            'avg_levels_per_eval': round(total / max(1, stats['eval_points']), 2),
            'hit_rate': round(touched / max(1, total) * 100, 2),
            'bounce_rate': round(bounced / max(1, touched) * 100, 2),
            'break_rate': round(broke / max(1, touched) * 100, 2),
            'false_positive_rate': round(not_touched / max(1, total) * 100, 2),
            'avg_distance_pct': round(np.mean(avg_dists), 4) if avg_dists else 0.0,
            'median_distance_pct': round(np.median(avg_dists), 4) if avg_dists else 0.0,
        }

    return {
        'success': True,
        'ticker': ticker,
        'timeframe': timeframe,
        'total_bars': total_bars,
        'eval_points': total_steps,
        'lookback_bars': lookback_bars,
        'eval_bars': eval_bars,
        'tolerance_pct': tolerance_pct,
        'algorithm_metrics': algo_metrics,
        'step_details': step_details,
        'level_details': level_details,
    }


def backtest_hodlod(
    ticker: str = 'SPY',
    timeframe: str = '1d',
    lookback_bars: int = 200,
    step_bars: int = 1,
    max_eval_points: int = 100,
    progress_callback=None,
):
    """
    Walk-forward backtest of HOD/LOD prediction methods.

    For each evaluation point:
    - Predict HOD/LOD using statistical and level-constrained methods
    - Compare against actual realized HOD/LOD in the next bar
    """
    import yfinance as yf
    import backend as be

    # Fetch data
    stock = yf.Ticker(ticker)
    period_map = {
        '1m': '7d', '5m': '1mo', '15m': '3mo',
        '1h': '6mo', '4h': '1y', '1d': '2y',
    }
    interval_map = {
        '1m': '1m', '5m': '5m', '15m': '15m',
        '1h': '1h', '4h': '1h', '1d': '1d',
    }
    period = period_map.get(timeframe, '2y')
    interval = interval_map.get(timeframe, '1d')

    hist = stock.history(period=period, interval=interval)
    if hist is None or len(hist) < lookback_bars + 10:
        return {'success': False, 'error': f'Insufficient data for {ticker} at {timeframe}'}

    if timeframe == '4h':
        hist = hist.resample('4h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min',
            'Close': 'last', 'Volume': 'sum',
        }).dropna()
        if len(hist) < lookback_bars + 10:
            return {'success': False, 'error': f'Insufficient data after 4h resampling for {ticker} (got {len(hist)} bars, need {lookback_bars + 10})'}

    total_bars = len(hist)
    start_idx = lookback_bars
    end_idx = total_bars - 1
    eval_indices = list(range(start_idx, end_idx, step_bars))

    if len(eval_indices) > max_eval_points:
        step = max(1, len(eval_indices) // max_eval_points)
        eval_indices = eval_indices[::step][:max_eval_points]

    total_steps = len(eval_indices)

    methods = ['statistical_1std', 'statistical_2std', 'statistical_3std', 'level_constrained']
    method_stats = {m: {
        'hod_errors': [],
        'lod_errors': [],
        'hod_abs_errors': [],
        'lod_abs_errors': [],
        'hod_pct_errors': [],
        'lod_pct_errors': [],
        'contained': 0,
        'total': 0,
        'hod_above_actual': 0,
        'lod_below_actual': 0,
    } for m in methods}

    step_details = []

    for step_num, idx in enumerate(eval_indices):
        if progress_callback:
            progress_callback(step_num, total_steps)

        hist_window = hist.iloc[idx - lookback_bars:idx]
        next_bar = hist.iloc[idx]

        if len(hist_window) < 60:
            continue

        closes = hist_window['Close'].values
        highs = hist_window['High'].values
        lows = hist_window['Low'].values
        volumes = hist_window['Volume'].values

        current_price = closes[-1]
        log_returns = np.diff(np.log(closes))
        sigma_price = np.std(log_returns) * 100 * current_price if len(log_returns) > 0 else current_price * 0.01

        actual_hod = float(next_bar['High'])
        actual_lod = float(next_bar['Low'])

        stat_pred = predict_hodlod_statistical(closes, current_price, sigma_price)

        all_levels = []
        try:
            hdbscan_lvls = be.calculate_hdbscan_levels(highs, lows, closes, timeframe=timeframe) or []
            all_levels.extend(hdbscan_lvls)
        except Exception:
            pass
        try:
            optics_lvls = be.enhanced_optics_levels(highs, lows, closes, timeframe=timeframe) or []
            all_levels.extend(optics_lvls)
        except Exception:
            pass
        try:
            kde_lvls = be.kde_based_levels(highs, lows, closes) or []
            all_levels.extend(kde_lvls)
        except Exception:
            pass
        try:
            interaction_lvls = be.calculate_local_interaction_levels(
                closes, current_price, sigma_price, lookback=200, bins=30, max_levels=5
            ) or []
            all_levels.extend(interaction_lvls)
        except Exception:
            pass

        lc_pred = predict_hodlod_level_constrained(
            current_price, sigma_price, all_levels, closes, volumes, highs, lows, timeframe
        )

        predictions = {
            'statistical_1std': {'hod': stat_pred['hod_1std'], 'lod': stat_pred['lod_1std']},
            'statistical_2std': {'hod': stat_pred['hod_2std'], 'lod': stat_pred['lod_2std']},
            'statistical_3std': {'hod': stat_pred['hod_3std'], 'lod': stat_pred['lod_3std']},
            'level_constrained': {'hod': lc_pred['hod'], 'lod': lc_pred['lod']},
        }

        step_detail = {
            'bar_index': int(idx),
            'date': str(hist.index[idx]),
            'current_price': float(current_price),
            'actual_hod': actual_hod,
            'actual_lod': actual_lod,
            'predictions': {},
        }

        for method_name, pred in predictions.items():
            pred_hod = pred['hod']
            pred_lod = pred['lod']

            hod_err = pred_hod - actual_hod
            lod_err = pred_lod - actual_lod
            hod_abs_err = abs(hod_err)
            lod_abs_err = abs(lod_err)
            hod_pct_err = abs(hod_err) / actual_hod * 100 if actual_hod > 0 else 0
            lod_pct_err = abs(lod_err) / actual_lod * 100 if actual_lod > 0 else 0

            contained = (pred_hod >= actual_hod) and (pred_lod <= actual_lod)

            stats = method_stats[method_name]
            stats['hod_errors'].append(hod_err)
            stats['lod_errors'].append(lod_err)
            stats['hod_abs_errors'].append(hod_abs_err)
            stats['lod_abs_errors'].append(lod_abs_err)
            stats['hod_pct_errors'].append(hod_pct_err)
            stats['lod_pct_errors'].append(lod_pct_err)
            stats['total'] += 1
            if contained:
                stats['contained'] += 1
            if pred_hod >= actual_hod:
                stats['hod_above_actual'] += 1
            if pred_lod <= actual_lod:
                stats['lod_below_actual'] += 1

            step_detail['predictions'][method_name] = {
                'predicted_hod': float(pred_hod),
                'predicted_lod': float(pred_lod),
                'hod_error': float(hod_err),
                'lod_error': float(lod_err),
                'contained': contained,
            }

        step_details.append(step_detail)

    method_metrics = {}
    for method_name, stats in method_stats.items():
        total = stats['total']
        if total == 0:
            method_metrics[method_name] = {'total': 0, 'error': 'No evaluation points'}
            continue

        method_metrics[method_name] = {
            'total_eval_points': total,
            'hod_mae': round(float(np.mean(stats['hod_abs_errors'])), 4),
            'lod_mae': round(float(np.mean(stats['lod_abs_errors'])), 4),
            'hod_mape': round(float(np.mean(stats['hod_pct_errors'])), 4),
            'lod_mape': round(float(np.mean(stats['lod_pct_errors'])), 4),
            'hod_bias': round(float(np.mean(stats['hod_errors'])), 4),
            'lod_bias': round(float(np.mean(stats['lod_errors'])), 4),
            'containment_rate': round(stats['contained'] / total * 100, 2),
            'hod_conservative_rate': round(stats['hod_above_actual'] / total * 100, 2),
            'lod_conservative_rate': round(stats['lod_below_actual'] / total * 100, 2),
            'hod_rmse': round(float(np.sqrt(np.mean(np.array(stats['hod_errors']) ** 2))), 4),
            'lod_rmse': round(float(np.sqrt(np.mean(np.array(stats['lod_errors']) ** 2))), 4),
        }

    return {
        'success': True,
        'ticker': ticker,
        'timeframe': timeframe,
        'total_bars': total_bars,
        'eval_points': total_steps,
        'lookback_bars': lookback_bars,
        'method_metrics': method_metrics,
        'step_details': step_details,
    }


def backtest_multi_timeframe(
    ticker: str = 'SPY',
    timeframes: list = None,
    lookback_bars: int = 200,
    eval_bars: int = 5,
    step_bars: int = 1,
    tolerance_pct: float = 0.15,
    max_eval_points: int = 50,
    progress_callback=None,
):
    """
    Run level backtests across multiple timeframes and return combined results
    for side-by-side comparison.

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    timeframes : list of str
        Timeframes to test (e.g., ['5m', '15m', '1h', '4h', '1d']).
    lookback_bars : int
        Number of bars of history per eval point.
    eval_bars : int
        Future bars to evaluate levels against.
    step_bars : int
        Step between eval points.
    tolerance_pct : float
        Tolerance for level touch.
    max_eval_points : int
        Max eval points per timeframe (lower default for multi-TF).
    progress_callback : callable, optional
        Called with (timeframe_index, total_timeframes, step, total_steps).

    Returns
    -------
    dict with per-timeframe results for comparison.
    """
    if timeframes is None:
        timeframes = ['15m', '1h', '4h', '1d']

    results = {
        'success': True,
        'ticker': ticker,
        'timeframes': timeframes,
        'timestamp': datetime.utcnow().isoformat(),
        'per_timeframe': {},
    }

    for tf_idx, tf in enumerate(timeframes):
        def tf_progress(step, total):
            if progress_callback:
                progress_callback(tf_idx, len(timeframes), step, total)

        try:
            tf_result = backtest_levels(
                ticker=ticker,
                timeframe=tf,
                lookback_bars=lookback_bars,
                eval_bars=eval_bars,
                step_bars=step_bars,
                tolerance_pct=tolerance_pct,
                max_eval_points=max_eval_points,
                progress_callback=tf_progress,
            )
            results['per_timeframe'][tf] = tf_result
        except Exception as e:
            results['per_timeframe'][tf] = {
                'success': False,
                'error': str(e),
                'trace': traceback.format_exc(),
            }

    return results


def run_full_backtest(
    ticker: str = 'SPY',
    timeframe: str = '1d',
    lookback_bars: int = 200,
    eval_bars: int = 5,
    step_bars: int = 1,
    tolerance_pct: float = 0.15,
    max_eval_points: int = 100,
    progress_callback=None,
):
    """
    Run both level and HOD/LOD backtests and return combined results.
    """
    results = {
        'success': True,
        'ticker': ticker,
        'timeframe': timeframe,
        'timestamp': datetime.utcnow().isoformat(),
    }

    # Level backtest
    try:
        level_results = backtest_levels(
            ticker=ticker,
            timeframe=timeframe,
            lookback_bars=lookback_bars,
            eval_bars=eval_bars,
            step_bars=step_bars,
            tolerance_pct=tolerance_pct,
            max_eval_points=max_eval_points,
            progress_callback=progress_callback,
        )
        results['levels'] = level_results
    except Exception as e:
        results['levels'] = {'success': False, 'error': str(e), 'trace': traceback.format_exc()}

    # HOD/LOD backtest
    try:
        hodlod_results = backtest_hodlod(
            ticker=ticker,
            timeframe=timeframe,
            lookback_bars=lookback_bars,
            step_bars=step_bars,
            max_eval_points=max_eval_points,
            progress_callback=progress_callback,
        )
        results['hodlod'] = hodlod_results
    except Exception as e:
        results['hodlod'] = {'success': False, 'error': str(e), 'trace': traceback.format_exc()}

    return results
