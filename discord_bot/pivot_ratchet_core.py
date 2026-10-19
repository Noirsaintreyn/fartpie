"""Light copy of the pivot-ratchet state machine used by vchart (extracted verbatim from backtest_pivot_ratchet.py by dependency closure,
so the Discord bot does not have to import the research harness, backend.py and their heavy dependencies)."""
import numpy as np


import pandas as pd


PIVOT_LEN = 8


ATR_LEN = 14


BREAK_ATR = 0.35


LEVEL_ATR = 0.15


MIN_MOVE_ATR = 0.80


MIN_BARS_RATCH = 8


CONFIRM_BARS = 1


FAST_LEN = 21


SLOW_LEN = 55


USE_FILTER = True


def wilder_rma(src, length):
    out = np.full(len(src), np.nan)
    if len(src) < length:
        return out
    seed = np.mean(src[:length])
    out[length - 1] = seed
    for i in range(length, len(src)):
        out[i] = (out[i - 1] * (length - 1) + src[i]) / length
    return out


def ema(src, length):
    out = np.full(len(src), np.nan)
    if len(src) < length:
        return out
    alpha = 2.0 / (length + 1)
    seed = np.mean(src[:length])
    out[length - 1] = seed
    for i in range(length, len(src)):
        out[i] = alpha * src[i] + (1 - alpha) * out[i - 1]
    return out


def compute_atr_series(high, low, close, length):
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    return wilder_rma(tr, length)


PIVOT_TOL_ATR_FRAC = 0.01  # a bar within 1% of an ATR of the window extremum still counts as the pivot


def find_pivots(values, left, right, is_high, atr=None, tol_atr_frac=PIVOT_TOL_ATR_FRAC):
    # Vectorized: rolling window extremum (center-aligned) instead of a
    # per-bar Python loop - needed since 1m NQ data alone is ~4.2M rows.
    #
    # Tolerance-based tie, not exact float equality (the documented
    # reproducibility bug: yfinance's auto_adjust=True retroactively
    # revises historical prices on every dividend event, and an exact
    # `s.values == roll.values` check flips which bar "is" the pivot on
    # a fraction-of-a-cent revision - cascading through the whole flip
    # history and making backtests non-reproducible run-to-run, see
    # vector_screener_data_cache.py's docstring for a documented case of
    # the SAME "identical" backtest returning +752%/+633%/+690% on three
    # live pulls). A bar within tol_atr_frac of an ATR of the window's
    # true extremum still counts as the pivot - tight enough to not
    # meaningfully change which bar wins a real, non-tied local extreme,
    # loose enough to absorb the sub-cent noise a price revision adds.
    # Falls back to a tiny relative-float tolerance if no ATR is passed
    # (keeps old call sites working, just without the actual fix).
    n = len(values)
    s = pd.Series(values)
    window = left + right + 1
    roll = s.rolling(window, center=True, min_periods=window).max() if is_high \
        else s.rolling(window, center=True, min_periods=window).min()
    if atr is not None:
        tol = np.asarray(atr, dtype=float) * tol_atr_frac
    else:
        tol = np.abs(values) * 1e-6
    diff = (roll.values - s.values) if is_high else (s.values - roll.values)
    is_piv = diff <= tol
    piv_price = np.where(is_piv, values, np.nan)
    out = np.full(n, np.nan)
    if right < n:
        out[right:] = piv_price[:n - right]
    return out


def run_state_machine(df, tol_atr_frac=PIVOT_TOL_ATR_FRAC):
    high, low, close = df['high'].values, df['low'].values, df['close'].values
    n = len(df)

    atr = compute_atr_series(high, low, close, ATR_LEN)
    ema_fast = ema(close, FAST_LEN)
    ema_slow = ema(close, SLOW_LEN)

    piv_low = find_pivots(low, PIVOT_LEN, PIVOT_LEN, is_high=False, atr=atr, tol_atr_frac=tol_atr_frac)
    piv_high = find_pivots(high, PIVOT_LEN, PIVOT_LEN, is_high=True, atr=atr, tol_atr_frac=tol_atr_frac)

    regime = 0
    level = np.nan
    below_count = 0
    above_count = 0
    last_ratchet_bar = None

    regimes = np.zeros(n, dtype=int)
    levels = np.full(n, np.nan)
    flips = []  # (bar_index, direction) direction: +1 = flip to support, -1 = flip to resistance

    for i in range(n):
        a = atr[i]
        if np.isnan(a):
            regimes[i] = regime
            levels[i] = level
            continue

        new_pivot_low = not np.isnan(piv_low[i])
        new_pivot_high = not np.isnan(piv_high[i])
        candidate_support = (piv_low[i] - a * LEVEL_ATR) if new_pivot_low else np.nan
        candidate_resistance = (piv_high[i] + a * LEVEL_ATR) if new_pivot_high else np.nan

        if regime == 0:
            trend_bull = ema_fast[i] >= ema_slow[i] if not np.isnan(ema_fast[i]) and not np.isnan(ema_slow[i]) else True
            regime = 1 if (USE_FILTER and trend_bull) or not USE_FILTER else -1
            level = (low[i] - a * LEVEL_ATR) if regime == 1 else (high[i] + a * LEVEL_ATR)

        support_break = regime == 1 and not np.isnan(level) and close[i] < level - a * BREAK_ATR
        resist_break = regime == -1 and not np.isnan(level) and close[i] > level + a * BREAK_ATR

        below_count = (below_count + 1) if (regime == 1 and support_break) else 0
        above_count = (above_count + 1) if (regime == -1 and resist_break) else 0

        flip_to_resistance = regime == 1 and below_count >= CONFIRM_BARS
        flip_to_support = regime == -1 and above_count >= CONFIRM_BARS

        if flip_to_resistance:
            regime = -1
            below_count = 0
            above_count = 0
            if new_pivot_high and candidate_resistance > close[i]:
                flip_resistance = candidate_resistance
            else:
                flip_resistance = high[i] + a * LEVEL_ATR
            level = max(flip_resistance, close[i] + a * LEVEL_ATR)
            last_ratchet_bar = i
            flips.append((i, -1))

        elif flip_to_support:
            regime = 1
            below_count = 0
            above_count = 0
            if new_pivot_low and candidate_support < close[i]:
                flip_support = candidate_support
            else:
                flip_support = low[i] - a * LEVEL_ATR
            level = min(flip_support, close[i] - a * LEVEL_ATR)
            last_ratchet_bar = i
            flips.append((i, 1))

        elif regime == 1:
            cooldown_ok = last_ratchet_bar is None or (i - last_ratchet_bar) >= MIN_BARS_RATCH
            if (new_pivot_low and candidate_support > level + a * MIN_MOVE_ATR
                    and candidate_support < close[i] and cooldown_ok):
                level = candidate_support
                last_ratchet_bar = i

        elif regime == -1:
            cooldown_ok = last_ratchet_bar is None or (i - last_ratchet_bar) >= MIN_BARS_RATCH
            if (new_pivot_high and candidate_resistance < level - a * MIN_MOVE_ATR
                    and candidate_resistance > close[i] and cooldown_ok):
                level = candidate_resistance
                last_ratchet_bar = i

        regimes[i] = regime
        levels[i] = level

    return regimes, levels, atr, flips


