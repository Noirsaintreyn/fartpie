"""Exact Python port of the Pine 'SVM - Fractals' vector (TradingView indicator). Differences from xs/vector_core.py:
strict ta.pivothigh/low (left side strict, right side >=), candidate levels offset by the ATR at the pivot bar (atr[pivotLen]),
lastRatchetBar set at initialisation. Returns (regimes, levels, atr, flips) like vector_core.run_state_machine."""
import numpy as np
from vector_core import compute_atr_series, ema, PIVOT_LEN, ATR_LEN, BREAK_ATR, LEVEL_ATR, MIN_MOVE_ATR, MIN_BARS_RATCH, CONFIRM_BARS, FAST_LEN, SLOW_LEN
def _piv(v, L, hi):
    n = len(v); out = np.full(n, np.nan)
    for i in range(2 * L, n):
        c = v[i - L]; left = v[i - 2 * L:i - L]; right = v[i - L + 1:i + 1]
        if (c > left.max() and c >= right.max()) if hi else (c < left.min() and c <= right.min()): out[i] = c
    return out
def run_state_machine(df, **_):
    high, low, close = df['high'].values, df['low'].values, df['close'].values; n = len(df)
    atr = compute_atr_series(high, low, close, ATR_LEN); ef, es = ema(close, FAST_LEN), ema(close, SLOW_LEN)
    pl, ph = _piv(low, PIVOT_LEN, False), _piv(high, PIVOT_LEN, True); patr = np.r_[np.full(PIVOT_LEN, np.nan), atr[:-PIVOT_LEN]]
    regime, level, bc, ac, lr = 0, np.nan, 0, 0, None; regs = np.zeros(n, int); levels = np.full(n, np.nan); flips = []
    for i in range(n):
        a = atr[i]
        if np.isnan(a) or np.isnan(ef[i]) or np.isnan(es[i]) and regime == 0:
            regs[i], levels[i] = regime, level; continue
        cs = pl[i] - patr[i] * LEVEL_ATR if not np.isnan(pl[i]) and not np.isnan(patr[i]) else np.nan
        cr = ph[i] + patr[i] * LEVEL_ATR if not np.isnan(ph[i]) and not np.isnan(patr[i]) else np.nan
        if regime == 0:
            regime = 1 if ef[i] >= es[i] else -1; level = low[i] - a * LEVEL_ATR if regime == 1 else high[i] + a * LEVEL_ATR; lr = i
        sb = regime == 1 and close[i] < level - a * BREAK_ATR; rb = regime == -1 and close[i] > level + a * BREAK_ATR
        if regime == 1: bc, ac = (bc + 1 if sb else 0), 0
        elif regime == -1: ac, bc = (ac + 1 if rb else 0), 0
        if regime == 1 and bc >= CONFIRM_BARS:
            regime, bc, ac = -1, 0, 0; nr = cr if (not np.isnan(cr) and cr > close[i]) else high[i] + a * LEVEL_ATR
            level = max(nr, close[i] + a * LEVEL_ATR); lr = i; flips.append((i, -1))
        elif regime == -1 and ac >= CONFIRM_BARS:
            regime, bc, ac = 1, 0, 0; ns = cs if (not np.isnan(cs) and cs < close[i]) else low[i] - a * LEVEL_ATR
            level = min(ns, close[i] - a * LEVEL_ATR); lr = i; flips.append((i, 1))
        elif regime == 1 and not np.isnan(cs) and cs < close[i] and cs > level + a * MIN_MOVE_ATR and (lr is None or i - lr >= MIN_BARS_RATCH):
            level, lr = cs, i
        elif regime == -1 and not np.isnan(cr) and cr > close[i] and cr < level - a * MIN_MOVE_ATR and (lr is None or i - lr >= MIN_BARS_RATCH):
            level, lr = cr, i
        regs[i], levels[i] = regime, level
    return regs, levels, atr, flips
