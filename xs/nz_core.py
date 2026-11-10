"""Shared core for the noise-area engine (copied from new_models.py lines 11-45): ET 5m arrays A, OPEN/CLOSE/PC/SIG and noise(). Nov 10 2026."""
import numpy as np, pandas as pd, pickle, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad'; COST = 0.5
X = pd.read_parquet('/tmp/claude-0/ml/nq5_real.parquet'); X = X[X.rth].copy(); X['day'] = pd.to_datetime(X.day).dt.normalize()
GRID = np.arange(570, 960, 5); NB = len(GRID); pos_of = {m: i for i, m in enumerate(GRID)}
DAYS = [d for d in sorted(X.day.unique()) if pd.Timestamp('2014-02-01') <= d <= pd.Timestamp('2026-01-30')]; ALLD = sorted(X.day.unique())
A = {}
for d, g in X.groupby('day'):
    a = np.full((6, NB), np.nan); ix = g.m.map(pos_of).values; ok = ~np.isnan(ix.astype(float))
    ix = ix[ok].astype(int); gg = g[ok]; tp = (gg.high.values + gg.low.values + gg.close.values) / 3
    a[0, ix], a[1, ix], a[2, ix], a[3, ix], a[4, ix] = gg.open.values, gg.high.values, gg.low.values, gg.close.values, gg.volume.values.astype(float)
    cv = np.cumsum(np.nan_to_num(a[4])); pv = np.cumsum(np.nan_to_num(a[4]) * np.nan_to_num(np.r_[np.full(0, 0), np.where(np.isnan(a[3]), 0, (a[1] + a[2] + a[3]) / 3)]))
    a[5] = np.where(cv > 0, pv / np.where(cv > 0, cv, 1), np.nan); A[d] = (a, g.sym.iloc[0])
OPEN = pd.Series({d: A[d][0][0][~np.isnan(A[d][0][0])][0] for d in ALLD}); CLOSE = pd.Series({d: A[d][0][3][~np.isnan(A[d][0][3])][-1] for d in ALLD}); SYM = pd.Series({d: A[d][1] for d in ALLD})
PC = CLOSE.shift(1).where(SYM == SYM.shift(1), np.nan)
AB = pd.DataFrame({d: np.abs(A[d][0][3] / OPEN[d] - 1) for d in ALLD}).T; SIG = AB.rolling(14, min_periods=10).mean().shift(1).values; SI = {d: i for i, d in enumerate(ALLD)}
def sg(d): return SIG[SI[d]]
def noise(d, k=1.0, step=30, stop=40.0, side=0, start=600, end=930):
    a, _ = A[d]; o, h, l, c, v, vw = a; s = sg(d) * k; op = OPEN[d]; pc = PC[d] if np.isfinite(PC[d]) else op; rows = []; pos = 0; e = em = None; worst = 0.0; pend = None
    for j in range(NB):
        if np.isnan(c[j]): continue
        if pend is not None: pos, e, em, worst = pend, o[j], j, 0.0; pend = None
        if pos != 0:
            lo = (l[j] - e) if pos > 0 else (e - h[j]); worst = min(worst, lo)
            if stop is not None and lo <= -stop: rows.append((d, em, pos, -stop - COST, worst - COST)); pos = 0; continue
        if j == NB - 1 or np.isnan(o[j + 1]):
            if j == NB - 1 and pos != 0: rows.append((d, em, pos, (c[j] - e) * pos - COST, worst - COST)); pos = 0
            continue
        t = GRID[j]
        if (t + 5) % step != 0 or t + 5 < start or t + 5 > end or not np.isfinite(s[j]): continue
        ub, lb = max(op, pc) * (1 + s[j]), min(op, pc) * (1 - s[j]); nx = o[j + 1]
        if pos == 1 and c[j] < max(ub, vw[j]): rows.append((d, em, 1, nx - e - COST, worst - COST)); pos = 0
        elif pos == -1 and c[j] > min(lb, vw[j]): rows.append((d, em, -1, e - nx - COST, worst - COST)); pos = 0
        if pos == 0 and c[j] > ub and side >= 0: pend = 1
        elif pos == 0 and c[j] < lb and side <= 0: pend = -1
    return rows
