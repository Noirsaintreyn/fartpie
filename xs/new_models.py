"""Nov 10 2026: A BATCH OF NEW ENGINES ON THE CORRECTED (ET) CLOCK. DECLARED BEFORE RESULTS.
Data: /tmp/claude-0/ml/nq5_real.parquet (ET, RTH 09:30-15:55 bars, the file the live engines use). Cost 0.5 pt a round trip, 1 NQ, 2014-02 .. 2026-01-30. Periods: 2014-21 and 2022-Jan26.
PASS = mean points / day > 0 with t >= 2 in BOTH periods (t of the daily mean; days with no trade count as 0). Everything else is a screen, not an edge. ~40 variants are run, so one or two lone passes are expected by chance;
only a family that passes together (neighbouring parameters, long AND short, or both periods with a similar size) counts.
FAMILY N  noise-area momentum (xs/nq_noise_onwide.py rules) grid: band multiple k {0.75, 1.0, 1.25} x decisions every {15, 30} min x hard stop {25, 40, none} x sides {both, long only, short only} (one at a time off the base)
FAMILY M  INSIDE-AREA FADE (new): while price is inside the noise area, a 5m close stretched from VWAP by z noise-sigmas (z = (close - VWAP) / (open x sigma_t)) is faded at the next open, target = VWAP,
          stop = close beyond the band, 60 min time limit, <= 2 a day, 10:00-14:30. z in {1.0, 1.5, 2.0}.
FAMILY L  LAST-30-MINUTE MOMENTUM (Gao, Han, Li, Zhou): enter 15:30, flat 15:55 close. Signal = sign of the first-30-minute return (prior close -> 10:00), or of the 15:00-15:30 return, or both agreeing.
FAMILY G  NOISE-NORMALISED GAP: gz = (open / prior close - 1) / sigma(10:00). |gz| >= {1, 2}: go with the gap or fade it; enter 09:35 open, stop 40, flat 12:00.
Also reported for each: long / short split, correlation of the daily series with the current system (S6 V2 trades x live sizes), and the same series split by the NZ-base sign."""
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
def fade(d, z=1.5, start=600, end=870):
    a, _ = A[d]; o, h, l, c, v, vw = a; s = sg(d); op = OPEN[d]; pc = PC[d] if np.isfinite(PC[d]) else op; rows = []; pos = 0; e = em = None; worst = 0.0; pend = None; n = 0; j0 = 0
    for j in range(NB):
        if np.isnan(c[j]): continue
        if pend is not None: pos, e, em, worst = pend, o[j], j, 0.0; pend = None; n += 1
        if pos != 0:
            lo = (l[j] - e) if pos > 0 else (e - h[j]); worst = min(worst, lo)
        if j == NB - 1 or np.isnan(o[j + 1]):
            if j == NB - 1 and pos != 0: rows.append((d, em, pos, (c[j] - e) * pos - COST, worst - COST)); pos = 0
            continue
        if not np.isfinite(s[j]): continue
        ub, lb = max(op, pc) * (1 + s[j]), min(op, pc) * (1 - s[j]); zz = (c[j] - vw[j]) / (op * s[j]); nx = o[j + 1]
        if pos != 0:
            tgt = (pos > 0 and c[j] >= vw[j]) or (pos < 0 and c[j] <= vw[j]); stp = (pos > 0 and c[j] < lb) or (pos < 0 and c[j] > ub); tl = j - em >= 12
            if tgt or stp or tl: rows.append((d, em, pos, (nx - e) * pos - COST, worst - COST)); pos = 0
        if pos == 0 and n < 2 and GRID[j] >= start and GRID[j] <= end and lb < c[j] < ub:
            if zz >= z: pend = -1
            elif zz <= -z: pend = 1
    return rows
def last30(d, mode):
    a, _ = A[d]; o, h, l, c, v, vw = a; j = pos_of[930]
    if np.isnan(o[j]) or np.isnan(c[-1]): return []
    r1 = (c[pos_of[595]] / PC[d] - 1) if np.isfinite(PC[d]) and np.isfinite(c[pos_of[595]]) else np.nan
    r2 = (c[pos_of[925]] / c[pos_of[895]] - 1) if np.isfinite(c[pos_of[925]]) and np.isfinite(c[pos_of[895]]) else np.nan
    s1, s2 = np.sign(r1) if np.isfinite(r1) else 0, np.sign(r2) if np.isfinite(r2) else 0
    dr = {'first30': s1, 'prior30': s2, 'both': s1 if s1 == s2 else 0}[mode]
    if dr == 0: return []
    lo = (np.nanmin(l[j:]) - o[j]) if dr > 0 else (o[j] - np.nanmax(h[j:]))
    return [(d, j, int(dr), (c[-1] - o[j]) * dr - COST, min(lo, 0) - COST)]
def gap(d, thr, mode):
    a, _ = A[d]; o, h, l, c, v, vw = a; s = sg(d); j = pos_of[595]
    if not np.isfinite(PC[d]) or not np.isfinite(s[j]) or np.isnan(o[pos_of[575]]): return []
    gz = (OPEN[d] / PC[d] - 1) / s[j]
    if abs(gz) < thr: return []
    dr = int(np.sign(gz)) * (1 if mode == 'go' else -1); e = o[pos_of[575]]; worst = 0.0; px = None
    for k in range(pos_of[575], pos_of[720]):
        if np.isnan(c[k]): continue
        lo = (l[k] - e) if dr > 0 else (e - h[k]); worst = min(worst, lo)
        if lo <= -40: px = -40.0; break
    if px is None: ek = pos_of[720]; px = ((o[ek] if not np.isnan(o[ek]) else c[ek - 1]) - e) * dr
    return [(d, pos_of[575], dr, px - COST, worst - COST)]
def stats(rows, label, base=None):
    T = pd.DataFrame(rows, columns=['day', 'j', 'dir', 'pts', 'mae']); s = T.groupby('day').pts.sum().reindex(DAYS).fillna(0.0); out = []; ok = True
    for lab, mm in (('14-21', s.index < '2022-01-01'), ('22-26', s.index >= '2022-01-01')):
        x = s[mm]; t = x.mean() / (x.std() / np.sqrt(len(x))); out.append(f'{lab} {x.mean():+6.2f} (t {t:+5.2f})'); ok &= (x.mean() > 0 and t >= 2)
    L, S = T[T.dir > 0], T[T.dir < 0]
    cr = np.corrcoef(s.values, base.reindex(s.index).values)[0, 1] if base is not None else np.nan
    print(f'  {label:<44} n {len(T):5d} ({len(T) / len(DAYS):.2f}/day) | pts/day ' + ' | '.join(out) + f' | long {L.pts.mean():+5.1f} short {S.pts.mean():+5.1f} per trade | corr S6 {cr:+.2f}{"   <== PASS" if ok else ""}', flush=True)
    return s, T
# --- the current system's daily series (S6 V2 x live size x engine weight)
OUT, NAMES = pickle.load(open(f'{SP}/s6.pkl', 'rb')); UW = {'RA': 1.0, 'M1': .3, 'R1': .5, 'ON+': .4, 'ON-': .1, 'RULES': 1.0, 'V15a': .3}; base = pd.Series(0.0, index=DAYS)
for (t, u, y), n in zip(OUT['V2'], NAMES):
    t = t[t.sz > 0]; dd = pd.to_datetime(t.day).dt.normalize(); base = base.add((t.pts * t.sz * UW[n]).groupby(dd.values).sum().reindex(DAYS).fillna(0.0), fill_value=0)
print(f'S6 daily series (sz x weight): 14-21 {base[base.index < "2022-01-01"].mean():+.2f} | 22-26 {base[base.index >= "2022-01-01"].mean():+.2f} pts/day\n')
R = {}
print('=== FAMILY N: noise-area momentum ===')
for nm, kw in (('base k 1.0, 30m, stop 40, both', {}), ('k 0.75', dict(k=.75)), ('k 1.25', dict(k=1.25)), ('15m decisions', dict(step=15)), ('stop 25', dict(stop=25.0)), ('no stop', dict(stop=None)),
               ('long only', dict(side=1)), ('short only', dict(side=-1)), ('start 10:30', dict(start=630)), ('long only, 15m', dict(side=1, step=15))):
    R['N ' + nm] = stats([r for d in DAYS for r in noise(d, **kw)], 'N ' + nm, base)
print('\n=== FAMILY M: inside-area fade to VWAP ===')
for z in (1.0, 1.5, 2.0): R[f'M z {z}'] = stats([r for d in DAYS for r in fade(d, z)], f'M fade z {z}', base)
print('\n=== FAMILY L: last-30-minute momentum ===')
for md in ('first30', 'prior30', 'both'): R['L ' + md] = stats([r for d in DAYS for r in last30(d, md)], 'L ' + md, base)
print('\n=== FAMILY G: noise-normalised gap ===')
for thr in (1.0, 2.0):
    for md in ('go', 'fade'): R[f'G {md} {thr}'] = stats([r for d in DAYS for r in gap(d, thr, md)], f'G {md} |gz| >= {thr}', base)
pickle.dump({k: v for k, v in R.items()}, open(f'{SP}/new_models.pkl', 'wb'))
