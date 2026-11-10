"""Nov 10 2026: EXTRACT THE NOISE-AREA STATE AND USE IT AS CONTEXT ON THE LIVE LEGS. DECLARED BEFORE RESULTS.
At each RA / V15a trade's entry bar, tag the noise state: price outside the upper band (UB) / lower band (LB) / inside, and whether it is outside VWAP-side with the trade (WITH) or against it. Report points per trade by state (2014-21 | 2022-26), t of with-minus-against; PASS = same sign and |t|>=2 both periods. Then the sizing test: x1.25 when WITH, x0.75 when AGAINST on system pts/day, std, mean/std (RA + V15a only, sz weighted)."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie/xs'); import nz_core as Z
from nz_core import A, ALLD, GRID, OPEN, PC, SIG, SI
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad'; BM = dict(pickle.load(open(f'{SP}/base_mv30.pkl', 'rb')))
OUT, NAMES = pickle.load(open(f'{SP}/s6.pkl', 'rb'))
def state(d, m):
    if d not in A or d not in SI: return np.nan, np.nan
    jb = int((m - 570) // 5) - 1; o, h, l, c, v, vw = A[d][0]
    if jb < 1 or jb >= len(c) or np.isnan(c[jb]): return np.nan, np.nan
    sg = SIG[SI[d]]; 
    if not np.isfinite(sg[jb]): return np.nan, np.nan
    op = OPEN[d]; pc = PC[d] if np.isfinite(PC[d]) else op; ub, lb = max(op, pc) * (1 + sg[jb]), min(op, pc) * (1 - sg[jb]); s_ = op * sg[-1]
    return (1 if c[jb] > ub else -1 if c[jb] < lb else 0), (c[jb] - vw[jb]) / s_
def tt(a, b): return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)) if len(a) > 8 and len(b) > 8 else np.nan
rows = {}
for (t, u, y), n in zip(OUT['V2'], NAMES):
    if n not in ('RA', 'V15a'): continue
    t = t.copy(); t['day'] = pd.to_datetime(t.day).dt.normalize(); B = BM[n].copy(); B['day'] = pd.to_datetime(B.day).dt.normalize(); B = B[B.sz > 0][['day', 'm', 'sz']].rename(columns={'sz': 'szb'}); t = t.merge(B, on=['day', 'm']); t['sz'] = t.szb
    t = t[(t.day >= '2014-02-01') & (t.day <= '2026-01-30')].reset_index(drop=True); st = [state(d, m) for d, m in zip(t.day, t.m)]; t['band'] = [s[0] for s in st]; t['vwd'] = [s[1] for s in st]; t = t.dropna(subset=['band'])
    t['rel'] = np.where(t.band == 0, 'inside', np.where(t.band * t.dir > 0, 'WITH', 'AGAINST')); t['vrel'] = np.where(t.vwd * t.dir > 0, 'vwap-with', 'vwap-against'); rows[n] = t
    print(f'\\n{n}: {len(t)} trades; points per trade by noise-band state (2014-21 | 2022-26)')
    for col in ('rel', 'vrel'):
        for k in sorted(t[col].unique()):
            a = [t[(t[col] == k) & (t.day < '2022-01-01')].pts, t[(t[col] == k) & (t.day >= '2022-01-01')].pts]; print(f'   {k:<13} {a[0].mean():+6.1f} (n{len(a[0])}) | {a[1].mean():+6.1f} (n{len(a[1])})')
    for col, hi, lo in (('rel', 'WITH', 'AGAINST'), ('vrel', 'vwap-with', 'vwap-against')):
        r = [(t[(t[col] == hi) & f].pts.mean() - t[(t[col] == lo) & f].pts.mean(), tt(t[(t[col] == hi) & f].pts, t[(t[col] == lo) & f].pts)) for f in (t.day < '2022-01-01', t.day >= '2022-01-01')]
        ok = np.sign(r[0][0]) == np.sign(r[1][0]) and all(abs(x[1]) >= 2 for x in r); print(f'   {hi} minus {lo}: {r[0][0]:+.1f} (t {r[0][1]:+.2f}) | {r[1][0]:+.1f} (t {r[1][1]:+.2f}){"   <== PASS" if ok else ""}')
T = pd.concat(rows.values()); 
for lab, f in (('base', lambda t: np.ones(len(t))), ('x1.25 WITH / x0.75 AGAINST', lambda t: np.where(t.rel == 'WITH', 1.25, np.where(t.rel == 'AGAINST', .75, 1.0))), ('drop AGAINST', lambda t: np.where(t.rel == 'AGAINST', 0.0, 1.0))):
    o = []
    for nm, m in (('14-21', T.day < '2022-01-01'), ('22-26', T.day >= '2022-01-01')):
        x = T[m]; s = (x.pts * x.sz * f(x)).groupby(x.day).sum(); days = pd.DatetimeIndex(sorted(set(x.day))); s = s.reindex(days).fillna(0); o.append(f'{nm}: {s.mean():+6.2f}/d std {s.std():5.1f} m/s {s.mean() / s.std():.3f}')
    print(f'  {lab:<30} ' + ' | '.join(o))
print('\nWITH-band minus INSIDE-band (points per trade), t:')
for n, t in rows.items():
    r = [(t[(t.rel == 'WITH') & f].pts.mean() - t[(t.rel == 'inside') & f].pts.mean(), tt(t[(t.rel == 'WITH') & f].pts, t[(t.rel == 'inside') & f].pts)) for f in (t.day < '2022-01-01', t.day >= '2022-01-01')]
    ok = np.sign(r[0][0]) == np.sign(r[1][0]) and all(abs(x[1]) >= 2 for x in r); print(f'  {n}: {r[0][0]:+.1f} (t {r[0][1]:+.2f}) | {r[1][0]:+.1f} (t {r[1][1]:+.2f}){"   <== PASS" if ok else ""}')
for lab, f in (('x1.25 WITH / x0.6 inside', lambda t: np.where(t.rel == 'WITH', 1.25, .6)),):
    o = []
    for nm, m in (('14-21', T.day < '2022-01-01'), ('22-26', T.day >= '2022-01-01')):
        x = T[m]; s = (x.pts * x.sz * f(x)).groupby(x.day).sum(); days = pd.DatetimeIndex(sorted(set(x.day))); s = s.reindex(days).fillna(0); o.append(f'{nm}: {s.mean():+6.2f}/d std {s.std():5.1f} m/s {s.mean() / s.std():.3f}')
    print(f'  {lab:<30} ' + ' | '.join(o))
