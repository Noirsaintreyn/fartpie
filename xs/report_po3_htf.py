"""PO3 sweeps by level: prior day (PD), pre-market session (PS, 06:00-09:25 ET same day), prior week (PW), prior month (PM). Nov 11 2026.
Sweep = a closed 5m bar wicks beyond the level and closes back inside (before entry bar). High swept -> distribution DOWN (-1); low swept -> UP (+1); latest sweep wins within a level.
aligned = dir*state per level. Also 'ANY HTF' (PW or PM) and 'ANY' (any of the four, latest). Leverage test: 2x size on aligned trades vs 1x."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie/xs'); import nz_core as Z
from nz_core import A, ALLD
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
O, N = pickle.load(open(SC + 's6.pkl', 'rb')); L = dict(zip(N, O['V2']))
R_ = pd.read_parquet('/tmp/claude-0/ml/nq5_real.parquet'); R_['day'] = pd.to_datetime(R_.day).dt.normalize()
PS = {}
for d, g in R_[(~R_.rth) & (R_.m >= 360) & (R_.m < 570)].groupby('day'):
    if len(g) > 20: PS[d] = (g.high.max(), g.low.min())
DH = {d: np.nanmax(A[d][0][1]) for d in ALLD}; DL = {d: np.nanmin(A[d][0][2]) for d in ALLD}
dd = pd.DataFrame({'h': DH, 'l': DL}); dd.index = pd.to_datetime(dd.index)
wk = dd.groupby(dd.index.to_period('W')).agg(h=('h', 'max'), l=('l', 'min')); mo = dd.groupby(dd.index.to_period('M')).agg(h=('h', 'max'), l=('l', 'min'))
def lvl(d, kind):
    d = pd.Timestamp(d); i = SI = ALLD.index(d) if False else None
    if kind == 'PD':
        k = LI.get(d); return (DH[ALLD[k - 1]], DL[ALLD[k - 1]]) if k else None
    if kind == 'PS': return PS.get(d)
    if kind == 'PW':
        p = d.to_period('W') - 1; return (wk.h[p], wk.l[p]) if p in wk.index else None
    if kind == 'PM':
        p = d.to_period('M') - 1; return (mo.h[p], mo.l[p]) if p in mo.index else None
LI = {d: i for i, d in enumerate(ALLD)}
def sweeps(d, jb, kind):
    lv = lvl(d, kind)
    if lv is None: return []
    H, Lw = lv; o, h, l, c, v, vw = A[d][0]; ev = []
    for q in range(0, jb + 1):
        if np.isnan(c[q]): continue
        if h[q] > H and c[q] < H: ev.append((q, -1))
        if l[q] < Lw and c[q] > Lw: ev.append((q, +1))
    return ev
def st(ev): return max(ev)[1] if ev else 0
def tstat(x): x = np.asarray(x, float); return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 8 and x.std() > 0 else np.nan
KINDS = ['PD', 'PS', 'PW', 'PM']
for nm in ('V15a', 'RA'):
    T = L[nm][0]; T = T[T.sz > 0].copy(); T['day'] = pd.to_datetime(T.day); rows = []
    for r in T.itertuples():
        d = r.day
        if d not in A: continue
        jb = int((int(r.m) - 570) // 5) - 1
        if jb < 2: continue
        S = {k: sweeps(d, jb, k) for k in KINDS}; row = dict(day=d, pts=r.pts, dir=int(r.dir))
        for k in KINDS: row[k] = int(r.dir) * st(S[k])
        row['HTF'] = int(r.dir) * st(S['PW'] + S['PM']); row['ANY'] = int(r.dir) * st(sum(S.values(), []))
        row['n_with'] = sum(1 for k in KINDS if row[k] == 1); rows.append(row)
    R = pd.DataFrame(rows); R['era'] = np.where(R.day < '2022-01-01', 'A', 'B')
    print(f'\n== {nm}: {len(R)} trades, mean {R.pts.mean():+.1f}, win {np.mean(R.pts>0):.0%}')
    for col in KINDS + ['HTF', 'ANY']:
        for v, lab in ((1, 'WITH distrib'), (0, 'no sweep'), (-1, 'AGAINST')):
            s = R[R[col] == v]; a = s[s.era == 'A'].pts; b = s[s.era == 'B'].pts
            ok = len(a) >= 30 and len(b) >= 30 and np.sign(a.mean()) == np.sign(b.mean()) and abs(tstat(a)) >= 2 and abs(tstat(b)) >= 2
            print(f' {col:4} {lab:13} n {len(s):4d} win {np.mean(s.pts>0) if len(s) else 0:3.0%} mean {s.pts.mean():+6.1f} | A n{len(a):3d} {a.mean():+6.1f} ({tstat(a):+.1f}) | B n{len(b):3d} {b.mean():+6.1f} ({tstat(b):+.1f}) {"PASS" if ok else ""}')
    print(' -- confluence: # of levels swept in the trade direction (WITH)')
    for k in (0, 1, 2, 3):
        s = R[R.n_with == k] if k < 3 else R[R.n_with >= 3]
        if len(s): print(f'   with-count {k}{"+" if k==3 else ""}: n {len(s):4d} win {np.mean(s.pts>0):3.0%} mean {s.pts.mean():+6.1f} A {s[s.era=="A"].pts.mean():+.1f} B {s[s.era=="B"].pts.mean():+.1f}')
    print(' -- leverage: 2x on WITH-any-level (n_with>=1), 1x otherwise')
    w = R.n_with >= 1; x = np.where(w, 2 * R.pts, R.pts)
    for lab, y in (('1x all', R.pts.values), ('2x aligned', x)):
        y = pd.Series(y, index=R.index); ey = y.groupby(R.day.dt.year).sum()
        print(f'   {lab:11} total {y.sum():+8.0f} mean {y.mean():+5.1f} std {y.std():5.1f} worst {y.min():+6.0f} yrs-neg {int((ey<0).sum())}/{len(ey)}')
    R.to_pickle(SC + f'po3htf_{nm}.pkl')
