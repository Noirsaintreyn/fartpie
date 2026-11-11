"""PO3 / AMD: trade only the DISTRIBUTION. Do flips (V15a, Rule A) that come AFTER a manipulation sweep and point the way of the distribution beat flips before any sweep / against it? Nov 11 2026.
State at entry (closed 5m bars only): MANIPULATION = a bar that wicks beyond the opening range (09:30-10:00) high/low (OR) or the prior-day high/low (PD) and CLOSES back inside. Sweep of the HIGH -> distribution DOWN (state -1); sweep of the LOW -> distribution UP (+1); latest sweep wins; none yet = accumulation (0).
Trades: s6.pkl V15a (0.15 ADR stop, flip exit) and RA (75/50, noon). aligned = dir*state: +1 with the distribution, -1 against it, 0 no sweep yet. Both eras (2018-21 / 2022-26)."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie/xs'); import nz_core as Z
from nz_core import A, ALLD
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
O, N = pickle.load(open(SC + 's6.pkl', 'rb')); L = dict(zip(N, O['V2']))
prev = {d: ALLD[i - 1] for i, d in enumerate(ALLD) if i > 0}
def state(d, jb, which):
    o, h, l, c, v, vw = A[d][0]; ev = []
    if which in ('OR', 'ANY') and jb >= 6:
        orh, orl = np.nanmax(h[:6]), np.nanmin(l[:6])
        for q in range(6, jb + 1):
            if np.isnan(c[q]): continue
            if h[q] > orh and c[q] < orh: ev.append((q, -1))
            if l[q] < orl and c[q] > orl: ev.append((q, +1))
    if which in ('PD', 'ANY') and d in prev and prev[d] in A:
        pdh, pdl = np.nanmax(A[prev[d]][0][1]), np.nanmin(A[prev[d]][0][2])
        for q in range(0, jb + 1):
            if np.isnan(c[q]): continue
            if h[q] > pdh and c[q] < pdh: ev.append((q, -1))
            if l[q] < pdl and c[q] > pdl: ev.append((q, +1))
    return max(ev)[1] if ev else 0          # latest sweep (ties: the later tuple)
def tstat(x): x = np.asarray(x, float); return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 8 and x.std() > 0 else np.nan
for nm in ('V15a', 'RA'):
    T = L[nm][0]; T = T[T.sz > 0].copy(); T['day'] = pd.to_datetime(T.day); rows = []
    for r in T.itertuples():
        d = r.day
        if d not in A: continue
        jb = int((int(r.m) - 570) // 5) - 1
        if jb < 2: continue
        rows.append(dict(day=d, m=int(r.m), pts=r.pts, aligned_or=int(r.dir) * state(d, jb, 'OR'), aligned_pd=int(r.dir) * state(d, jb, 'PD'), aligned=int(r.dir) * state(d, jb, 'ANY')))
    R = pd.DataFrame(rows); R['era'] = np.where(R.day < '2022-01-01', 'A', 'B')
    print(f'\n== {nm}: {len(R)} trades, mean {R.pts.mean():+.1f} pts, win {np.mean(R.pts>0):.0%}')
    for col, lab in (('aligned', 'ANY sweep (OR or prior-day)'), ('aligned_or', 'opening-range sweep'), ('aligned_pd', 'prior-day H/L sweep')):
        for v, nmv in ((1, 'WITH distribution'), (0, 'no sweep yet (accumulation)'), (-1, 'AGAINST distribution')):
            s = R[R[col] == v]; a = s[s.era == 'A'].pts; b = s[s.era == 'B'].pts
            ok = len(a) >= 30 and len(b) >= 30 and np.sign(a.mean()) == np.sign(b.mean()) and abs(tstat(a)) >= 2 and abs(tstat(b)) >= 2
            print(f' {lab:28} {nmv:28} n {len(s):4d} win {np.mean(s.pts>0) if len(s) else 0:3.0%} mean {s.pts.mean():+7.1f} | A n{len(a):3d} {a.mean():+6.1f} ({tstat(a):+.1f}) | B n{len(b):3d} {b.mean():+6.1f} ({tstat(b):+.1f}) {"PASS" if ok else ""}')
    R.to_pickle(SC + f'po3_{nm}.pkl')
