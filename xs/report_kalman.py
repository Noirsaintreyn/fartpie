"""KALMAN (causal local-linear-trend filter on intraday 5m closes, params fixed in advance, no tuning) as entry-time features on V15a vector flips, full history.
sigma_bar = OPEN*SIG[-1]/sqrt(NB). Fast: q_level=(.3s)^2 q_slope=(.03s)^2 ; Slow: q_level=(.15s)^2 q_slope=(.01s)^2 ; R=s^2.
Features signed with the flip direction: slope (dir*b/s), innov (dir*(c-level)/s), kvw (dir*(level-vwap)/(OPEN*SIG[-1])). Terciles cut on 2018-21. PASS = same sign & |t|>=2 in 2018-21 and 2022-26. Nov 11 2026."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/fartpie/xs'); import nz_core as Z
from nz_core import A, SIG, SI, OPEN, NB
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
O, N = pickle.load(open(SC + 's6.pkl', 'rb')); T = dict(zip(N, O['V2']))['V15a'][0]; T = T[T.sz > 0].copy(); T['day'] = pd.to_datetime(T.day)
def kalman(c, s, ql, qb):
    F = np.array([[1., 1.], [0., 1.]]); Q = np.diag([(ql * s) ** 2, (qb * s) ** 2]); Rr = s ** 2
    x = np.array([c[~np.isnan(c)][0], 0.]); P = np.diag([s ** 2, (qb * s) ** 2 * 10]); L = np.full(len(c), np.nan); Bn = np.full(len(c), np.nan)
    for j in range(len(c)):
        x = F @ x; P = F @ P @ F.T + Q
        if not np.isnan(c[j]):
            y = c[j] - x[0]; S = P[0, 0] + Rr; K = P[:, 0] / S; x = x + K * y; P = P - np.outer(K, P[0])
        L[j], Bn[j] = x[0], x[1]
    return L, Bn
PAR = {'fast': (.3, .03), 'slow': (.15, .01)}; cache = {}
rows = []
for r in T.itertuples():
    d = r.day; jb = int((int(r.m) - 570) // 5) - 1
    if d not in A or d not in SI or jb < 6: continue
    s_day = OPEN[d] * SIG[SI[d]][-1]
    if not np.isfinite(s_day) or s_day <= 0: continue
    o_, h_, l_, c_, v_, vw_ = A[d][0]
    if np.isnan(c_[jb]): continue
    sb = s_day / np.sqrt(NB); f = dict(day=d, m=int(r.m), pts=r.pts)
    for nm, (ql, qb) in PAR.items():
        if (d, nm) not in cache: cache[(d, nm)] = kalman(c_, sb, ql, qb)
        L, Bn = cache[(d, nm)]
        f[f'slope_{nm}'] = r.dir * Bn[jb] / sb; f[f'innov_{nm}'] = r.dir * (c_[jb] - L[jb]) / sb; f[f'kvw_{nm}'] = r.dir * (L[jb] - vw_[jb]) / s_day
    rows.append(f)
R = pd.DataFrame(rows).dropna(); R['era'] = np.where(R.day < '2022-01-01', 'A', 'B'); Aera = R[R.era == 'A']
print(f'V15a trades with Kalman features: {len(R)} | 2018-21 {len(Aera)} | 2022-26 {np.sum(R.era=="B")} | mean pts {R.pts.mean():+.2f}')
def tstat(x): return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 8 and x.std() > 0 else np.nan
print(f'{"feature":14} {"group":10} | {"nA":>4} {"A mean":>7} {"A t":>5} | {"nB":>4} {"B mean":>7} {"B t":>5} |')
good = []
for c in [x for x in R.columns if x.split('_')[0] in ('slope', 'innov', 'kvw')]:
    q = Aera[c].quantile([1 / 3, 2 / 3]).values
    for nm, sel in (('low third', R[c] <= q[0]), ('mid', (R[c] > q[0]) & (R[c] <= q[1])), ('high third', R[c] > q[1])):
        a = R[sel & (R.era == 'A')].pts; b = R[sel & (R.era == 'B')].pts
        ok = len(a) >= 30 and len(b) >= 30 and np.sign(a.mean()) == np.sign(b.mean()) and abs(tstat(a)) >= 2 and abs(tstat(b)) >= 2
        print(f'{c:14} {nm:10} | {len(a):4d} {a.mean():+7.2f} {tstat(a):+5.1f} | {len(b):4d} {b.mean():+7.2f} {tstat(b):+5.1f} | {"PASS" if ok else ""}')
        if ok: good.append((c, nm))
print('\nPASS list:', good)
R.to_pickle(SC + 'kalman_feats.pkl')
