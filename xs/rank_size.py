"""Nov 4 2026: rank-weighted sizing instead of equal slots. DECLARED BEFORE RESULTS. Budget fixed (sum of open weights <= n; no leverage).
At entry the new position is ranked against the positions already open (1 = best); its weight is fixed for the trade: weight = ladder[rank-1], the last rung for rank >= n.
Rank key: R options ratio | V 20d realized vol (high vol first) | RV average of the ratio rank and the vol rank | IV control: LOW vol first (less risky in front).
Ladders (mean 1): mild [1.3,1.15,1.05,.95,.85,.7], strong [1.6,1.3,1.1,.9,.7,.4]. Entry order, 8% breaker and slot count as live (6) unless 'n8' (8 slots, ladder stretched).
40 matched runs vs live (equal slots), both periods; pass = >=70% of runs beat live ret/DD in both periods."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('xs/stop_1h_ev.py').read(); exec(src[:src.index("B0 = ev(XL)")])
C = pd.read_parquet('/tmp/claude-0/ml/ctx_rank_X.parquet')[['tk', 'entry', 'h1', 'rv']]; Z = XL.merge(C, on=['tk', 'entry', 'h1'], how='left'); Z['rv'] = Z.rv.fillna(Z.rv.median())
def simw(E, Xx, R, K, TK, RK, lad, n=6):
    o = np.lexsort((TK, -K, E)); op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; lad = np.asarray(lad, float)
    for ii in o:
        e, x, r, rk = E[ii], Xx[ii], R[ii], RK[ii]
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr in done: real += np.log1p(rr / n); hist.append((xt, real)); win.append((xt, real))
        op = [q for q in op if q[0] > e]
        if len(op) >= n: continue
        if hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-.08): continue
        rank = 1 + sum(1 for q in op if q[1] > rk); w = lad[min(rank - 1, len(lad) - 1)]; used = sum(q[2] for q in op)
        w = min(w, n - used)
        if w < .2: continue
        op.append((x, rk, w)); pend.append((x, r * w))
    for xt, rr in sorted(pend, key=lambda z: z[0]): real += np.log1p(rr / n); hist.append((xt, real))
    return pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index()
def evw(Y, RKf, lad, n=6):
    HSH = np.array([zlib.crc32(f'{a}{b}'.encode()) for a, b in zip(Y.tk, Y.entry.astype('int64'))], dtype=np.uint64); K = Y.ratio.values.astype(float); RK = RKf(Y)
    res = []
    for i in range(1, 41):
        dl = ((HSH * 2654435761 + i * 40503) % 1000) < 50; m = ~dl
        res.append(stat2(simw(Y.entry.values[m], Y.exit.values[m], Y.ret.values[m], K[m], Y.tk.values[m], RK[m], lad, n)))
    return np.array(res)
pr = lambda s: s.rank(pct=True).values
KEYS = {'R ratio': lambda Y: Y.ratio.values.astype(float), 'V high vol': lambda Y: Y.rv.values.astype(float), 'RV both': lambda Y: pr(Y.ratio) + pr(Y.rv), 'IV low vol first': lambda Y: -Y.rv.values.astype(float)}
LAD = {'mild': [1.3, 1.15, 1.05, .95, .85, .7], 'strong': [1.6, 1.3, 1.1, .9, .7, .4]}
B0 = evw(Z, KEYS['R ratio'], [1] * 6); print('40 matched runs, 2019-21 / 2022-26'); line('L live (equal slots)', B0)
for ln, lad in LAD.items():
    for kn, kf in KEYS.items(): line(f'{ln} | {kn}', evw(Z, kf, lad), B0)
L8 = [1.5, 1.3, 1.15, 1.0, .9, .8, .7, .65]; B8 = evw(Z, KEYS['R ratio'], [1] * 8, 8); line('8 slots equal', B8, B0)
line('8 slots ladder | R ratio', evw(Z, KEYS['R ratio'], L8, 8), B0); line('8 slots ladder | RV both', evw(Z, KEYS['RV both'], L8, 8), B0)
