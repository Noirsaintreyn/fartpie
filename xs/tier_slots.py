"""Nov 4 2026: tiered slots - empty slots wait for an A+ trade. DECLARED BEFORE RESULTS. Same 6 x 1/6 sizing (unused slots = cash), 8% breaker, live exits.
Slots 1..k fill from any signal (ratio >= 1.5 universe); slots k+1..6 open ONLY when the signal's options ratio >= T (A+).
 k in {3, 4, 5}; T = 75th / 90th percentile of the ratios of the 2019-21 signals (so the bar is fixed from the first period).
 Also 'half-size B': slots k+1..6 open for any signal at half size (1/12) and A+ at full size (1/6)  [k=4, T=p75].
40 matched runs vs live both periods; pass = >= 70% beat live in both."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('xs/stop_1h_ev.py').read(); exec(src[:src.index("B0 = ev(XL)")])
r1 = XL[XL.entry < '2022-01-01'].ratio; P75, P90 = r1.quantile(.75), r1.quantile(.90); print(f'A+ bars from 2019-21 ratios: p75 {P75:.2f}  p90 {P90:.2f}  (median {r1.median():.2f})')
def simt(E, Xx, R, K, TK, k, T, half=False, n=6):
    o = np.lexsort((TK, -K, E)); op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0
    for ii in o:
        e, x, r, kk = E[ii], Xx[ii], R[ii], K[ii]
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr in done: real += np.log1p(rr / n); hist.append((xt, real)); win.append((xt, real))
        op = [q for q in op if q[0] > e]
        if len(op) >= n: continue
        if hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-.08): continue
        w = 1.0
        if len(op) >= k and kk < T:
            if not half: continue
            w = .5
        op.append((x, w)); pend.append((x, r * w))
    for xt, rr in sorted(pend, key=lambda z: z[0]): real += np.log1p(rr / n); hist.append((xt, real))
    return pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index()
def evt(Y, k, T, half=False):
    HSH = np.array([zlib.crc32(f'{a}{b}'.encode()) for a, b in zip(Y.tk, Y.entry.astype('int64'))], dtype=np.uint64); K = Y.ratio.values.astype(float); res = []
    for i in range(1, 41):
        dl = ((HSH * 2654435761 + i * 40503) % 1000) < 50; m = ~dl
        res.append(stat2(simt(Y.entry.values[m], Y.exit.values[m], Y.ret.values[m], K[m], Y.tk.values[m], k, T, half)))
    return np.array(res)
B0r = evt(XL, 6, 0); print('40 matched runs, 2019-21 / 2022-26'); line('L live', B0r)
for k in (3, 4, 5):
    for nm, T in (('p75', P75), ('p90', P90)): line(f'k={k} slots any, rest A+ >= {nm} ({T:.1f})', evt(XL, k, T), B0r)
line('k=4, rest half-size unless A+ (p75)', evt(XL, 4, P75, True), B0r)
