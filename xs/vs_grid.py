"""Nov 2 2026: Q3 entry/exit attribution grid (diagnostic controls, not strategies). Same universe, budget, 6 slots, breaker, costs.
 VV  vector entry + vector exit (live)
 VR  vector entry + random-duration exit: hold for a length drawn from the live holding-length distribution of the same track,
     independent of the price path (isolates what the vector's EXIT adds)
 RV  random entry (same name, random 9:30-13:45 bar within +-60 days) + vector exit: the live exit engine run from that bar
     (sticky 1h stop, +5% lock; exits at the first bar the track's state is bearish) -> isolates what the vector's ENTRY adds
 RR  random entry + random-duration exit (= control C2)
40 runs, matched signal drops; random draws seeded by run index. 2019-21 / 2022-26."""
import sys, os, glob, zlib, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from collections import deque
HERE = '/home/claude/fartpie/xs'; sys.path.insert(0, HERE); _a = sys.argv; sys.argv = sys.argv[:1] + ['x']
exec(open(os.path.join(HERE, 'signal_lab.py')).read().split("if sys.argv[1] == '1':")[0]); sys.argv = _a
X = pd.read_parquet('/tmp/claude-0/ml/noise_X.parquet').sort_values('entry').reset_index(drop=True)
D = {}
for tk in X.tk.unique():
    H = load(glob.glob(f'/mnt/user-data/uploads/fartpie/lse_clean/{tk}_15m_*.parquet')[0]); reg, s1 = states(H, vec)
    g = np.array([j for j in range(1, len(H)) if 570 <= H.m.values[j] < 960])
    D[tk] = dict(g=g, t=H.dt.values[g], o=H.open.values[g], c=H.close.values[g - 1], hi=H.high.values[np.r_[g[0], g[:-1]]], m=H.m.values[g], KR=np.asarray(reg)[g - 1], KS=np.asarray(s1)[g - 1])
print('states built', flush=True)
def vexit(d, p0, h1track):
    """live exit engine from RTH-bar position p0 (entry at its open); returns exit position"""
    px = d['o'][p0]; peak = px; b1 = False; h1 = False
    for p in range(p0 + 1, len(d['t'])):
        if p - 1 >= p0: peak = max(peak, d['hi'][p])
        b1 = b1 or d['KS'][p] == 1; h1 = h1 or (d['c'][p] / px - 1 >= .02 and b1)
        ex = d['KS'][p] == -1 if h1track else ((d['KS'][p] == -1) if h1 else (d['KR'][p] == -1)); pg = peak / px - 1
        if pg >= .05 and (d['c'][p] / px - 1) <= .5 * pg: ex = True
        if ex: return p
    return len(d['t']) - 1
POS = np.zeros(len(X), int); NB = np.zeros(len(X), int); ok = np.zeros(len(X), bool)
for i, (tk, e, x) in enumerate(zip(X.tk, X.entry.values, X.exit.values)):
    d = D[tk]; a, b = np.searchsorted(d['t'], e), np.searchsorted(d['t'], x)
    if a < len(d['t']) and b < len(d['t']) and d['t'][a] == e and b > a: POS[i], NB[i], ok[i] = a, b - a, True
X = X[ok].reset_index(drop=True); POS, NB = POS[ok], NB[ok]
# parity: the vexit engine must reproduce the live exits from the live entries
chk = [vexit(D[tk], p, h) - p == nb for tk, p, h, nb in zip(X.tk.values[:3000], POS[:3000], X.h1.values[:3000], NB[:3000])]
print(f'exit-engine parity on 3000 live trades: {np.mean(chk):.1%} identical', flush=True)
NB15, NB1h = NB[~X.h1.values], NB[X.h1.values]
def draw(kind, rng):
    E, Xx, R = X.entry.values.copy(), X.exit.values.copy(), X.ret.values.copy()
    for i in range(len(X)):
        d = D[X.tk.values[i]]; h1t = X.h1.values[i]
        if kind in ('RV', 'RR'):
            c = POS[i]; lo, hi = np.searchsorted(d['t'], d['t'][c] - np.timedelta64(60, 'D')), np.searchsorted(d['t'], d['t'][c] + np.timedelta64(60, 'D'))
            hi = min(hi, len(d['t']) - 300)
            if hi <= lo: continue
            for _ in range(20):
                a = rng.integers(lo, hi)
                if d['m'][a] < 840: break
        else: a = POS[i]
        if kind == 'RV': b = vexit(d, a, h1t)
        elif kind == 'RR': b = a + NB[i]
        else: b = a + (NB1h if h1t else NB15)[rng.integers(len(NB1h) if h1t else len(NB15))]
        b = min(b, len(d['t']) - 1)
        if b <= a: continue
        E[i], Xx[i], R[i] = d['t'][a], d['t'][b], d['o'][b] / d['o'][a] - 1 - C2
    return E, Xx, R
HSH = np.array([zlib.crc32(f'{a}{b}'.encode()) for a, b in zip(X.tk, X.entry.astype('int64'))], dtype=np.uint64)
DROPS = [((HSH * 2654435761 + i * 40503) % 1000) < 50 for i in range(1, 41)]
IDX = pd.bdate_range('2019-01-01', '2027-06-30')
P2 = [(np.datetime64('2019-01-01'), np.datetime64('2022-01-01')), (np.datetime64('2022-01-01'), np.datetime64('2027-01-01'))]
def stat2(s):
    eq = s.groupby(s.index.normalize()).last().reindex(IDX, method='ffill').fillna(0); out = []
    for a, b in P2:
        e = eq[(eq.index >= a) & (eq.index < b)]; e = np.exp(e - e.iloc[0]); e = e[e.index <= s.index.max()]; yr = (e.index[-1] - e.index[0]).days / 365.25
        c = e.iloc[-1] ** (1 / yr) - 1; dd = (e / e.cummax() - 1).min(); out += [c, dd, c / -dd]
    return out
def sim(E, Xx, R, K, TK):
    o = np.lexsort((TK, -K, E)); op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0
    for e, x, r in zip(E[o], Xx[o], R[o]):
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr in done: real += np.log1p(rr / 6); hist.append((xt, real)); win.append((xt, real))
        op = [q for q in op if q > e]
        if len(op) >= 6: continue
        if hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-.08): continue
        op.append(x); pend.append((x, r))
    for xt, rr in sorted(pend, key=lambda z: z[0]): real += np.log1p(rr / 6); hist.append((xt, real))
    return pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index()
K = X.ratio.values.astype(float); TK = X.tk.values
L0 = None
for kind, nm in (('VV', 'vector entry + vector exit (live)'), ('VR', 'vector entry + random-length exit'), ('RV', 'random entry + vector exit'), ('RR', 'random entry + random-length exit')):
    out, av = [], []
    for i, dl in enumerate(DROPS):
        E, Xx, R = (X.entry.values, X.exit.values, X.ret.values) if kind == 'VV' else draw(kind, np.random.default_rng(2000 + i))
        out.append(stat2(sim(E[~dl], Xx[~dl], R[~dl], K[~dl], TK[~dl]))); av.append(R.mean())
        if kind == 'VV': break
    a = np.array(out); m = np.median(a, axis=0)
    s = f'  {kind} {nm:<38} avg trade {np.mean(av):+.3%}   ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
    print(s, flush=True)
