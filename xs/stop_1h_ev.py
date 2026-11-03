import sys, os, zlib, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from collections import deque
V = pd.read_parquet('/tmp/claude-0/ml/stop1h_V.parquet'); Xo = pd.read_parquet('/tmp/claude-0/ml/noise_X.parquet')
W = pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/wide_T.parquet'); W['d'] = W.entry.dt.normalize()
RT = W.groupby(['tk', 'd']).agg(ratio=('ratio', 'first'), sp=('sp', 'first')).reset_index().sort_values('d')
def build(var):
    S = pd.concat([V[V['var'] == var], V[V['var'] == 'H1']]); S = S[(S.entry >= '2019-01-01') & ((S.entry.dt.hour * 60 + S.entry.dt.minute) < 840)].copy(); S['d'] = S.entry.dt.normalize()
    S = pd.merge_asof(S.sort_values('d'), RT, on='d', by='tk', direction='nearest', tolerance=pd.Timedelta(days=5))
    return S[(S.sp == True) & (S.ratio >= 1.5)].sort_values('entry').reset_index(drop=True)
XL = build('L'); print('parity: live-rebuilt trades', len(XL), 'vs stored', len(Xo), ' mean ret', f'{XL.ret.mean():+.4%} vs {Xo.ret.mean():+.4%}')
IDX = pd.bdate_range('2019-01-01', Xo.exit.max()); P2 = [(np.datetime64('2019-01-01'), np.datetime64('2022-01-01')), (np.datetime64('2022-01-01'), np.datetime64('2027-01-01'))]
def stat2(s):
    eq = s.groupby(s.index.normalize()).last().reindex(IDX, method='ffill').fillna(0); out = []
    for a, b in P2:
        e = eq[(eq.index >= a) & (eq.index < b)]; e = np.exp(e - e.iloc[0]); yr = (e.index[-1] - e.index[0]).days / 365.25
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
def ev(Xv):
    HSH = np.array([zlib.crc32(f'{a}{b}'.encode()) for a, b in zip(Xv.tk, Xv.entry.astype('int64'))], dtype=np.uint64); K = Xv.ratio.values.astype(float)
    res = []
    for i in range(1, 41):
        dl = ((HSH * 2654435761 + i * 40503) % 1000) < 50; m = ~dl
        res.append(stat2(sim(Xv.entry.values[m], Xv.exit.values[m], Xv.ret.values[m], K[m], Xv.tk.values[m])))
    return np.array(res)
def line(nm, a, b=None):
    m = np.median(a, axis=0); s = f'  {nm:<34} ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
    if b is not None:
        w1, w2 = (a[:, 2] > b[:, 2]).mean(), (a[:, 5] > b[:, 5]).mean(); s += f'   beats live {w1:.0%} / {w2:.0%}' + ('  PASS' if w1 >= .7 and w2 >= .7 else '')
    print(s, flush=True)
B0 = ev(XL); print('40 matched runs, 2019-21 / 2022-26'); line('L live (rebuilt)', B0)
for nm in ('A3', 'A5', 'Ainf', 'G4', 'G8'):
    Xv = build(nm); h = Xv[~Xv.h1]; print(f'  {nm}: trades {len(Xv)}, 15m avg trade {h.ret.mean():+.3%} (live {XL[~XL.h1].ret.mean():+.3%}), runners>=15% {(h.ret >= .15).sum()} (live {(XL[~XL.h1].ret >= .15).sum()}), avg hold days {((h.exit - h.entry).dt.total_seconds() / 86400).mean():.1f} (live {((XL[~XL.h1].exit - XL[~XL.h1].entry).dt.total_seconds() / 86400).mean():.1f})')
    line(nm, ev(Xv), B0)
