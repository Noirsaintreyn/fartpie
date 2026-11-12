"""SLOT CONTENTION & LEVERAGE. When more flips fire than the 6 slots, who gets skipped and does it cost? Then: more slots (exposure rises when signals cluster) vs same-exposure split vs flat leverage.
Same engine (ratio tie-break, 8% breaker, entries before 14:00), 40 matched runs, 2019-21 / 2022-26. Universes: live S&P book and S&P+Nasdaq-100 (current NDX list: hindsight caveat)."""
import numpy as np, pandas as pd, zlib, warnings; warnings.filterwarnings('ignore')
from collections import deque
a = open('xs/stop_1h_ev.py').read(); a = a[a.index('IDX = '):a.index('B0 = ev(XL)')].replace("Xo.exit.max()", "'2026-10-21'")
exec(a)
u = open('xs/universe_test.py').read(); u = u[u.index('NDX = set'):u.index("print('parity")]
exec(u.replace("print('unclassified", "pass #"))
def sim2(E, Xx, R, K, TK, N=6, D=6.0, lev=1.0, rec=None):
    o = np.lexsort((TK, -K, E)); op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0; rk = []
    for idx, e, x, r in zip(o, E[o], Xx[o], R[o]):
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr in done: real += np.log1p(rr * lev / D); hist.append((xt, real)); win.append((xt, real))
        op = [q for q in op if q > e]
        if len(op) >= N:
            if rec is not None: rec.append((e, r, 'full'))
            continue
        if hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-.08):
                if rec is not None: rec.append((e, r, 'breaker'))
                continue
        op.append(x); pend.append((x, r))
        if rec is not None: rec.append((e, r, 'taken'))
    for xt, rr in sorted(pend, key=lambda z: z[0]): real += np.log1p(rr * lev / D); hist.append((xt, real))
    return pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index()
def stat_s(s):
    eq = s.groupby(s.index.normalize()).last().reindex(IDX, method='ffill').fillna(0); out = []
    for a_, b_ in P2:
        e = eq[(eq.index >= a_) & (eq.index < b_)]; e = np.exp(e - e.iloc[0]); yr = (e.index[-1] - e.index[0]).days / 365.25
        c = e.iloc[-1] ** (1 / yr) - 1; dd = (e / e.cummax() - 1).min(); out += [c, dd, c / -dd]
    return out
def ev2(Xv, **kw):
    HSH = np.array([zlib.crc32(f'{x}{y}'.encode()) for x, y in zip(Xv.tk, Xv.entry.astype('int64'))], dtype=np.uint64); K = Xv.ratio.values.astype(float); res = []
    for i in range(1, 41):
        m = ~(((HSH * 2654435761 + i * 40503) % 1000) < 50); res.append(stat_s(sim2(Xv.entry.values[m], Xv.exit.values[m], Xv.ret.values[m], K[m], Xv.tk.values[m], **kw)))
    return np.array(res)
def line2(nm, a, b=None):
    m = np.median(a, axis=0); s = f'  {nm:<46} ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
    if b is not None: s += f'   ret/DD beats live {(a[:,2]>b[:,2]).mean():.0%} / {(a[:,5]>b[:,5]).mean():.0%}'
    print(s, flush=True)
UNI = {'S&P live book': W[W.sp].reset_index(drop=True), 'S&P + Nasdaq-100': W[W.sp | W.ndx].reset_index(drop=True)}
for un, Xv in UNI.items():
    print(f'\n######## {un}: {len(Xv)} flips', flush=True)
    rec = []; sim2(Xv.entry.values, Xv.exit.values, Xv.ret.values, Xv.ratio.values.astype(float), Xv.tk.values, rec=rec)
    R = pd.DataFrame(rec, columns=['e', 'r', 'st']); R['era'] = np.where(R.e < np.datetime64('2022-01-01'), '19-21', '22-26')
    print('who gets taken vs skipped (no deletions, 6 slots): count, mean ret, win%')
    for era in ('19-21', '22-26'):
        print(f'   {era}: ' + ' | '.join(f'{k}: n {len(g)} avg {g.r.mean()*100:+.2f}% win {(g.r>0).mean():.0%}' for k, g in R[R.era == era].groupby('st')), flush=True)
    full = R[R.st == 'full']; print(f'   share of flips skipped because all slots full: {len(full)/len(R):.0%}; skipped by breaker: {(R.st=="breaker").mean():.0%}', flush=True)
    B = ev2(Xv); line2('6 slots, size 1/6 (live)', B)
    for N in (8, 10, 12): line2(f'{N} slots, size 1/6 each (exposure up to {N/6:.2f}x)', ev2(Xv, N=N, D=6.0), B)
    for N in (8, 10, 12): line2(f'{N} slots, size 1/{N} each (same total exposure)', ev2(Xv, N=N, D=float(N)), B)
    for L in (1.25, 1.5, 2.0): line2(f'6 slots, flat leverage x{L}', ev2(Xv, lev=L), B)
    line2('8 slots size 1/6 + breaker as live, lev x1.0', ev2(Xv, N=8), B)
