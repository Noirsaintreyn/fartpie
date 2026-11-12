"""SECOND SCREENER BOOK: live S&P book (6 slots, ratio>=1.5) + an independent 6-slot book on another universe, capital split. Same engine as universe_test.py (40 matched runs, ratio tie-break, 8% breaker, no buys after 14:00).
Book B candidates: Nasdaq-100 names NOT in point-in-time S&P (no overlap) | Non-S&P non-ETF | Nasdaq-100 full (overlap with A allowed) | Dow 30 (overlap) . IWM / Russell-2000 constituents: not in the options-ratio data (checked).
Combined equity = w*A + (1-w)*B (levels, daily), scored 2019-21 / 2022-26 like the single book. Reports daily-return correlation and share of B trades whose ticker A also holds."""
import numpy as np, pandas as pd, zlib, warnings; warnings.filterwarnings('ignore')
from collections import deque
a = open('xs/stop_1h_ev.py').read(); a = a[a.index('IDX = '):a.index('B0 = ev(XL)')].replace("Xo.exit.max()", "'2026-10-21'")
exec(a)
u = open('xs/universe_test.py').read(); u = u[u.index('NDX = set'):u.index("print('parity")]
exec(u.replace("print('unclassified", "pass #"))
print('IWM in data:', 'IWM' in set(W.tk), '| tickers total', W.tk.nunique(), flush=True)
def deq(Xv, i):
    HSH = np.array([zlib.crc32(f'{x}{y}'.encode()) for x, y in zip(Xv.tk, Xv.entry.astype('int64'))], dtype=np.uint64); K = Xv.ratio.values.astype(float)
    dl = ((HSH * 2654435761 + i * 40503) % 1000) < 50; m = ~dl
    s = sim(Xv.entry.values[m], Xv.exit.values[m], Xv.ret.values[m], K[m], Xv.tk.values[m])
    return np.exp(s.groupby(s.index.normalize()).last().reindex(IDX, method='ffill').fillna(0))
def stat_lvl(eq):
    out = []
    for a_, b_ in P2:
        e = eq[(eq.index >= a_) & (eq.index < b_)]; e = e / e.iloc[0]; yr = (e.index[-1] - e.index[0]).days / 365.25
        c = e.iloc[-1] ** (1 / yr) - 1; dd = (e / e.cummax() - 1).min(); out += [c, dd, c / -dd]
    return out
A = W[W.sp].reset_index(drop=True)
BK = {'Nasdaq-100 ex-S&P (no overlap)': W[W.ndx & ~W.sp & ~W.etf], 'Non-S&P non-ETF (no overlap)': W[~W.sp & ~W.etf], 'Nasdaq-100 full (overlap ok)': W[W.ndx], 'Dow 30 (overlap ok)': W[W.dow]}
for k, v in BK.items():
    n1, n2 = (v.entry < '2022-01-01').sum(), (v.entry >= '2022-01-01').sum(); print(f'  book {k}: {v.tk.nunique()} tickers, {len(v)} trades ({n1}/{n2}), avg ret {v[v.entry<"2022-01-01"].ret.mean()*100:+.2f}% / {v[v.entry>="2022-01-01"].ret.mean()*100:+.2f}%', flush=True)
RUNS = range(1, 41); eqA = [deq(A, i) for i in RUNS]; sA = np.array([stat_lvl(e) for e in eqA])
def med(x): m = np.median(x, axis=0); return f'ret {m[0]:+.1%} / {m[3]:+.1%}  DD {m[1]:.0%} / {m[4]:.0%}  ret/DD {m[2]:.2f} / {m[5]:.2f}'
print('\n=== (2019-21 / 2022-26), median of 40 matched runs'); print(f'{"A: live S&P book alone":52s} {med(sA)}', flush=True)
for k, B in BK.items():
    B = B.reset_index(drop=True); eqB = [deq(B, i) for i in RUNS]; sB = np.array([stat_lvl(e) for e in eqB])
    ra = pd.concat([e.pct_change() for e in eqA[:1]]); rb = eqB[0].pct_change(); cor = [ra[(ra.index>=a_)&(ra.index<b_)].corr(rb[(rb.index>=a_)&(rb.index<b_)]) for a_, b_ in P2]
    ov = np.mean([((A.tk == t) & (A.entry < x) & (A.exit > e)).any() for t, e, x in zip(B.tk.values[:800], B.entry.values[:800], B.exit.values[:800])])
    print(f'\n-- B = {k}: daily-return corr with A {cor[0]:.2f} / {cor[1]:.2f}; share of B trades overlapping an A position in same ticker (first 800) {ov:.0%}')
    print(f'{"   B alone":52s} {med(sB)}', flush=True)
    for w in (.7, .5):
        sC = np.array([stat_lvl(w * eqA[j] + (1 - w) * eqB[j]) for j in range(40)])
        bt = ((sC[:, 2] > sA[:, 2]).mean(), (sC[:, 5] > sA[:, 5]).mean())
        print(f'{f"   A {w:.0%} + B {1-w:.0%} (capital split)":52s} {med(sC)}   ret/DD beats A alone {bt[0]:.0%} / {bt[1]:.0%}', flush=True)
