"""Gold (GLD) and oil (CL futures) vector trades in the 6-slot rotation, optionally only when VIX (prior close) is elevated.
Same live model on each (premarket+RTH 15m vector, 15m + 1h tracks, sticky, LOCK5/0.5). Stocks keep priority on ties unless noted."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.argv = ['x', '2']
exec(open('xs/ctx_sim.py').read().split('T = pd.concat')[0])
T = pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/ahx/*.parquet')); T = T[T['var'] == 'base+sticky']
T['entry'] = pd.to_datetime(T.entry); T['exit'] = pd.to_datetime(T.exit)
T = T[(T.entry >= '2019-01-01') & (T.entry <= T.tk.map(last) + pd.Timedelta(days=30))]
T['ratio'] = [RAT[t].asof(d.normalize()) for t, d in zip(T.tk, T.entry)]; T = T[T.ratio >= 2.5]
M = pd.concat(pd.read_parquet(f'xs/ext/ahx_macro/{t}.parquet') for t in ('GLD', 'CL')); M = M[M['var'] == 'base+sticky']
M['entry'] = pd.to_datetime(M.entry); M['exit'] = pd.to_datetime(M.exit); M = M[M.entry >= '2019-01-01']; M['ratio'] = 0.0
vix = pd.read_parquet('etf_daily/VIX.parquet').close; vix.index = pd.to_datetime(vix.index).normalize()
M['vix'] = vix.shift(1).reindex(M.entry.dt.normalize(), method='ffill').values
def sim(Z, n=6):
    Z = Z.sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable'); op, ev = [], []
    for e, x, r in zip(Z.entry.values, Z.exit.values, Z.ret.values):
        op = [o for o in op if o > e]
        if len(op) < n: op.append(x); ev.append((x, r / n))
    return pd.Series([v for _, v in ev], index=pd.DatetimeIndex([t for t, _ in ev])).sort_index()
def st(s):
    e = (1 + s).groupby(s.index.normalize()).prod().cumprod(); y = (e.index[-1] - e.index[0]).days / 365.25
    return e.iloc[-1] ** (1 / y) - 1, (e / e.cummax() - 1).min()
P = (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))
print(f'{"6 slots, ratio >= 2.5, sticky":<44}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
def show(nm, Z):
    res = [st(sim(Z[(Z.entry >= a) & (Z.entry < b)])) for a, b in P]
    print(f'  {nm:<42}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in res), flush=True)
show('live (stocks only)', T)
for who in (('GLD',), ('CL',), ('GLD', 'CL')):
    nm = ' + '.join('gold' if w == 'GLD' else 'oil' for w in who); X = M[M.tk.isin(who)]
    show(f'+ {nm}, always', pd.concat([T, X]))
    for v in (20, 25, 30): show(f'+ {nm}, only when VIX > {v}', pd.concat([T, X[X.vix > v]]))
    X2 = X[X.vix > 25].assign(ratio=99.0); show(f'+ {nm}, VIX > 25 and first in line', pd.concat([T, X2]))
