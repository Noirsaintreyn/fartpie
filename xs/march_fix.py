"""March-2026 fixes, tested on 2019-26. Live trades (ahx base+sticky), ratio >= 2.5, 6 slots.
 one-per-name : a ticker can't hold two slots at once (15m + 1h track on the same name)
 cooldown     : after a losing trade in a name, skip that name's entries for N trading days (or after K losses in a row)
 breaker      : no new entries while the book is more than X% below its 20-day high"""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.argv = ['x', '2']
exec(open('xs/ctx_sim.py').read().split('T = pd.concat')[0])
T = pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/ahx/*.parquet')); T = T[T['var'] == 'base+sticky']
T['entry'] = pd.to_datetime(T.entry); T['exit'] = pd.to_datetime(T.exit)
T = T[(T.entry >= '2019-01-01') & (T.entry <= T.tk.map(last) + pd.Timedelta(days=30))]
T['ratio'] = [RAT[t].asof(d.normalize()) for t, d in zip(T.tk, T.entry)]; T = T[T.ratio >= 2.5]
Z0 = T.sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable')
cols = [Z0[c].values for c in ('entry', 'exit', 'ret', 'tk')]
def sim(a, b, n=6, one=False, cd=0, k=1, brk=0):
    op = []; ev = []; lastloss = {}; streak = {}; pend = []            # pend: trades taken but not yet closed (for book state)
    real = 0.0; hist = []                                               # realized equity (log) + (time, equity) for the breaker
    for e, x, r, tk in zip(*cols):
        if e < a or e >= b: continue
        done = [p for p in pend if p[0] <= e]; pend = [p for p in pend if p[0] > e]
        for xt, rr, t in sorted(done, key=lambda z: z[0]):
            real += np.log1p(rr / n); hist.append((xt, real))
            if rr < 0: lastloss[t] = xt; streak[t] = streak.get(t, 0) + 1
            else: streak[t] = 0
        op = [o for o in op if o[0] > e]
        if len(op) >= n: continue
        if one and any(o[1] == tk for o in op): continue
        if cd and tk in lastloss and streak.get(tk, 0) >= k and np.busday_count(lastloss[tk].astype('datetime64[D]'), e.astype('datetime64[D]')) < cd: continue
        if brk and hist:
            w = [h for t_, h in hist if t_ >= e - np.timedelta64(28, 'D')]
            if w and real - max(w) < np.log1p(-brk): continue
        op.append((x, tk)); pend.append((x, r, tk)); ev.append((x, r / n))
    s = pd.Series([v for _, v in ev], index=pd.DatetimeIndex([t for t, _ in ev])).sort_index()
    eq = (1 + s).groupby(s.index.normalize()).prod().cumprod(); y = (eq.index[-1] - eq.index[0]).days / 365.25
    return eq.iloc[-1] ** (1 / y) - 1, (eq / eq.cummax() - 1).min()
P = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'), ('2026-02-15', '2026-04-15'))]
print(f'{"6 slots, ratio >= 2.5, sticky":<44}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}{"Feb15-Apr15":>16}')
def show(nm, **kw):
    print(f'  {nm:<42}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in (sim(a, b, **kw) for a, b in P)), flush=True)
show('live')
show('one slot per name', one=True)
for cd in (1, 2, 3, 5): show(f'cooldown {cd} day(s) after a loss', cd=cd)
for cd in (3, 5): show(f'cooldown {cd} days after 2 losses in a row', cd=cd, k=2)
show('one per name + cooldown 2d', one=True, cd=2)
for b in (.06, .08, .10): show(f'breaker: no entries if book < -{b:.0%} vs 20d high', brk=b)
