"""Nov 1 2026: how to hold the extended-market shorts, and separate baskets by timeframe.
 A shorts (SPY z50 > 1.5 or z20 > 2) sharing the 6 long slots vs their own basket of k slots at 1/6 each (gross up to 1 + k/6)
 B separate pools for the 15m and 1h tracks (a + b = 6, each slot 1/6) vs one shared pool. Return, DD, return/DD; then 40 matched runs."""
exec(open('/home/claude/fartpie/xs/zone_z.py').read().split("bins = [")[0])
from collections import deque
def simw(Z, n=6, size=1 / 6, brk=.08):
    Z = Z.sort_values(['entry', 'ratio', 'h1', 'tk'], ascending=[True, False, True, True], kind='stable')
    op = []; pend = []; real = 0.0; hist = []; win = deque(); lvl0 = 0.0
    for e, x, r_ in zip(Z.entry.values, Z.exit.values, Z.ret.values):
        done = sorted([p for p in pend if p[0] <= e], key=lambda z: z[0]); pend = [p for p in pend if p[0] > e]
        for xt, rr in done: real += np.log1p(rr * size); hist.append((xt, real)); win.append((xt, real))
        op = [o for o in op if o > e]
        if len(op) >= n: continue
        if brk and hist:
            t0 = e - np.timedelta64(28, 'D')
            while win and win[0][0] < t0: lvl0 = win.popleft()[1]
            if real - max([lvl0] + [h for _, h in win]) < np.log1p(-brk): continue
        op.append(x); pend.append((x, r_))
    for xt, rr in sorted(pend, key=lambda z: z[0]): real += np.log1p(rr * size); hist.append((xt, real))
    s = pd.Series([h for _, h in hist], index=pd.DatetimeIndex([t for t, _ in hist])).sort_index(); return s.groupby(s.index.normalize()).last()
idx = pd.bdate_range('2019-01-01', Ls.entry.max())
def dret(eq): return eq.reindex(idx, method='ffill').fillna(0).diff().fillna(0)
def show(nm, r): print(f'  {nm:<46}{fmt(stat(r.cumsum()))}', flush=True)
L0 = dret(simw(Ls)); show('long book (live)', L0)
for gnm, g in (('z50 > 1.5', Sh.mz50 > 1.5), ('z20 > 2', Sh.mz20 > 2)):
    s = Sh[g.fillna(False)]
    show(f'shorts [{gnm}] SHARE the 6 long slots', dret(simw(pd.concat([Ls, s]))))
    for k in (1, 2, 3, 6): show(f'shorts [{gnm}] own basket, {k} slot(s) @1/6', L0 + dret(simw(s, n=k)))
print('\nB) separate pools for the 15m and 1h tracks (each slot 1/6)')
for a in (5, 4, 3, 2):
    show(f'{a} slots for 15m + {6 - a} for 1h', dret(simw(Ls[~Ls.h1], n=a)) + dret(simw(Ls[Ls.h1], n=6 - a)))
