exec(open('/tmp/claude-0/ml/rangez.py').read().split("\nP = [(np.datetime64")[0].replace("print(f'trades", "#").replace("print(f'\\n{c} quintiles", "#").replace("print(g.round(3)", "#"))
P = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
def show(nm, Z): print(f'  {nm:<62}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in (sim(Z, a, b) for a, b in P)), flush=True)
for trk, m in (('15m', ~T.h1), ('1h', T.h1)):
    x = T[m]; s = x[(x.used > 1.07)]; print(f'{trk}: stretched entries (used > 107%) avg {s.ret.mean():+.3%} ({len(s)}), others {x[~(x.used > 1.07)].ret.mean():+.3%}; stretched AND z > 0.5: {x[(x.used > 1.07) & (x.z > .5)].ret.mean():+.3%} ({((x.used > 1.07) & (x.z > .5)).sum()})')
print(f'\n{"rule":<64}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}'); show('live', T)
show('skip 1h-track entries that used > 107% of the range', T[~(T.h1 & (T.used > 1.07))])
show('skip 1h-track entries > 0.5 ranges above the open', T[~(T.h1 & (T.z > .5))])
show('skip 1h-track entries stretched AND above the open', T[~(T.h1 & (T.used > 1.07) & (T.z > .5))])
show('skip any entry stretched AND > 0.75 ranges above the open', T[~((T.used > 1.07) & (T.z > .75))])
show('skip any entry with > 150% of the range used', T[~(T.used > 1.5)])
show('skip 15m-track entries that used > 107% of the range', T[~((~T.h1) & (T.used > 1.07))])
show('skip 15m-track entries stretched AND > 0.5 ranges above open', T[~((~T.h1) & (T.used > 1.07) & (T.z > .5))])
show('skip 15m-track entries with > 130% used', T[~((~T.h1) & (T.used > 1.3))])
for y in range(2019, 2027):
    x = T[(~T.h1) & (T.entry.dt.year == y)]; s = x.used > 1.07
    print(f'  {y}: 15m stretched {x[s].ret.mean():+.2%} ({s.sum()})  vs rest {x[~s].ret.mean():+.2%}')
