exec(open('/tmp/claude-0/ml/rangez.py').read().split("\nP = [(np.datetime64")[0].replace("print(f'trades", "#").replace("print(f'\\n{c} quintiles", "#").replace("print(g.round(3)", "#"))
P = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
def show(nm, Z): print(f'  {nm:<66}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in (sim(Z, a, b) for a, b in P)), flush=True)
st = T[(~T.h1) & (T.used > 1.07)][['tk', 'entry']].copy(); st['day'] = st.entry.dt.normalize()
T['day'] = T.entry.dt.normalize()
def blocked(days, tracks):
    m = np.zeros(len(T), bool)
    S = st.rename(columns={'entry': 'se'})
    J = T.reset_index().merge(S, on='tk')
    ok = (J.entry > J.se) & (J.entry < J.se.dt.normalize() + pd.Timedelta(days=days)) & (J.h1 if tracks == '1h' else True)
    m[J.loc[ok, 'index'].unique()] = True; return m
print(f'{"stretched 15m flip skipped, and the same name is blocked...":<68}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
show('live', T)
base = T[~((~T.h1) & (T.used > 1.07))]; show('skip stretched 15m only (no block)', base)
for days, tr, lab in ((1, '1h', 'from 1h re-entry the rest of that day'), (1, 'both', 'from any re-entry the rest of that day'), (3, 'both', 'from any re-entry for 3 days')):
    b = blocked(days, tr); Z = T[~(((~T.h1) & (T.used > 1.07)) | b)]; show(f'skip + block {lab} ({b.sum()} blocked)', Z)
