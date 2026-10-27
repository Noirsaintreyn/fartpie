import numpy as np, pandas as pd, glob
exec(open('xs/book_mtm.py').read().split("if __name__")[0])
src = open('xs/stop_book.py').read(); exec(src[src.index('def sim('):src.index("B = S[S['var']")])
L = prep(glob.glob('xs/ext/tm/*.parquet')); L = L[L['var'] == 'vec']
F = pd.concat(pd.read_parquet(f) for f in glob.glob('xs/ext/fx/*.parquet')).drop(columns=['exit', 'ret'])
L = L.merge(F, on=['tk', 'h1', 'entry'], how='left'); print('live trades', len(L), 'with features', L.age1h.notna().sum())
L['half'] = np.where(L.d.dt.year >= 2022, '22-26', '19-21')
FE = {'age1h': '1h uptrend age at entry (1h bars; -1 = 1h not bull)', 'dage': 'daily uptrend age (days bull)', 'mkt20': 'market (SPY) 20d trend',
      'sec20': 'sector 20d trend net of market', 'res20': 'stock-specific 20d trend (net of market + sector)', 'stk20': 'raw stock 20d trend',
      'cp5': 'changepoints in the last ~5 sessions', 'dse': 'trading days since last earnings', 'rse': 'return since the close before last earnings'}
P_ = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
def show(nm, Z): print(f'    {nm:<58}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in (sim(Z, a, b) for a, b in P_)), flush=True)
print('\nfast/slow: daily vector state at entry (avg trade %, n)'); print((L.groupby(['dstate', 'h1', 'half']).ret.agg(['mean', 'size']).unstack().assign(**{}).round(4) * [100, 100, 1, 1]).to_string())
print(f'\n{"":<62}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}'); show('live', L)
show('skip entries when the daily vector is bear', L[~(L.dstate == -1)])
show('skip 15m entries when the daily vector is bear', L[~((L.dstate == -1) & ~L.h1)])
for k, lab in FE.items():
    Z = L[L[k].notna()]; tr = Z[Z.half == '19-21']
    if k == 'age1h': q = pd.cut(Z[k], [-2, -.5, 4, 12, 40, 1e9], labels=['1h not bull', '0-4', '5-12', '13-40', '40+'])
    elif k == 'cp5': q = pd.cut(Z[k], [-1, 0, 1, 99], labels=['0', '1', '2+'])
    else: q = pd.qcut(Z[k].rank(method='first'), 3, labels=['low', 'mid', 'high'])
    g = Z.groupby([q, 'half']).ret.agg(['mean', 'size']).unstack(); g[[('mean', '19-21'), ('mean', '22-26')]] *= 100
    print(f'\n{lab}:'); print(g.round(3).to_string())
    by = tr.groupby(q.loc[tr.index]).ret.mean(); worst = by.idxmin(); best = by.idxmax()
    show(f'skip "{worst}" (worst bucket in 2019-21)', L.drop(Z.index[q == worst]))
    z = L.copy(); rk = z[k].rank(pct=True).fillna(.5); sign = 1 if by.get(best, 0) == by.iloc[-1] else -1
    z['ratio'] = z.ratio + (rk if sign == 1 else 1 - rk) * .5; show(f'tie-break toward the better side ({"high" if sign == 1 else "low"} {k})', z)
