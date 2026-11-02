"""Nov 2 2026: follow-up to noise_audit: 'lone flips' (few other universe names flipping the same day) had fewer big winners in BOTH
periods (ratio 0.64 / 0.64). Re-rank crowd within each year (universe size changes), then book-test dropping the loneliest 10/15/20/30%."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/fartpie/xs/stress_budget.py').read().split("rng = np.random.default_rng(43)")[0].replace("print(", "(lambda *a, **k: None)("))
X = pd.read_parquet('/tmp/claude-0/ml/noise_X.parquet')
n_univ = X.groupby(X.entry.dt.to_period('Q')).tk.transform('nunique'); X['crowd_s'] = X.crowd / n_univ
X['cq'] = X.groupby(X.entry.dt.year).crowd_s.rank(pct=True)
top = X.groupby('h').ret.transform(lambda s: s >= s.quantile(.95))
print('crowd (share of universe flipping same day), ranked within year')
for lo, hi in ((0, .1), (.1, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)):
    m = (X.cq > lo) & (X.cq <= hi); row = []
    for h in ('2019-21', '2022-26'):
        z = X.h == h; mm = m & z; row.append(f'{X.ret[mm].mean():+.2%} jump {(top[mm].sum() / top[z].sum()) / (mm.sum() / z.sum()):.2f}')
    print(f'  {lo:.0%}-{min(hi,1):.0%}:  ' + '   '.join(row))
rng = np.random.default_rng(97); DR = [rng.random(len(X)) < .05 if i else np.zeros(len(X), bool) for i in range(41)]
def ev(Z): return np.array([[v_ for cd in stat(dret(simw(Z[~dl[Z.index]])).cumsum()) for v_ in (cd[0], cd[1], cd[0] / -cd[1])] for dl in [pd.Series(d_, X.index) for d_ in DR]])
def line(nm, a, b=None):
    m = np.median(a[1:], axis=0); s = f'  {nm:<30} ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
    if b is not None: s += f'   beats live {(a[1:, 2] > b[1:, 2]).mean():.0%} / {(a[1:, 5] > b[1:, 5]).mean():.0%}  (return {(a[1:, 0] > b[1:, 0]).mean():.0%} / {(a[1:, 3] > b[1:, 3]).mean():.0%})'
    print(s, flush=True)
print('\nbook test (fixed budget, 6 slots, 40 matched runs)'); B0 = ev(X); line('live', B0)
for p in (.1, .15, .2, .3): line(f'drop loneliest {p:.0%}', ev(X[~(X.cq <= p)]), B0)
for k in (0, 1, 2): line(f'drop if <= {k} other flips that day', ev(X[~(X.crowd <= k)]), B0)
