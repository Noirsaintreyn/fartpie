"""Nov 2 2026: no-look-ahead version of noise_crowd. crowd_k = other universe flips with entry at or before this entry's bar, same day
(known at entry); also prior session's total flips. Scaled by universe size; percentile vs the trailing 250 trading days of the same
statistic (no full-year ranking). Book test incl. neighbouring cut-offs."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/fartpie/xs/stress_budget.py').read().split("rng = np.random.default_rng(43)")[0].replace("print(", "(lambda *a, **k: None)("))
X = pd.read_parquet('/tmp/claude-0/ml/noise_X.parquet').sort_values('entry').reset_index(drop=True)
X['k_now'] = X.groupby('d').entry.rank(method='max') - 1                                         # flips at or before this bar, excl. self
nu = X.groupby(X.entry.dt.to_period('Q')).tk.transform('nunique').values
day = X.groupby('d').size(); prev = day.shift(1).reindex(pd.bdate_range(day.index.min(), day.index.max())).ffill()
X['k_prev'] = prev.reindex(X.d).fillna(0).values
for c in ('k_now', 'k_prev'): X[c + '_s'] = X[c] / nu
X['k_both_s'] = (X.k_now + X.k_prev) / nu
def trail_pct(col):                                                                             # percentile vs trailing 250 days of daily values
    s = X.groupby('d')[col].min()  if col != 'k_now_s' else None
    out = np.full(len(X), np.nan); v = X[col].values; dd = X.d.values; ref = X[['d', col]].copy()
    dmax = X.groupby('d')[col].max(); days = dmax.index.values
    for i in range(len(X)):
        lo = dd[i] - np.timedelta64(365, 'D'); w = dmax[(days >= lo) & (days < dd[i])].values
        if len(w) >= 60: out[i] = (w < v[i]).mean()
    return out
for c in ('k_now_s', 'k_prev_s', 'k_both_s'): X['p_' + c] = trail_pct(c)
top = X.groupby('h').ret.transform(lambda s: s >= s.quantile(.95))
for c in ('p_k_now_s', 'p_k_prev_s', 'p_k_both_s'):
    print(f'{c}: avg trade / big-winner ratio by bucket (2019-21 | 2022-26)')
    q = X[c]
    for lo, hi in ((-1, .1), (.1, .2), (.2, .4), (.4, .7), (.7, 1.01)):
        m = (q > lo) & (q <= hi); row = []
        for h in ('2019-21', '2022-26'):
            z = (X.h == h) & q.notna(); mm = m & z; row.append(f'{X.ret[mm].mean():+.2%} jump {(top[mm].sum() / top[z].sum()) / (mm.sum() / z.sum()):.2f} n{mm.sum()}')
        print(f'  {max(lo,0):.0%}-{min(hi,1):.0%}:  ' + '   '.join(row))
rng = np.random.default_rng(97); DR = [rng.random(len(X)) < .05 if i else np.zeros(len(X), bool) for i in range(41)]
def ev(Z): return np.array([[v_ for cd in stat(dret(simw(Z[~dl[Z.index]])).cumsum()) for v_ in (cd[0], cd[1], cd[0] / -cd[1])] for dl in [pd.Series(d_, X.index) for d_ in DR]])
def line(nm, a, b=None):
    m = np.median(a[1:], axis=0); s = f'  {nm:<34} ret {m[0]:+.1%} / {m[3]:+.1%}   DD {m[1]:.0%} / {m[4]:.0%}   ret/DD {m[2]:.2f} / {m[5]:.2f}'
    if b is not None: s += f'   beats live {(a[1:, 2] > b[1:, 2]).mean():.0%} / {(a[1:, 5] > b[1:, 5]).mean():.0%}  (return {(a[1:, 0] > b[1:, 0]).mean():.0%} / {(a[1:, 3] > b[1:, 3]).mean():.0%})'
    print(s, flush=True)
print('\nbook test (fixed budget, 6 slots, 40 matched runs)'); B0 = ev(X); line('live', B0)
for c in ('p_k_now_s', 'p_k_prev_s', 'p_k_both_s'):
    for p in (.1, .2, .3): line(f'drop {c} <= {p:.0%}', ev(X[~(X[c] <= p)]), B0)
for k in (0, 1, 2): line(f'drop if <= {k} flips so far today', ev(X[~(X.k_now <= k)]), B0)
X.to_parquet('/tmp/claude-0/ml/noise_X2.parquet')
