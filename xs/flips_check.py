"""Nov 4 2026: is 'flips that day' informative without look-ahead? DECLARED BEFORE RESULTS. x_crowd / x_flips_day count ALL flips of the entry day (including ones after the entry time).
Tests with only what was known at the entry: flips_sofar (same day, entry time <= own), flips_prev (previous session), flips_3d (previous 3 sessions), by period, month-clustered t."""
import numpy as np, pandas as pd
X = pd.read_parquet('/tmp/claude-0/ml/ctx_rank_X.parquet'); U = X[X.ratio >= 1.5].sort_values('entry').reset_index(drop=True); U['d'] = U.entry.dt.normalize(); U['big'] = (U.ret >= .15).astype(float)
print('crowd == flips that day?', (U.crowd == U.groupby('d').tk.transform('size')).mean())
U['so_far'] = U.groupby('d').cumcount(); cnt = U.groupby('d').size(); days = cnt.index
prev = pd.Series(cnt.shift(1).values, index=days); p3 = pd.Series(cnt.shift(1).rolling(3).sum().values, index=days)
U['prev'] = U.d.map(prev); U['p3'] = U.d.map(p3); U['mo'] = U.entry.dt.strftime('%Y-%m')
def reg(y, x, cl):
    ok = ~(np.isnan(x) | np.isnan(y)); x, y, cl = x[ok], y[ok], cl[ok]; xc = x - x.mean(); b = (xc * (y - y.mean())).sum() / (xc ** 2).sum(); e = (y - y.mean()) - b * xc
    s = pd.Series(xc * e).groupby(cl).sum(); return b, b / (np.sqrt((s ** 2).sum()) / (xc ** 2).sum())
for f in ('crowd', 'so_far', 'prev', 'p3'):
    rk = U[f].rank(pct=True) - .5; out = []
    for p, m in (('19-21', U.entry.dt.year <= 2021), ('22-26', U.entry.dt.year >= 2022)):
        b, t = reg(U.loc[m, 'ret'].values * 100, rk[m].values.astype(float), U.loc[m, 'mo'].values); b2, t2 = reg(U.loc[m, 'big'].values * 100, rk[m].values.astype(float), U.loc[m, 'mo'].values); out.append(f'{p}: avg trade {b:+.2f} (t {t:+.1f}), runner {b2:+.2f} (t {t2:+.1f})')
    print(f'{f:<8}', ' | '.join(out))
for f in ('so_far', 'prev'):
    for p, m in (('19-21', U.entry.dt.year <= 2021), ('22-26', U.entry.dt.year >= 2022)):
        z = U[m]; q = pd.qcut(z[f].rank(method='first'), 3, labels=['low', 'mid', 'high']); print(f' {f} {p}', {k: f'{g.ret.mean()*100:+.2f}% win {g.ret.gt(0).mean():.0%} runners {g.big.mean()*100:.1f}% n{len(g)} (flips {g[f].min():.0f}-{g[f].max():.0f})' for k, g in z.groupby(q)})
