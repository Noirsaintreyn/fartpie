"""Nov 18 2026 report 307. THE EQUATION for the NQ stack (explanatory decomposition; contemporaneous labels are fine for explanation, ex-ante columns are marked).
Identity: algo pts/day = capture x |net move of the day| ; |net| = range x efficiency.  So P&L = capture x range x efficiency.
Predictable ex-ante (from 253-270, 305): RANGE (HAR+IV R2 .39-.56; early volume rho .3-.4). Not predictable: EFFICIENCY (AUC <= .57).  Lever candidates therefore: (1) range -> size, (2) capture -> exits/entries, (3) frequency of trend days -> nothing we can do.
Here: (a) pts/day by (range tercile x efficiency tercile) per model, train 2019-21 | test; (b) capture = pts / |net| on TREND days by model and year; (c) ex-ante: pts/day by yesterday-range-vs-ADR tercile (cuts on train) per model; (d) share of annual P&L variance explained by trend-day count x mean range."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
D = pd.read_parquet(SP + 'd305.parquet'); M = ['RA', 'M1', 'R1', 'ON+', 'ON-', 'RULES', 'V15a', 'S1', 'ALL']
D['absnet'] = D.net.abs(); D['rn'] = D.rng / D.adr; q = lambda s, m: s[m].quantile([1 / 3, 2 / 3]).values
tr = D.tr
qe = q(D.effday, tr); qr = q(D.rn, tr); D['eT'] = np.where(D.effday <= qe[0], 'MR', np.where(D.effday >= qe[1], 'TREND', 'MID')); D['rT'] = np.where(D.rn <= qr[0], 'small', np.where(D.rn >= qr[1], 'big', 'mid'))
print(f'days {len(D)}; day-type cuts (train): efficiency {np.round(qe, 2)}, range/ADR {np.round(qr, 2)}')
print('\n(a) pts/day by day efficiency (MR | MID | TREND), train | test, for each model; and share of days')
print('   share of days train', D[tr].eT.value_counts(normalize=True).round(2).to_dict(), '| test', D[~tr].eT.value_counts(normalize=True).round(2).to_dict())
for k in M:
    f = lambda w: ' / '.join(f'{w[w.eT == t][k].mean():+6.1f}' for t in ('MR', 'MID', 'TREND')); print(f'  {k:6s} train {f(D[tr])}  | test {f(D[~tr])}')
print('\n    TREND days only, pts/day by range (small | mid | big)')
for k in M:
    g = lambda w: ' / '.join(f'{w[(w.eT == "TREND") & (w.rT == t)][k].mean():+6.1f}' for t in ('small', 'mid', 'big')); print(f'  {k:6s} train {g(D[tr])}  | test {g(D[~tr])}')
print('\n(b) capture = sum(pts) / sum(|net|) on TREND days, by year (ALL models, 1 contract each; |net| in pts)')
T = D[D.eT == 'TREND']; cap = (T.groupby(T.index.year).ALL.sum() / T.groupby(T.index.year).absnet.sum()).round(3); print('  ', cap.to_dict()); print('   TREND-day count by year:', T.groupby(T.index.year).size().to_dict(), '| mean |net| bps:', (T.absnet / T.o * 1e4).groupby(T.index.year).mean().round(0).to_dict())
for k in ('M1', 'V15a', 'RA', 'RULES', 'S1', 'ON+'): print(f'   capture {k:5s}', (T.groupby(T.index.year)[k].sum() / T.groupby(T.index.year).absnet.sum()).round(3).to_dict())
print('\n(c) EX-ANTE: yesterday range / ADR14 (known at the open) tercile -> pts/day (cuts train), low / mid / high')
D['yr'] = D.rn.shift(1); qy = q(D.yr, tr)
for k in M:
    f = lambda w: ' / '.join(f'{w[w.yr <= qy[0]][k].mean():+6.1f}' if i == 0 else (f'{w[w.yr >= qy[1]][k].mean():+6.1f}' if i == 2 else f'{w[(w.yr > qy[0]) & (w.yr < qy[1])][k].mean():+6.1f}') for i in range(3)); print(f'  {k:6s} train {f(D[tr])}  | test {f(D[~tr])}')
print('\n(d) annual P&L (ALL pts) vs trend-day count x mean TREND |net|  (year-level, n years):')
Y = pd.DataFrame({'pnl': D.groupby(D.index.year).ALL.sum(), 'trend_pts': T.groupby(T.index.year).absnet.sum(), 'ndays': D.groupby(D.index.year).size()}); Y['r2'] = np.nan; print(Y.round(0).to_string()); print('   corr(pnl, trend_pts) =', round(Y.pnl.corr(Y.trend_pts), 2), ' corr in bps terms:', round((Y.pnl / Y.ndays).corr(Y.trend_pts / Y.ndays), 2))
mo = D.groupby(D.index.to_period('M')).agg(pnl=('ALL', 'sum'), tp=('absnet', lambda s: s[D.loc[s.index, 'eT'] == 'TREND'].sum())); print('   monthly corr(pnl, trend-day |net| sum) =', round(mo.pnl.corr(mo.tp), 2), '; R2', round(mo.pnl.corr(mo.tp) ** 2, 2), f'(n={len(mo)})')
