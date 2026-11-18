"""Nov 18 2026 report 290: WHY did the Rule A calm-ATRP edge (low ATRP +25.7 test) stop in 2025-26? Rule A uses FIXED 75-pt target / 50-pt stop; price and ADR in points drifted. Look by quarter at: RA pts/trade by ATRP tercile, win rate, ADR(20) in pts, target/ADR, ATRP level."""
exec(open('/home/claude/fartpie/xs/report_288_model_environments.py').read().split("rows = []")[0])
T = L['RA']; T = T[T.sz > 0].copy(); T['d'] = pd.to_datetime(T.day); T = T[T.d <= '2026-10-31'].set_index('d')
T['atrp'] = F.ATRP.reindex(T.index).values; T['tc'] = TC['ATRP'].reindex(T.index).values; T['adr'] = (rng.rolling(20).mean().shift(1)).reindex(T.index).values; T['win'] = T.pts > 0
T['q'] = T.index.to_period('Q')
g = T.groupby('q').agg(n=('pts', 'size'), mean=('pts', 'mean'), win=('win', 'mean'), adr=('adr', 'mean'), atrp=('atrp', 'mean'))
g['tgt/adr'] = 75 / g.adr
for k, nm in ((0, 'lowATRP'), (1, 'midATRP'), (2, 'hiATRP')): g[nm] = T[T.tc == k].groupby('q').pts.mean()
print(g.loc['2023Q1':].round(2).to_string())
print('\nTercile cuts (train 2019-21):', CUT['ATRP'].round(2)); 
print('share of RA trades in each ATRP tercile by year:'); print(pd.crosstab(T.index.year, T.tc, normalize='index').round(2).to_string())
print('\nRA pts/trade by year x tercile:'); print(T.pivot_table(index=T.index.year, columns='tc', values='pts', aggfunc=['mean', 'size']).round(1).to_string())
# monthly 2025-26 low-ATRP trades
m = T[(T.tc == 0) & (T.index >= '2024-07-01')]; print('\nlow-ATRP RA trades since 2024-07 by month:'); print(m.groupby(m.index.to_period('M')).pts.agg(['size', 'sum']).T.to_string())
print('\nrolling ADR pts: 2022', rng['2022'].mean().round(0), '2023', rng['2023'].mean().round(0), '2024', rng['2024'].mean().round(0), '2025', rng['2025'].mean().round(0), '2026', rng['2026'].mean().round(0))
