"""Nov 18 2026 report 308. 307 showed capture of trend-day movement collapsed in 2026 (ALL .20 vs .47 in 2025; M1 .04 vs .13; V15a .07 vs .15) while TREND-day count (56 of 155) and size (133 bp) are normal.  Question: WHEN in the day does the trend arrive, and did that shift?  Declared hypothesis: 2026 trend days are back-loaded (trend starts after the models' entry windows) or gap-driven.
Measures on TREND days (efficiency top third, train cut .16): share of |net| realised by 10:30 (same sign), by 12:00, hour of the day extreme, overnight gap / |net|.  By year."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
D = pd.read_parquet(SP + 'd305.parquet'); N = pd.read_parquet('/tmp/claude-0/ml/nq5_real_ext.parquet'); N = N[N.rth & (N.m >= 570) & (N.m <= 955)].copy(); N['day'] = pd.to_datetime(N.day).dt.normalize()
rows = {}
for d, x in N.groupby('day'):
    if d not in D.index: continue
    x = x.sort_values('m'); o = x.open.iloc[0]; e = x.close.iloc[-1]; net = e - o
    if abs(net) < 1: continue
    s = np.sign(net); f = lambda mm: (x[x.m <= mm].close.iloc[-1] - o) * s / abs(net) if (x.m <= mm).any() else np.nan
    ihi = x.high.idxmax(); ilo = x.low.idxmin(); ext = x.loc[ihi if s > 0 else ilo, 'm']  # time of the extreme in the trend direction
    rows[d] = dict(by1030=f(625), by1200=f(715), by1400=f(835), ext_h=ext / 60)
Z = pd.DataFrame(rows).T; D2 = D.join(Z, how='inner'); qe = D2.loc[D2.tr, 'effday'].quantile(2 / 3); T = D2[D2.effday >= qe]
print('TREND days by year: share of |net| realised by 10:30 | 12:00 | 14:00 (median), hour of the trend-direction extreme (median), n')
print(pd.DataFrame({'by1030': T.groupby(T.index.year).by1030.median(), 'by1200': T.groupby(T.index.year).by1200.median(), 'by1400': T.groupby(T.index.year).by1400.median(), 'ext_h': T.groupby(T.index.year).ext_h.median(), 'n': T.groupby(T.index.year).size()}).round(2).to_string())
print('\nALL-model pts/day on TREND days by year, and per model (1 ct each):')
print(T.groupby(T.index.year)[['ALL', 'M1', 'V15a', 'RA', 'ON+', 'ON-', 'RULES', 'S1', 'R1']].mean().round(1).to_string())
print('\n2026 by month: TREND days, MR days, ALL pts/day on trend, ALL pts/day on others')
m = D2[D2.index.year == 2026]; m['trend'] = m.effday >= qe; print(m.groupby(m.index.to_period('M')).apply(lambda w: pd.Series({'trendN': int(w.trend.sum()), 'days': len(w), 'trend_ALL': w[w.trend].ALL.mean(), 'other_ALL': w[~w.trend].ALL.mean(), 'trend_|net|bps': (w[w.trend].net.abs() / w[w.trend].o * 1e4).mean()})).round(1).to_string())
