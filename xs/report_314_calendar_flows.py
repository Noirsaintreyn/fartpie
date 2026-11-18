"""Nov 18 2026 report 314. Mechanical calendar flows named in the bank papers (BofA: month-end/monthly expiry gamma, vol-control and CTA rebalancing; JPM/GS weeks keyed to PCE, ISM, NFP). Flags derivable WITHOUT a data feed: NFP (first Friday), ISM/first business day, month-end (last business day), quarter-end, monthly opex (third Friday), triple witching (opex in Mar/Jun/Sep/Dec), day before holiday-less Monday. Hypothesis (declared): event/flow days have larger range/ADR (size) and the algo (trend-day engine) earns more on them; month-end/quarter-end rebalancing flow is directional-agnostic so adds range not efficiency. Why it would stop: scheduled releases are priced, 0DTE; FOMC/CPI/PPI dates are not rule-derivable so not tested here. Train 2019-21 | test 2018+2022+, pts/day for ALL and day range/ADR."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; D = pd.read_parquet(SP + 'd312.parquet'); D['rn'] = D.rng / D.adr; i = D.index; ym = i.to_period('M')
D['nfp'] = (i.dayofweek == 4) & (i.day <= 7)
D['opex'] = (i.dayofweek == 4) & (i.day >= 15) & (i.day <= 21); D['tw'] = D.opex & i.month.isin([3, 6, 9, 12])
first = pd.Series(i, index=i).groupby(ym).transform('min'); last = pd.Series(i, index=i).groupby(ym).transform('max')
D['first'] = (pd.Series(i, index=i) == first).values; D['mend'] = (pd.Series(i, index=i) == last).values; D['qend'] = D.mend & i.month.isin([3, 6, 9, 12])
D['mend3'] = pd.Series(i, index=i).groupby(ym).rank(ascending=False).values <= 3
print(f'days {len(D)} through {i.max().date()}; base ALL pts/day train {D[D.tr].ALL.mean():+.1f} test {D[~D.tr].ALL.mean():+.1f}; base range/ADR {D[D.tr].rn.mean():.2f} {D[~D.tr].rn.mean():.2f}')
print('flag            n(tr|te)    range/ADR tr|te     ALL pts/day tr|te      M1 tr|te    V15a tr|te    ON+ tr|te   eff tr|te')
for f in ('nfp', 'first', 'mend', 'mend3', 'qend', 'opex', 'tw'):
    a, b = D[D.tr & D[f]], D[~D.tr & D[f]]
    print(f'{f:6s} {len(a):4d}|{len(b):4d}   {a.rn.mean():5.2f}|{b.rn.mean():5.2f}    {a.ALL.mean():+7.1f}|{b.ALL.mean():+7.1f}   {a.M1.mean():+6.1f}|{b.M1.mean():+6.1f}  {a.V15a.mean():+6.1f}|{b.V15a.mean():+6.1f}  {a["ON+"].mean():+6.1f}|{b["ON+"].mean():+6.1f}  {a.effday.mean():.2f}|{b.effday.mean():.2f}')
print('\nby year, NFP ALL pts/day and month-end ALL pts/day:'); print(pd.DataFrame({'nfp': D[D.nfp].groupby(D[D.nfp].index.year).ALL.mean().round(0), 'mend': D[D.mend].groupby(D[D.mend].index.year).ALL.mean().round(0), 'opex': D[D.opex].groupby(D[D.opex].index.year).ALL.mean().round(0)}).T.to_string())
