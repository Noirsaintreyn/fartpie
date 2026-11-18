"""Nov 18 2026 report 320. Base NQ stack vs CTA-flow tilt, in NQ points per contract (Rey's unit). Base = sum of the 8 models, 1 contract each. Tilt variants, all declared a priori from report 312-313 (NOT out-of-sample: the sign split was found in this data): T1 all legs x1.5 after CTA-proxy NQ position increase, x0.5 after decrease; T2 same but only ON-, M1, V15a tilted; T3 T2 with x0 after decrease. Data through 2026-08-14 (d312). pts/year, max drawdown (pts, cumulative daily), worst day, mean/std."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; D = pd.read_parquet(SP + 'd312.parquet'); s = np.sign(D.NQ_dpos.round(3)); K = ['RA', 'M1', 'R1', 'ON+', 'ON-', 'RULES', 'V15a', 'S1']
def tilt(cols, up, dn): w = np.where(s > 0, up, np.where(s < 0, dn, 1.0)); return sum(D[k] * (w if k in cols else 1.0) for k in K)
V = {'BASE (8 models)': D.ALL, 'T1 all legs 1.5/0.5': tilt(K, 1.5, .5), 'T2 ON-,M1,V15a 1.5/0.5': tilt(['ON-', 'M1', 'V15a'], 1.5, .5), 'T3 ON-,M1,V15a 1.5/0': tilt(['ON-', 'M1', 'V15a'], 1.5, 0.0)}
def dd(x): c = x.cumsum(); return (c - c.cummax()).min()
print(f'days {len(D)} {D.index.min().date()} to {D.index.max().date()}\n'); print(f'{"":26s} {"pts/yr tr":>10s} {"pts/yr te":>10s} {"mean/day":>9s} {"std":>6s} {"mean/std":>8s} {"maxDD pts":>9s} {"worst day":>9s}')
for n, x in V.items():
    tr, te = x[D.tr], x[~D.tr]; print(f'{n:26s} {tr.mean() * 252:10.0f} {te.mean() * 252:10.0f} {x.mean():9.1f} {x.std():6.0f} {x.mean() / x.std():8.3f} {dd(x):9.0f} {x.min():9.0f}')
print('\npts per year:'); print(pd.DataFrame({n: x.groupby(D.index.year).sum().round(0) for n, x in V.items()}).to_string())
print('\nmax drawdown pts by era: ', {n: (round(dd(x[D.tr])), round(dd(x[~D.tr]))) for n, x in V.items()})
