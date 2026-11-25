"""Report 481 (Nov 25 2026): Rey - 'add catalysts and cut if it's bad'. DAILY flips (structure_small, px<20/<10/<5), base5 top-30% group (train-ECDF, test 2022+), return = flip's vector-exit r (30bp).
DECLARED BEFORE RESULTS. Catalysts known before the entry (8-K filed in the prior 7 days; earnings reaction in the prior 5 trading days):
 BADFIN  8-K 3.02 (unregistered equity sale = dilution) | BADEVT 8-K 3.01 (listing deficiency), 1.02 (agreement terminated), 1.03 (bankruptcy), 2.05/2.06 (restructuring / impairment), 4.02 (non-reliance on financials) | EARN_DN earnings (item 2.02) with the 2-day price reaction (close before -> close after) <= -5% | CUT = any of these.
 GOOD (for the other side of the cut): EARN_UP earnings reaction >= +5%, DEAL (1.01/2.01), INS (insider open-market buy 30d).
Mechanism: dilution, listing/bankruptcy/accounting problems and a negative earnings reaction are 'reasons to fall'; a flip into one is a bounce into a bad story. PASS for the cut = removing CUT flips raises the group's mean AND median AND p10 in test, with the same sign in 2022 and 2023-26, and removes < 25% of flips."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('xs/report_464_catalyst_flips.py').read(); exec(src[:src.index('def mp_(')])
O['tr'] = O.d.dt.year <= 2021; W = pd.read_pickle(SP + 'ext/wide_daily_ohlcv.pkl'); it_ = K['items'].astype(str).str.split(',')
K['BADFIN'] = it_.apply(lambda l: '3.02' in l); K['BADEVT'] = it_.apply(lambda l: any(x in l for x in ('3.01', '1.02', '1.03', '2.05', '2.06', '4.02')))
for c in ('BADFIN', 'BADEVT'): O[c] = flag(c)
# earnings reaction in the prior 5 trading days
E_ = K[K.EARN][['ticker', 'known']].drop_duplicates(); er = {}
for tk, x in E_.groupby('ticker'):
    df = W.get(tk)
    if df is None: continue
    df = df[~df.index.duplicated()]; ix = df.index
    for kd in x.known:
        j = ix.searchsorted(kd)
        if j < 1 or j + 1 >= len(ix): continue
        er.setdefault(tk, []).append((ix[min(j + 1, len(ix) - 1)], df.close.iloc[j + 1] / df.close.iloc[j - 1] - 1))
O['EARN_RX'] = np.nan
for i, (tk, d) in enumerate(zip(O.tk.values, O.d.values)):
    for dd, rx in er.get(tk, []):
        if np.datetime64(dd) <= d and np.datetime64(dd) > d - np.timedelta64(8, 'D'): O.iat[i, O.columns.get_loc('EARN_RX')] = rx
O['EARN_DN'] = O.EARN_RX <= -.05; O['EARN_UP'] = O.EARN_RX >= .05; O['CUT'] = O.BADFIN | O.BADEVT | O.EARN_DN
B = {'ATRP': 1, 'F11': 1, 'VOLX': 1, 'LTURN': 1, 'LMC': -1}
def score(D):
    R = pd.DataFrame(index=D.index)
    for f, s in B.items():
        x = D[f].astype(float); ref = np.sort(x[D.tr & x.notna()].values); q = np.searchsorted(ref, x.values) / len(ref); R[f] = q if s > 0 else 1 - q; R.loc[x.isna(), f] = np.nan
    return R.mean(axis=1)
st = lambda x: f'n {len(x):4d} REACH {(x.PEAK >= .1).mean():.0%} win {(x.r > 0).mean():.0%} mean {x.r.mean():+.1%} median {x.r.median():+.1%} p10 {x.r.quantile(.1):+.0%}' if len(x) else 'n 0'
for cap in (20, 10, 5):
    S = O[O.px < cap].copy(); S['s_'] = score(S); cut = np.quantile(S[S.tr].s_.dropna(), .7); G = S[(~S.tr) & (S.s_ >= cut)]; keep = G[~G.CUT]; bad = G[G.CUT]
    print(f'\n=== sub-${cap} top30% test 2022+: {len(G)} flips')
    print('  ALL          ', st(G)); print('  CUT removed  ', st(keep), f'| removes {len(bad) / len(G):.0%}'); print('  the CUT flips', st(bad))
    for nm in ('BADFIN', 'BADEVT', 'EARN_DN', 'EARN_UP', 'DEAL', 'INS'):
        x = G[G[nm] == True]; print(f'    {nm:8s}', st(x), f'| 2022 mean {x[x.d.dt.year == 2022].r.mean():+.1%} ({(x.d.dt.year == 2022).sum()}) / 2023-26 {x[x.d.dt.year >= 2023].r.mean():+.1%} ({(x.d.dt.year >= 2023).sum()})')
    for y, nm in ((G.d.dt.year == 2022, '2022'), (G.d.dt.year >= 2023, '2023-26')):
        a, b = G[y], G[y & ~G.CUT]; print(f'    {nm:7s} all mean {a.r.mean():+.1%} median {a.r.median():+.1%} -> cut {b.r.mean():+.1%} / {b.r.median():+.1%} (removed {1 - len(b) / len(a):.0%})')
