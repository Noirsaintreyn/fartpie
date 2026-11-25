"""Report 484 (Nov 25 2026): Rey - 'from the 5 features, is the combination better than one of them alone?' Daily flips (structure_small), px<20/<10/<5, train 2019-21 ECDF + cut (top 30% of the bucket's train rows), test 2022+.
DECLARED BEFORE RESULTS. Groups: each single feature (ATRP, F11, VOLX, LTURN, -LMC), the 5-feature mean-rank score (base5), and drop-one-out versions of base5. Read REACH (peak>=+10%), mean r, median, win, era split (2022 / 2023-26), n. Pass for 'combination beats the best single feature' = base5 mean AND REACH >= the best single in BOTH eras, and month-clustered paired t of (base5 - best single) group-month means >= 2."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; O = pd.read_pickle(SP + 'ext/structure_small.pkl'); O['d'] = pd.to_datetime(O.d); O['tr'] = O.d.dt.year <= 2021
B = {'ATRP': 1, 'F11': 1, 'VOLX': 1, 'LTURN': 1, 'LMC': -1}
def ranks(D, fs):
    R = pd.DataFrame(index=D.index)
    for f in fs:
        x = D[f].astype(float); ref = np.sort(x[D.tr & x.notna()].values); q = np.searchsorted(ref, x.values) / len(ref); R[f] = q if B[f] > 0 else 1 - q; R.loc[x.isna(), f] = np.nan
    return R.mean(axis=1)
st = lambda x: f'n {len(x):4d} REACH {(x.PEAK >= .1).mean():.0%} win {(x.r > 0).mean():.0%} mean {x.r.mean():+6.1%} med {x.r.median():+.1%}'
era = lambda x: f"2022 {x[x.d.dt.year == 2022].r.mean():+.1%} | 2023-26 {x[x.d.dt.year >= 2023].r.mean():+.1%}"
for cap in (20, 10, 5):
    S = O[(O.px < cap)].dropna(subset=['r', 'PEAK']).copy(); S['mo'] = S.d.dt.to_period('M').astype(str); T = S[~S.tr]; sets = {f'only {f}': [f] for f in B}; sets['BASE5 (all 5)'] = list(B); sets.update({f'base5 minus {f}': [g for g in B if g != f] for f in B}); res = {}
    print(f'\n######## sub-${cap}: all test flips {st(T)} | {era(T)}')
    for k, fs in sets.items():
        s = ranks(S, fs); cut = np.quantile(s[S.tr].dropna(), .7); G = S[(~S.tr) & (s >= cut)]; res[k] = G; print(f'  {k:18s} {st(G)} | {era(G)}')
    best = max((k for k in sets if k.startswith('only')), key=lambda k: res[k].r.mean()); a = res['BASE5 (all 5)'].groupby('mo').r.mean(); b = res[best].groupby('mo').r.mean(); j = pd.concat([a, b], axis=1, keys=['a', 'b']).dropna(); d = j.a - j.b
    print(f'  best single = {best}; base5 - {best} by month: {d.mean():+.2%} t {d.mean() / (d.std() / np.sqrt(len(d))):+.2f} ({len(d)} months)')
