"""Report 490 (Nov 25 2026): Rey - 'Mondays calm stocks beat wild ones, Fridays wild ones win'. Daily flips (structure_small), px<20/<10/<5. WILD = ATRP in the top 30% of the bucket's train rows, CALM = bottom 30% (train-frozen cuts); weekday = the flip-day close's weekday (entry next open), plus the LIVE group (4-feature top 30%) by weekday.
DECLARED BEFORE RESULTS. Hypothesis (day-of-week lottery/attention literature): the calm-minus-wild gap is positive on Mondays and negative on Fridays. Read mean r and REACH by weekday x {WILD, CALM}, era split (2019-21 | 2022 | 2023-26), month-clustered t of (WILD - CALM) per weekday. PASS = Monday (WILD-CALM) < 0 and Friday > 0 in BOTH train and test with |t| >= 2 in test. Also read the live group by flip weekday."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; O = pd.read_pickle(SP + 'ext/structure_small.pkl'); O['d'] = pd.to_datetime(O.d); O['tr'] = O.d.dt.year <= 2021; O['wd'] = O.d.dt.dayofweek; O['mo'] = O.d.dt.to_period('M').astype(str)
B = {'ATRP': 1, 'F11': 1, 'LTURN': 1, 'LMC': -1}; DN = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
def rank4(D):
    R = []
    for f, s in B.items():
        x = D[f].astype(float); ref = np.sort(x[D.tr & x.notna()].values); q = np.searchsorted(ref, x.values) / len(ref); q = np.where(x.isna(), np.nan, q); R.append(q if s > 0 else 1 - q)
    return np.nanmean(np.vstack(R), axis=0)
def tg(a, b):
    ma = a.groupby('mo').r.mean(); mb = b.groupby('mo').r.mean(); j = pd.concat([ma, mb], axis=1, keys=['a', 'b']).dropna(); d = j.a - j.b; return d.mean(), (d.mean() / (d.std() / np.sqrt(len(d))) if len(d) > 2 else np.nan)
for cap in (20, 10, 5):
    S = O[O.px < cap].dropna(subset=['r', 'PEAK', 'ATRP']).copy(); ref = np.sort(S[S.tr].ATRP.values); S['a'] = np.searchsorted(ref, S.ATRP.values) / len(ref); S['wild'] = S.a >= .7; S['calm'] = S.a <= .3
    print(f'\n######## sub-${cap}')
    for era, E in (('train 2019-21', S[S.tr]), ('test 2022+', S[~S.tr])):
        print(f' {era}: weekday  WILD mean/REACH (n) | CALM mean/REACH (n) | WILD-CALM (t)')
        for w in range(5):
            a = E[(E.wd == w) & E.wild]; b = E[(E.wd == w) & E.calm]; g, t = tg(a, b)
            print(f'   {DN[w]}  {a.r.mean():+6.1%}/{(a.PEAK >= .1).mean():.0%} ({len(a):4d}) | {b.r.mean():+6.1%}/{(b.PEAK >= .1).mean():.0%} ({len(b):4d}) | {g:+6.1%} (t {t:+.2f})')
    S['s4'] = rank4(S); cut = np.quantile(S[S.tr].s4.dropna(), .7); G = S[(~S.tr) & (S.s4 >= cut)]
    print(f' LIVE group (4-feature top 30%) test by flip weekday: ' + ' | '.join(f"{DN[w]} n {len(G[G.wd == w])} mean {G[G.wd == w].r.mean():+.1%} REACH {(G[G.wd == w].PEAK >= .1).mean():.0%}" for w in range(5)))
