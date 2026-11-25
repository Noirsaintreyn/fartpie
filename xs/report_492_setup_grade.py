"""Report 492 (Nov 25 2026): Rey - 'rank GRADE by all current context and how it performed: if ATRP is high and there is unusual buying that is likely grade A; in the backtest it hit 30 of 49 times and on this stock 10 of 15 ... and earnings moved early = low grade'.
DECLARED BEFORE RESULTS. Population: the LIVE group (4-feature base5 top 30% of the px<5 bucket, train-ECDF cut), daily flips 2019-Mar 2026; fit/cut on 2019-21 only, judge on 2022+. 'Hit' = the flip reaches +10% above the next-open entry before the vector exit (PEAK >= +10%).
Points (declared, no fitting): +1 ATRP in the top third of the group's TRAIN ATRP (frozen cut); +1 UNUSUAL insider buy (FIRST: an open-market buy in the prior 30d by someone with no purchase at this issuer in 24+ months); +1 STOCK RECORD (this ticker's earlier flips, any year, hit +10% in >= 60% with >= 5 prior flips); -1 EARLY30 (latest earnings 8-K in the prior 30d filed >= 5 days earlier than a year before); -2 NEGBAD (8-K 3.01/1.02/1.03/2.05/2.06/4.02 in the prior 7d).
Grade: A >= 2 points, B = 1, C = 0, D < 0 (low). Read hits/n (REACH), mean, median per grade for ALL years (what the alert would have shown) and for TEST 2022+ (honest), whether A > B > C > D on REACH and mean in the test, and the same for px<10. Pass = monotone in the test on both REACH and mean with A's n >= 20; otherwise the grade is a descriptive label with its n shown."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('xs/report_489_informed_insiders_early_earnings.py').read(); exec(src[:src.index("B = {'ATRP'")])
it2 = K['items'].astype(str).str.split(','); K['NEGBAD'] = it2.apply(lambda l: any(x in l for x in ('3.01', '1.02', '1.03', '2.05', '2.06', '4.02'))); O['NEGBAD'] = flag('NEGBAD')
B = {'ATRP': 1, 'F11': 1, 'LTURN': 1, 'LMC': -1}
def score(D):
    R = []
    for f, s in B.items():
        x = D[f].astype(float); ref = np.sort(x[D.tr & x.notna()].values); q = np.searchsorted(ref, x.values) / len(ref); q = np.where(x.isna(), np.nan, q); R.append(q if s > 0 else 1 - q)
    return np.nanmean(np.vstack(R), axis=0)
O['hit'] = O.PEAK >= .10; O = O.sort_values('d').reset_index(drop=True); prior_n = np.zeros(len(O)); prior_h = np.zeros(len(O))
for tk, g in O.groupby('tk'):
    h = g.hit.values.astype(float); n = np.arange(len(g)); ch = np.r_[0, np.cumsum(h)[:-1]]
    prior_n[g.index] = n; prior_h[g.index] = ch
O['pn'] = prior_n; O['ph'] = prior_h; O['REC'] = (O.pn >= 5) & (O.ph / O.pn.clip(lower=1) >= .6)
for cap in (5, 10):
    S = O[O.px < cap].copy(); S['s_'] = score(S); cut = np.quantile(S[S.tr].s_.dropna(), .7); G = S[S.s_ >= cut].copy(); ac = np.quantile(G[G.tr].ATRP, 2 / 3); G['HI'] = G.ATRP >= ac
    G['pts'] = G.HI.astype(int) + G.FIRST.astype(int) + G.REC.astype(int) - G.EARLY30.astype(int) - 2 * G.NEGBAD.astype(int); G['grade'] = np.select([G.pts >= 2, G.pts == 1, G.pts == 0], ['A', 'B', 'C'], 'D')
    f = lambda x: f"{int(x.hit.sum()):3d}/{len(x):3d} = {x.hit.mean():.0%} reach | mean {x.r.mean():+6.1%} med {x.r.median():+.1%}" if len(x) else 'n 0'
    print(f'\n######## sub-${cap} live group: {len(G)} flips 2019-Mar 2026 (train {int(G.tr.sum())}, test {int((~G.tr).sum())}); ATRP top-third cut {ac:.3f}')
    for lab, E in (('ALL years', G), ('TEST 2022+', G[~G.tr])):
        print(f'  {lab}:'); [print(f'    grade {g}: {f(E[E.grade == g])}') for g in 'ABCD']
    T = G[~G.tr]; print('  components (TEST, flag vs rest):', ' | '.join(f"{c} {int(T[c].sum())}: {T[T[c]].r.mean():+.1%}/{T[T[c]].hit.mean():.0%} vs {T[~T[c]].r.mean():+.1%}/{T[~T[c]].hit.mean():.0%}" for c in ('HI', 'FIRST', 'REC', 'EARLY30', 'NEGBAD')))
    for g in 'ABCD':
        x = T[T.grade == g]
        if len(x) > 5:
            m = x.groupby('mo' if 'mo' in x else x.d.dt.to_period('M').astype(str)).r.mean()
    print('  era split of grade means (TEST): ' + ' | '.join(f"{g}: 2022 {T[(T.grade == g) & (T.d.dt.year == 2022)].r.mean():+.1%} (n {((T.grade == g) & (T.d.dt.year == 2022)).sum()}) 2023-26 {T[(T.grade == g) & (T.d.dt.year >= 2023)].r.mean():+.1%} (n {((T.grade == g) & (T.d.dt.year >= 2023)).sum()})" for g in 'ABCD'))
