"""Report 466 (Nov 24 2026): a GROUP of low-priced flips with a HIGH probability of a >=10% move. Rey: 30% is lovely, but a group with a high chance of >10% is enough; 10 stocks, one runs, enter off the flip.
DECLARED BEFORE RESULTS. Score = mean of train-ECDF ranks of 6 features chosen from report 465 (all passed BOTH targets, signs fixed from 465, no tuning): ATRP (+), F11 distance to level (+), VOLX volume surge (+), LTURN turnover (+), log cash runway (- : short runway), LMC market cap (- : small). Missing runway -> mean of the other features. Rank cuts frozen on TRAIN (2019-21); groups = top 5% / 10% / 20% of the train score cut; tested on 2022+.
Targets: REAL = the flip's realised return r >= +10%; REACH = the flip's peak return (before its vector exit) >= +10%. Also compare with the same-size top group of the live outlier score OS, and a 'top 10 flips per month' basket.
Report per group (test): n and n per year, P(REAL), P(REACH), mean/median/win, month-paired mean diff vs the rest, and P(REAL)/P(REACH) by era (2022 | 2023-26). Declared bar for 'high probability': P(REAL >= 10%) >= 40% in BOTH test eras with n >= 40 per era, else report the best honest group."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('xs/report_465_mover_features.py').read(); exec(src[:src.index("from scipy.stats import rankdata")])
O['PEAKv'] = O.PEAK
feats = {'ATRP': 1, 'F11': 1, 'VOLX': 1, 'LTURN': 1, 'log_runway': -1, 'LMC': -1}
tr = O.tr.values; R = pd.DataFrame(index=O.index)
for f, s in feats.items():
    x = O[f].astype(float); ref = np.sort(x[tr & x.notna()].values); R[f] = (np.searchsorted(ref, x.values) / len(ref)) if s > 0 else (1 - np.searchsorted(ref, x.values) / len(ref)); R.loc[x.isna(), f] = np.nan
O['score'] = R.mean(axis=1); O['pass_n'] = R.notna().sum(axis=1)
refs = O.score[tr].dropna(); O['yr'] = O.d.dt.year; te = O[~O.tr].copy(); trn = O[O.tr]
print('flips', len(O), 'train', len(trn), 'test', len(te), '| test years', sorted(te.yr.unique()), '| base test: P(REAL>=10%)', f"{(te.r >= .1).mean():.1%}", 'P(REACH>=10%)', f"{(te.PEAKv >= .1).mean():.1%}", 'median', f"{te.r.median() * 100:+.1f}%")
def mp_(a, b):
    x = (a.groupby('mon').r.mean() - b.groupby('mon').r.mean()).dropna(); return x.mean() * 100, (x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan)
def line(name, a, rest, yrs):
    d, t = mp_(a, rest); e = ' | '.join(f"{e}: REAL {(a[a.era3 == e].r >= .1).mean():.0%} REACH {(a[a.era3 == e].PEAKv >= .1).mean():.0%} (n{(a.era3 == e).sum()})" for e in ('2022', '2023-26'))
    print(f"{name:26s} n {len(a):4d} ({len(a) / yrs:4.0f}/yr) | P(REAL>=10%) {(a.r >= .1).mean():5.1%} P(REACH>=10%) {(a.PEAKv >= .1).mean():5.1%} | mean {a.r.mean() * 100:+6.1f}% median {a.r.median() * 100:+6.1f}% win {(a.r > 0).mean():3.0%} | vs rest {d:+5.1f}pp t {t:4.2f} | {e}")
yrs = te.d.dt.year.nunique() - 1 + (te.d.max().dayofyear / 365)
for sub, label in ((lambda D: D[D.px < 10], 'px < $10'), (lambda D: D, 'px < $20')):
    T = sub(te).dropna(subset=['score']); print(f'\n##### {label}: test flips {len(T)}')
    for q in (.80, .90, .95):
        cut = np.quantile(sub(trn).score.dropna(), q); a = T[T.score >= cut]; line(f'score top {int((1 - q) * 100)}% (train cut)', a, T[T.score < cut], yrs)
    cut = np.quantile(sub(trn).OS.dropna(), .90); a = T[T.OS >= cut]; line('OS top 10% (train cut)', a, T[T.OS < cut], yrs)
    bk = T.sort_values('score', ascending=False).groupby('mon').head(10); line('top-10 per month by score', bk, T[~T.index.isin(bk.index)], yrs)
    cut = np.quantile(sub(trn).score.dropna(), .8); a = T[T.score >= cut]
    print('   per-year in score top 20%: ' + ' '.join(f"{y}: REAL {(g.r >= .1).mean():.0%} REACH {(g.PEAKv >= .1).mean():.0%} n{len(g)}" for y, g in a.groupby('yr')))
    print('   P(REAL>=10%) by score decile of TEST (train cuts): ' + ' '.join(f"{(g.r >= .1).mean():.0%}" for _, g in T.groupby(pd.cut(T.score, np.quantile(sub(trn).score.dropna(), np.linspace(0, 1, 11)), include_lowest=True, duplicates='drop'))))
