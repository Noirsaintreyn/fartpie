"""Follow-up to report_80 (declared before results). report_80's early-vs-late split used the hindsight ratio L/M, which is partly circular (a huge move makes L/M small by construction). The part a classifier can OBSERVE at entry is L itself (entry / trough-of-the-bear-leg - 1, how far price has already come off the low) and days since the trough, also scaled by the stock's ATR%. Test per era: terciles of L, L/ATR% and days-since-trough, mean net (30bp), median, HUGE per 100, on all flips and on the screener rule (BV>=1.5). Hypothesis (Rey): flips that print closer to the trough pay more. Pass = same direction in both eras for mean and HUGE rate."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('report_80_leg_anatomy.py').read(); exec(src.split("lg = lambda x: np.log1p(x)")[0])
P2 = P.reset_index(drop=True); A['ATRP'] = P2.loc[A.i, 'ATRP'].values; A['rn'] = P2.loc[A.i, 'ret'].values - 0.003; A['LA'] = A.L / A.ATRP
for lab, S0 in (('ALL FLIPS', A), ('SCREENER RULE BV>=1.5', A[A.BV >= 1.5])):
    print(f'\n######## {lab}')
    for f, nm in (('L', 'L: already off the trough'), ('LA', 'L in ATR% units'), ('dte', 'days since trough')):
        for en in ('2019-21', '2022-26Q1'):
            z = S0[S0.era == en].copy(); z['q'] = pd.qcut(z[f].rank(method='first'), 3, labels=['lowest third (EARLY)', 'middle', 'highest third (LATE)'])
            print(f'  {nm:26s} {en:9s}: ' + ' | '.join(f'{k[:5]} {v[f].median():6.2f} n {len(v)} mean {v.rn.mean()*100:+6.2f}% med {v.rn.median()*100:+5.2f} HUGE/100 {v.HUGE.mean()*100:4.2f}' for k, v in z.groupby('q')))
