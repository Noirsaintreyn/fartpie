"""Nov 10 2026: TESTING THE FLIP-CONTEXT EDITS (declared before results). Candidates read off the flip_context tables, so judged strictly OUT OF SAMPLE: the bucket was weak in 2018-21 first, and the verdict is 2022-26.
 E1 M1 midday (12:00-13:59 entries) x0 | E2 M1 midday x0.5 | E3 R1 midday x0 | E4 RULES first hour (09:35-10:29) x0.5 | E5 RULES far from the 52-week POC (dpoc_y top third, cut from 2018-21) x0.5
 E6 M1 morning (before 10:30) x1.25 (exploratory, the strong bucket) | E7 = E2 + E3 + E4 + E5 | E8 = E7 + E6.
Report system weighted points / day, daily std, mean/std, worst month, both periods, vs the live base (S6 V2 + mv30 skip, RTH engines x live sizes x engine weights; ON legs unchanged)."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('/home/claude/fartpie/xs/flip_context.py').read(); exec(src[:src.index('def tt(')])
PER = (('18-21', lambda x: x < '2022-01-01'), ('22-26', lambda x: x >= '2022-01-01')); DAYS_ = pd.DatetimeIndex([d for d in DAYS if d >= pd.Timestamp('2018-02-01')])
q_y = TR['RULES'].dpoc_y[TR['RULES'].day < '2022-01-01'].quantile(2 / 3)
MULT = {
 'E1 M1 midday x0': {'M1': lambda t: np.where(t.tod == 2, 0.0, 1.0)}, 'E2 M1 midday x0.5': {'M1': lambda t: np.where(t.tod == 2, .5, 1.0)}, 'E3 R1 midday x0': {'R1': lambda t: np.where(t.tod == 2, 0.0, 1.0)},
 'E4 RULES first hour x0.5': {'RULES': lambda t: np.where(t.tod == 0, .5, 1.0)}, 'E5 RULES far from 52w POC x0.5': {'RULES': lambda t: np.where(t.dpoc_y > q_y, .5, 1.0)},
 'E6 M1 morning x1.25': {'M1': lambda t: np.where(t.tod == 0, 1.25, 1.0)}}
MULT['E7 = E2+E3+E4+E5'] = {'M1': MULT['E2 M1 midday x0.5']['M1'], 'R1': MULT['E3 R1 midday x0']['R1'], 'RULES': lambda t: np.where(t.tod == 0, .5, 1.0) * np.where(t.dpoc_y > q_y, .5, 1.0)}
MULT['E8 = E7 + E6'] = {**MULT['E7 = E2+E3+E4+E5'], 'M1': lambda t: np.where(t.tod == 2, .5, np.where(t.tod == 0, 1.25, 1.0))}
def system(mm):
    s = pd.Series(0.0, index=DAYS_)
    for n, t in TR.items():
        mu = mm[n](t) if n in mm else 1.0; s = s.add((t.pts * t.sz * UW[n] * mu).groupby(t.day.values).sum().reindex(DAYS_).fillna(0.0), fill_value=0)
    return s
def rep(lab, mm):
    s = system(mm); o = []
    for nm, f in PER:
        x = s[f(s.index)]; mo_ = x.groupby(x.index.to_period('M')).sum(); o.append(f'{nm}: {x.mean():+5.2f}/d std {x.std():5.1f} m/s {x.mean() / x.std():.3f} worstM {mo_.min():+5.0f} M>0 {np.mean(mo_ > 0):.0%}')
    print(f'  {lab:<32}' + ' | '.join(o), flush=True)
rep('BASE', {})
for k, v in MULT.items(): rep(k, v)
print('\nper-engine bucket check (pts per trade, early // late):')
for n, col, k, lab in (('M1', 'tod', 2, 'midday'), ('M1', 'tod', 0, 'morning'), ('R1', 'tod', 2, 'midday'), ('RULES', 'tod', 0, 'first hour')):
    t = TR[n]; o = []
    for nm, f in PER:
        x = t[f(t.day)]; a, b = x[x[col] == k].pts, x[x[col] != k].pts; o.append(f'{a.mean():+6.1f} (n{len(a)}) vs rest {b.mean():+6.1f} (t {(a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b)):+5.2f})')
    print(f'  {n} {lab:<10} {o[0]} // {o[1]}')
