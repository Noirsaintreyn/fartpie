"""Oct 26 2026: bigger book when macro says vector trades work best (the three inputs whose direction held in both periods on admitted
trades: asset managers' S&P positioning low = not crowded, VIX/VIX3M high = stress, SPY close to / under its 200-day = not stretched).
Thresholds = terciles of admitted 2019-21 trades. 'good' = in the favorable third; score = how many of the three are favorable.
 lever: up to 8 (or 10) positions at 1/6 each when good (up to 133% / 167% invested), 6 otherwise
 more names: 8 positions at 1/8 each when good, 6 at 1/6 otherwise (always <= 100%)"""
import numpy as np, pandas as pd
src = open('xs/macro_lab.py').read(); exec(src[:src.index("B = L[L.inbook]; rows = []")])
ss2 = ss.replace('def sim_cap(', 'def sim_cw(').replace("for e, x, r_, ww, cp in", "for e, x, r_, ww, cp, wn in").replace("Z.w.values, Z.cap.values)", "Z.w.values, Z.cap.values, Z.wn.values)").replace("pend.append((x, r_, ww / n))", "pend.append((x, r_, ww / wn))")
exec(ss2)
PER = [(np.datetime64(a), np.datetime64(b)) for a, b in (('2019-01-01', '2022-01-01'), ('2022-01-01', '2027-01-01'), ('2019-01-01', '2027-01-01'), ('2026-01-01', '2027-01-01'))]
B = L[L.inbook & (L.half == '19-21')]
good = {'not crowded (asset mgrs low third)': L.es_am < B.es_am.quantile(1 / 3), 'VIX stress (VIX/VIX3M high third)': L.vix_ts > B.vix_ts.quantile(2 / 3),
        'SPY not stretched (200d low third)': L.spy200 < B.spy200.quantile(1 / 3)}
score = sum(g.astype(int) for g in good.values()); good['any 1 of 3'] = score >= 1; good['2+ of 3'] = score >= 2
print('share of signals in each state:', {k: round(v.mean(), 2) for k, v in good.items()})
def show(nm, Z):
    R_ = [sim_cw(Z, a, b) for a, b in PER]; flag = '  <-- outside the luck band in both' if R_[0][0] > .60 and R_[1][0] > .30 else ''
    print(f'  {nm:<58}' + ''.join(f'{c:>+9.1%} /{d:>5.0%}' for c, d in R_) + flag, flush=True)
print(f'\n{"":<60}{"2019-21":>16}{"2022-26":>16}{"2019-26":>16}{"2026 YTD":>16}')
z = L.copy(); z['cap'] = 6; z['wn'] = 6; show('live (6 x 1/6)', z)
for k, g in good.items():
    for cap, wn, lab in ((8, 6, 'up to 8 x 1/6 (133%)'), (10, 6, 'up to 10 x 1/6 (167%)'), (8, 8, '8 x 1/8 (100%)')):
        z = L.copy(); z['cap'] = np.where(g, cap, 6); z['wn'] = np.where(g, wn, 6); show(f'{k}: {lab}', z)
for cap, lab in ((8, 'always up to 8 x 1/6 (control: leverage with no macro)'), (10, 'always up to 10 x 1/6 (control)')):
    z = L.copy(); z['cap'] = cap; z['wn'] = 6; show(lab, z)
