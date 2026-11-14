"""Follow-up to report_73 (declared before results): the one driver that passed is the 60-day change in the 2y Treasury yield (falling = easing = better). Express it ex ante and show economics: (1) expanding-window terciles (rank of today's value vs ALL PRIOR months only) -> mean monthly outcomes by tercile and era for O1 (all flips), O3 (high-short & basket leads), O4 (strong sector lead); (2) the flip-level version: flips entered in months classified EASING (value in the lower expanding tercile) vs TIGHTENING (upper) -> mean net, median, HUGE per 100 by era; (3) the current reading. No threshold tuning: terciles only."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; Y = pd.read_pickle(SC + 'regime_monthly.pkl'); FR = pd.read_pickle(SC + 'fred.pkl'); FR['date'] = pd.to_datetime(FR.date); FR = FR.set_index('date').sort_index()
y2 = FR.DGS2.dropna(); chg = y2 - y2.shift(60); print('current 2y yield', round(float(y2.iloc[-1]), 2), 'on', y2.index[-1].date(), '| 60d change', round(float(chg.iloc[-1]), 2), 'pts')
x = Y.X5_2ychg60.copy(); cls = pd.Series('mid', index=Y.index)
for i in range(len(Y)):
    h = x.iloc[:i].dropna()
    if len(h) < 12: cls.iloc[i] = 'na'; continue
    lo, hi = h.quantile([1 / 3, 2 / 3]); cls.iloc[i] = 'EASING' if x.iloc[i] <= lo else ('TIGHTENING' if x.iloc[i] >= hi else 'mid')
Y['cls'] = cls; print('\n(1) monthly outcome means by expanding tercile of the 2y-yield 60d change (O1 all-flip mean %, O3 high-short&lead mean %, O4 strong-sector-lead mean %, O2 HUGE per 100)')
for e, z in (('2019-21', Y[Y.era == 'A']), ('2022-26Q1', Y[Y.era == 'B']), ('ALL', Y)):
    print(f'  {e}: ' + ' | '.join(f'{c}: n {int((z.cls==c).sum())} O1 {z[z.cls==c].O1.mean()*100:+.2f} O2 {z[z.cls==c].O2.mean():.2f} O3 {z[z.cls==c].O3.mean()*100:+.1f} O4 {z[z.cls==c].O4.mean()*100:+.1f}' for c in ('EASING', 'mid', 'TIGHTENING')))
src = open('report_71_sector_and_si_change.py').read(); exec(src.split("def mc(S, m, base):")[0].replace("src = open('report_69_short_regime.py').read(); exec(src.split(\"print('PART A\")[0])", "src0 = open('report_69_short_regime.py').read(); exec(src0.split(\"print('PART A\")[0])"))
P['mo'] = P.t.dt.to_period('M'); P['cls'] = P.mo.map(Y.cls); print('\n(2) flip-level: flips entered in EASING vs TIGHTENING months (class known at month start)')
for lab, f in (('all flips', lambda S: S), ('flip-bar volume >=1.5x', lambda S: S[S.BV >= 1.5]), ('high-short & basket lead', lambda S: S[S.SQ10 & S.LEAD]), ('strong sector lead >5%', lambda S: S[S.GVAL > .05]), ('insider buy & BV>=1.5', lambda S: S[(S.NI >= 1) & (S.BV >= 1.5)])):
    for en, a, b in (('2019-21', '2019', '2022'), ('2022-26Q1', '2022', '2026-04')):
        S = P[(P.t >= a) & (P.t < b)]; D = f(S); out = []
        for c in ('EASING', 'mid', 'TIGHTENING'):
            z = D[D.cls == c]; out.append(f'{c[:5]} n {len(z):5d} mean {z.rn.mean()*100 if len(z) else np.nan:+6.2f}% med {z.rn.median()*100 if len(z) else np.nan:+5.2f} HUGE/100 {z.HUGE.mean()*100 if len(z) else np.nan:4.2f}')
        print(f'  {lab:26s} {en}: ' + ' | '.join(out))
