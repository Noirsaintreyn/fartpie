"""Nov 14 2026 ARE LESS-EFFICIENT ASSETS MORE TRENDING, AND IS THE VECTOR A TREND MEASURE OR A STATE CLASSIFIER? (Rey: expect trend tools to work better on less efficient assets, maybe trends are just less crowded there; vector spots the START of a move, so it is a classifier, not a trend measure.) Declared before results. Part A: trend persistence by liquidity. For each ticker and era, variance ratio VR(h) = var of h-day returns / (h * var of 1-day returns) for h = 5, 20, 60 (overlapping, standard small-sample correction ignored; same for all groups). VR>1 = trending, <1 = mean reverting. Report the MEDIAN VR across tickers in each dollar-volume tercile (ticker's own median dollar volume per era), per era. Hypothesis: low-liquidity tercile has higher VR. Part B: what the vector flip adds. Flips by liquidity tercile (LDV at the flip): n, mean net, median, HUGE per 100, median hold in days and share held >60d. Part C: is the flip a start or a continuation? Return of the 20 days BEFORE the flip (R20) and the 60 days before (R60 from R120 approx not available -> use R20 only) vs the flip's later return, by era: correlation and mean net for flips after a down vs up 20 days."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; D = pd.read_pickle(SC + 'wide_daily_ohlcv.pkl'); P = pd.read_pickle(SC + 'wide_flips_feat.pkl'); P['rn'] = P.ret - 0.003
def vr(c, h):
    r1 = np.log(c).diff().dropna(); rh = np.log(c).diff(h).dropna()
    return np.nan if len(r1) < 100 or r1.var() == 0 else rh.var() / (h * r1.var())
print('PART A: median variance ratio by liquidity tercile (VR>1 trending, <1 mean-reverting); n tickers in brackets')
rows = []
for tk, B in D.items():
    B = B[~B.index.duplicated()]
    for en, a, b in (('2019-21', '2019', '2022'), ('2022-26', '2022', '2027')):
        z = B[(B.index >= a) & (B.index < b)]
        if len(z) < 300: continue
        c = z.close[z.close > 1]; dv = (z.close * z.volume).median()
        if len(c) < 300: continue
        rows.append(dict(tk=tk, era=en, dv=dv, vr5=vr(c, 5), vr20=vr(c, 20), vr60=vr(c, 60)))
R = pd.DataFrame(rows)
for en in ('2019-21', '2022-26'):
    z = R[R.era == en].dropna(); z['ter'] = pd.qcut(z.dv, 3, labels=['LOW liq', 'MID', 'HIGH liq']); g = z.groupby('ter')
    print(f'  {en}: ' + ' | '.join(f'{k}: VR5 {v.vr5.median():.2f} VR20 {v.vr20.median():.2f} VR60 {v.vr60.median():.2f} [{len(v)}, med $vol {v.dv.median()/1e6:.1f}M]' for k, v in g))
print('\nPART B: flips by liquidity tercile (LDV at flip; terciles within era)')
for en, z in (('2019-21', P[P.t < '2022']), ('2022-26Q1', P[P.t >= '2022'])):
    z = z.copy(); z['ter'] = pd.qcut(z.LDV, 3, labels=['LOW liq', 'MID', 'HIGH liq'])
    print(f'  {en}: ' + ' | '.join(f'{k}: n {len(v)} mean {v.rn.mean()*100:+.2f}% med {v.rn.median()*100:+.2f} HUGE/100 {v.HUGE.mean()*100:.2f} medhold {v.hold.median():.0f}d >60d {(v.hold>60).mean()*100:.0f}%' for k, v in z.groupby('ter')))
print('\nPART C: start or continuation? flips after a falling vs rising 20 days')
for en, z in (('2019-21', P[P.t < '2022']), ('2022-26Q1', P[P.t >= '2022'])):
    z = z.dropna(subset=['R20']); q = z.R20.rank(pct=True)
    print(f'  {en}: spearman(R20 before, net return after) {z.R20.corr(z.rn, method="spearman"):+.3f} | lowest-third R20 mean {z[q<=1/3].rn.mean()*100:+.2f}% HUGE/100 {z[q<=1/3].HUGE.mean()*100:.2f} | highest-third {z[q>=2/3].rn.mean()*100:+.2f}% HUGE/100 {z[q>=2/3].HUGE.mean()*100:.2f}')
