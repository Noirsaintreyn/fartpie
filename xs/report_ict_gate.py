"""Follow-up to report_ict: gate effect (skip trades) and difference tests, per era. Nov 11 2026."""
import numpy as np, pandas as pd
SC = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
def wt(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) if len(a) > 5 and len(b) > 5 else np.nan
for nm in ('V15a', 'RA'):
    R = pd.read_pickle(SC + f'ict_{nm}.pkl'); R['yr'] = R.day.dt.year
    G = {'against displacement': R.disp == -1, 'against structure (MSS)': R.mss == -1, 'against disp OR mss': (R.disp == -1) | (R.mss == -1),
         'buying below 9:30 open': R.open_930 == 1, 'ADX low tercile': R.adx <= R.adx.quantile(1/3), 'RSI-aligned low tercile': R.rsi <= R.rsi.quantile(1/3),
         'disp/mss against OR ADX low': (R.disp == -1) | (R.mss == -1) | (R.adx <= R.adx.quantile(1/3)), 'any SMT present': R.smt != 0}
    print(f'\n== {nm}: base n {len(R)} total {R.pts.sum():+.0f} mean {R.pts.mean():+.1f} win {np.mean(R.pts>0):.0%} worst-year {R.groupby("yr").pts.sum().min():+.0f}')
    for k, g in G.items():
        s, kp = R[g], R[~g]; ey = kp.groupby('yr').pts.sum()
        dA = wt(kp[kp.era == 'A'].pts, s[s.era == 'A'].pts); dB = wt(kp[kp.era == 'B'].pts, s[s.era == 'B'].pts)
        print(f' skip {k:30} skipped n {len(s):3d} (mean {s.pts.mean():+6.1f}, win {np.mean(s.pts>0):3.0%}) | kept n {len(kp):3d} mean {kp.pts.mean():+5.1f} win {np.mean(kp.pts>0):3.0%} total {kp.pts.sum():+6.0f} worst-yr {ey.min():+5.0f} | kept-vs-skipped t: A {dA:+.1f} B {dB:+.1f}')
