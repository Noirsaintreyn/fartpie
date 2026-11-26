"""flip_history.csv for the small-cap report (Nov 26 2026): every sub-$5 daily flip since 2019 with its setup readings (ATRP, overhead supply OVH, breadth BRD) and outcome (hit = PEAK>=+10% before the vector exit, PEAK, r) + frozen tertile cuts (train 2019-21) -> flip_history_cuts.json. Feeds report_pages.py: 'what happened the last times THIS stock / flips like this were in this spot'."""
import json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('xs/report_502_breakout_pairings.py').read(); exec(src[:src.index("def eff(D, flag, y):")])
S = O[(O.px < 5) & O.ATRP.notna()].copy(); tr = S.tr
S['hit'] = (S.PEAK >= .10).astype(int)
cuts = dict(atrp=[float(x) for x in np.nanquantile(S.ATRP[tr], [1/3, 2/3])], ovh=[float(x) for x in np.nanquantile(S.OVH[tr], [1/3, 2/3])], brd=[float(x) for x in np.quantile(S.BRD[tr], [1/3, 2/3])])
S[['tk', 'd', 'px', 'ATRP', 'OVH', 'BRD', 'PEAK', 'r', 'hit']].sort_values(['tk', 'd']).to_csv('flip_history.csv', index=False); json.dump(cuts, open('flip_history_cuts.json', 'w'))
print(len(S), S.tk.nunique(), cuts, 'OVH known', S.OVH.notna().mean())
