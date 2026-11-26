"""Builds daily_pairing_stats.json (sub-$5 flips, all years, reach +10% by OVH / BRD tercile, cuts frozen on train) and renders two example charts for a ticker."""
import sys, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('xs/report_502_breakout_pairings.py').read(); exec(src[:src.index("def eff(D, flag, y):")])
S = O[(O.px < 5) & O.ATRP.notna()].copy(); tr = S.tr
co = np.nanquantile(S.OVH[tr], [1/3, 2/3]); cb = np.quantile(S.BRD[tr], [1/3, 2/3]); S['ot'] = np.where(S.OVH <= co[0], 0, np.where(S.OVH >= co[1], 2, 1)); S['bt'] = np.where(S.BRD <= cb[0], 0, np.where(S.BRD >= cb[1], 2, 1))
f = lambda x: [None, float(x.hit.mean()), int(len(x))]
st = dict(ovh=dict(cut_lo=float(co[0]), cut_hi=float(co[1]), tiers=[[['Clear', *f(S[S.ot == k])[1:]][0], *f(S[S.ot == k])[1:]] for k in (0, 1, 2)], note='all sub-$5 flips since 2019 (n in brackets)'),
          brd=dict(cut_lo=float(cb[0]), cut_hi=float(cb[1]), tiers=[[['q', *f(S[S.bt == k])[1:]][0], *f(S[S.bt == k])[1:]] for k in (0, 1, 2)], note='all sub-$5 flips since 2019 (n in brackets)'))
json.dump(st, open('daily_pairing_stats.json', 'w')); print(json.dumps(st))
