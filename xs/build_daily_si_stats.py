"""Short-interest stats for the live sub-$5 group (4-feature base5 top 30%, train cut): SI_PCT (short >= 15% of shares) vs regular. Writes si block into daily_factor_stats.json."""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
src = open('xs/report_497_titles_dilution_shortint.py').read(); exec(src[:src.index("flags = ['BUY_TOP'")]); O['sip'] = sip; O['dtc'] = dtc
B = {'ATRP': 1, 'F11': 1, 'LTURN': 1, 'LMC': -1}
def score(D):
    R = []
    for f, s in B.items():
        x = D[f].astype(float); ref = np.sort(x[D.tr & x.notna()].values); q = np.searchsorted(ref, x.values) / len(ref); q = np.where(x.isna(), np.nan, q); R.append(q if s > 0 else 1 - q)
    return np.nanmean(np.vstack(R), axis=0)
S = O[O.px < 5].copy(); S['s_'] = score(S); cut = np.quantile(S[S.tr].s_.dropna(), .7); G = S[(S.s_ >= cut) & S.sip.notna()].copy(); G['SI'] = G.sip >= .15
f = lambda x: dict(hits=int(x.hit.sum()), n=int(len(x)), reach=float(x.hit.mean()) if len(x) else None, mean=float(x.r.mean()) if len(x) else None, median=float(x.r.median()) if len(x) else None)
out = {}
for lab, E in (('all', G), ('test', G[~G.tr])):
    out[lab] = dict(si=f(E[E.SI]), reg=f(E[~E.SI]), mid=f(E[(E.sip >= .05) & (E.sip < .15)]), lo=f(E[E.sip < .05]))
    print(lab, json.dumps(out[lab]))
    t = E[E.SI]; r = E[~E.SI]; print(f"  {lab}: SI>=15% n {len(t)} reach {t.hit.mean():.0%} mean {t.r.mean():+.1%} median {t.r.median():+.1%} p10 {t.r.quantile(.1):+.1%} | regular n {len(r)} reach {r.hit.mean():.0%} mean {r.r.mean():+.1%} median {r.r.median():+.1%} p10 {r.r.quantile(.1):+.1%}")
for y0, y1, nm in ((2019, 2021, 'train'), (2022, 2022, '2022'), (2023, 2026, '2023-26')):
    E = G[(G.d.dt.year >= y0) & (G.d.dt.year <= y1)]; print(f"  era {nm}: SI n {int(E.SI.sum())} reach {E[E.SI].hit.mean():.0%} mean {E[E.SI].r.mean():+.1%} | regular n {int((~E.SI).sum())} reach {E[~E.SI].hit.mean():.0%} mean {E[~E.SI].r.mean():+.1%}")
J = json.load(open('daily_factor_stats.json')); J['si'] = out; json.dump(J, open('daily_factor_stats.json', 'w'))
