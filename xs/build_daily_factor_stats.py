"""Nov 25 2026: frozen statistics behind the alert's SETUP SCORECARD (report 492/493 machinery). Sub-$5 live group (4-feature base5 top 30%), daily flips 2019-Mar 2026. For every factor: hits/n (hit = flip reaches +10% before the vector exit) and mean return with the factor ON and OFF; grade cohort table A-D (points rule of report 492); per-ticker flip record; ATRP cut; filing-similarity cut (report 491). Written to daily_factor_stats.json (read by catalysts.scorecard)."""
import io, contextlib, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(open('xs/report_492_setup_grade.py').read().replace("for cap in (5, 10):", "for cap in ():"))
    S = O[O.px < 5].copy(); S['s_'] = score(S); cut5 = np.quantile(S[S.tr].s_.dropna(), .7); G = S[S.s_ >= cut5].copy(); ac = float(np.quantile(G[G.tr].ATRP, 2 / 3)); G['HI'] = G.ATRP >= ac
    sh = np.array([early(t, d, 30) for t, d in zip(G.tk.values, G.d.values)]); G['LATE30'] = sh >= 5
    G['pts'] = G.HI.astype(int) + G.FIRST.astype(int) + G.REC.astype(int) - G.EARLY30.astype(int) - 2 * G.NEGBAD.astype(int); G['grade'] = np.select([G.pts >= 2, G.pts == 1, G.pts == 0], ['A', 'B', 'C'], 'D')
st = lambda x: dict(hits=int(x.hit.sum()), n=int(len(x)), mean=float(x.r.mean()) if len(x) else None, median=float(x.r.median()) if len(x) else None)
st = lambda x: dict(hits=int(x.hit.sum()), n=int(len(x)), mean=float(x.r.mean()) if len(x) else None, median=float(x.r.median()) if len(x) else None)
out = dict(atrp_cut=ac, n_group=int(len(G)), factors={}, grades={}, grades_test={}, stock={}, built='2026-11-25')
for f in ('HI', 'FIRST', 'REC', 'EARLY30', 'LATE30', 'NEGBAD'): out['factors'][f] = dict(on=st(G[G[f]]), off=st(G[~G[f]]))
for g in 'ABCD': out['grades'][g] = st(G[G.grade == g]); out['grades_test'][g] = st(G[(G.grade == g) & (~G.tr)])
out['grades']['ALL'] = st(G)
for tk, x in O.groupby('tk'): out['stock'][tk] = [int(x.hit.sum()), int(len(x))]                    # every flip of the ticker in the table (px<20), through Mar 2026
G5 = G; src491 = open('xs/report_491_lazy_prices.py').read()
with contextlib.redirect_stdout(buf):
    exec(src491[:src491.index('t0 = time.time()')])
    sims = []
    for tk, d in zip(G.tk, G.d):
        try: p = pair(tk, d)
        except Exception: p = None
        sims.append(p[0] if p else np.nan)
    SM = pd.DataFrame({'tk': G.tk.values, 'd': G.d.values, 'sim': sims}).dropna()
GG = G5.merge(SM, on=['tk', 'd'], how='inner')
st = lambda x: dict(hits=int(x.hit.sum()), n=int(len(x)), mean=float(x.r.mean()) if len(x) else None, median=float(x.r.median()) if len(x) else None)
lo = float(GG.sim.quantile(1 / 3)); out['filing'] = dict(cut=lo, changed=st(GG[GG.sim <= lo]), same=st(GG[GG.sim > lo]), note='test 2022+ flips with a 10-K/10-Q pair; changed = lowest tercile of cosine similarity vs the same form a year earlier')
json.dump(out, open('daily_factor_stats.json', 'w'), indent=1); print(json.dumps({k: v for k, v in out.items() if k != 'stock'}, indent=1)[:2600])
