"""Report 483 (Nov 25 2026): Rey - replace the arbitrary -20% gate with a drop RELATIVE to the stock's own ATR, and decide rank vs cut. Same daily top-30% base5 group, test 2022+, px<20/<10/<5, r = vector exit (30bp).
DECLARED BEFORE RESULTS. DROPX = DROP / ATRP (how many average daily ranges the 20d drawdown was; ATRP = flip-day ATR as fraction of price). Gate = DROPX top tercile with cut frozen from TRAIN rows of the same bucket (no hand number). UNEXPL/EXPL as report 482.
Tests: (1) DROPX tercile table (mean, REACH, median, era split); (2) inside the top-DROPX tercile: UNEXPL vs EXPL; (3) spread UNEXPL-EXPL (deep) with month-clustered t, full test and 2022 alone; (4) RANK vs CUT: portfolio of all group flips, equal weight, mean r of (a) all, (b) CUT = drop explained deep flips, (c) RANK = weight 2 for deep+UNEXPL, 1 otherwise, 0.5 for deep+EXPL; compare mean, and month-clustered t of (RANK - ALL) paired by month.
Pass = deep UNEXPL minus deep EXPL spread positive with |t|>=2 clustered by month (full test) AND positive in 2022 alone."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('xs/report_482_overreaction_bottom.py').read(); exec(src[src.index('import numpy'):src.index('for cap in')].replace('import numpy as np, pandas as pd, warnings; warnings.filterwarnings(\'ignore\')',''))
def tm(x, m):
    g = pd.DataFrame({'x': x.values, 'm': m.values}).groupby('m').x.mean(); return g.mean(), g.mean() / (g.std() / np.sqrt(len(g))) if len(g) > 2 else np.nan
for cap in (20, 10, 5):
    S = O[(O.px < cap) & O.DROP.notna()].copy(); S['s_'] = score(S); cut = np.quantile(S[S.tr].s_.dropna(), .7)
    S['DX'] = -S.DROP / S.ATRP.astype(float); trn = S[S.tr & (S.s_ >= cut)]; q = np.quantile(trn.DX.dropna(), [1/3, 2/3])
    G = S[(~S.tr) & (S.s_ >= cut)].copy(); G['t3'] = np.searchsorted(q, G.DX) + 1; G['mo'] = G.d.dt.to_period('M').astype(str)
    print(f'\n######## sub-${cap}: {len(G)} flips | train DX tercile cuts {np.round(q,2)} (median DX {G.DX.median():.1f} ATRs)')
    for t, x in G.groupby('t3'): print(f'  tercile {t} (DX~{x.DX.mean():.1f}):', st(x), '|', eras(x))
    D = G[G.t3 == 3]; U = D[D.EXPL == 0]; E = D[D.EXPL == 1]
    print('  deep(top tercile):', st(D)); print('    UNEXPL:', st(U), '|', eras(U)); print('    EXPL  :', st(E), '|', eras(E))
    for lab, a, b in (('full', U, E), ('2022', U[U.d.dt.year == 2022], E[E.d.dt.year == 2022])):
        ma = a.groupby('mo').r.mean(); mb = b.groupby('mo').r.mean(); j = pd.concat([ma, mb], axis=1, keys=['u', 'e']).dropna(); df = j.u - j.e
        print(f'    spread UNEXPL-EXPL {lab}: months {len(j)} mean {df.mean():+.1%} t {df.mean()/(df.std()/np.sqrt(len(df))) if len(df)>2 else float("nan"):+.2f}')
    w = np.where((G.t3 == 3) & (G.EXPL == 0), 2., np.where((G.t3 == 3) & (G.EXPL == 1), .5, 1.)); G['w'] = w
    keep = ~((G.t3 == 3) & (G.EXPL == 1))
    mo = lambda f: G.assign(v=f).groupby('mo').apply(lambda x: (x.v * x.r).sum() / x.v.sum())
    A = mo(1.); C = mo(keep.astype(float).where(keep, 0) + 0.0001 * (~keep)); R = mo(G.w)
    for lab, X in (('CUT explained-deep', C), ('RANK 2/1/.5', R)):
        d = (X - A).dropna(); print(f'  {lab}: mean {X.mean():+.1%} vs ALL {A.mean():+.1%} | paired diff {d.mean():+.2%} t {d.mean()/(d.std()/np.sqrt(len(d))):+.2f} | kept {keep.mean():.0%} of flips')
