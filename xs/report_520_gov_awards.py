"""Report 520: federal award (USAspending contract/grant/loan, amount>=$0.5M, start date in the prior 90d but not the last 3d) as a lead. Strict-name-matched coverage only 61 of 508 names (disclosed). Same template as 518/519; start date = signing date, public with a lag (mild foresight, hence the 3-day gap)."""
import numpy as np, pandas as pd, io, contextlib, warnings; warnings.filterwarnings('ignore')
src = open('xs/report_516_sector.py').read().split("TRm = (P.yr")[0]
with contextlib.redirect_stdout(io.StringIO()): exec(src)
P['pb'] = np.select([P.c < 1, P.c < 2.5], [0, 1], 2); P['era4'] = np.select([P.yr == 2018, P.yr <= 2021, P.yr == 2022], ['2018', '2019-21', '2022'], '2023-26')
A = pd.read_pickle(X + 'usaspending_awards.pkl'); A['start'] = pd.to_datetime(A.start).astype('datetime64[ns]'); A = A[A.amount.astype(float) >= 5e5]; cov = set(A.tk); print('names with awards', len(cov), '| awards', len(A), '| by group', A.group.value_counts().to_dict(), '| sector of covered names', S[S.tk.isin(cov)].sector.value_counts().to_dict())
g = {t: np.sort(x.start.values) for t, x in A.groupby('tk')}; P['AWD'] = np.nan; idx = pd.Series(range(len(P))).groupby(P.tk.values).indices; arr = np.full(len(P), np.nan)
for tk, ii in idx.items():
    if tk not in g: continue
    d = P.d.values[ii]; a = g[tk]; arr[ii] = (np.searchsorted(a, d - np.timedelta64(3, 'D'), side='right') > np.searchsorted(a, d - np.timedelta64(90, 'D'), side='left')).astype(float)
P['AWD'] = arr; C = P[P.AWD.notna()].copy(); C['ab10'] = pd.qcut(C.ATRP.rank(method='first'), 10, labels=False); print('covered stock-days', len(C), 'AWD=1 share', round(C.AWD.mean(), 2), 'base P(y) covered', round(C.y.mean(), 3))
def lift2(G, f):
    d = G.assign(f=f.loc[G.index]); d = d[d.f.isin([0, 1])]; r = []
    for (m, b, p), gp in d.groupby(['mon', 'ab10', 'pb']):
        a, c = gp[gp.f == 1].y, gp[gp.f == 0].y
        if len(a) >= 2 and len(c) >= 2: r.append((m, a.mean() - c.mean(), len(a)))
    if len(r) < 4: return np.nan, np.nan
    o = pd.DataFrame(r, columns=['m', 'x', 'n']); mm = o.groupby('m').apply(lambda z: np.average(z.x, weights=z.n)); s = mm.std(); return mm.mean(), (mm.mean() / (s / np.sqrt(len(mm))) if s and s > 0 else np.nan)
for lab, G in (('ALL covered', C), ('TECH', C[C.sector == 'TECH']), ('ENERGY', C[C.sector == 'ENERGY']), ('HEALTH', C[C.sector == 'HEALTH']), ('OTHER', C[C.sector == 'OTHER'])):
    if len(G) < 1500: print(lab, 'n too small', len(G)); continue
    d, t = lift2(G, G.AWD); row = {e: round(lift2(G[G.era4 == e], G.AWD)[0] * 100, 1) if lift2(G[G.era4 == e], G.AWD)[0] == lift2(G[G.era4 == e], G.AWD)[0] else None for e in ('2018', '2019-21', '2022', '2023-26')}
    print(f'{lab:12s} n={len(G):6d} award-in-90d share {G.AWD.mean():.2f}  P1 {G[G.AWD == 1].y.mean():.3f} P0 {G[G.AWD == 0].y.mean():.3f}  within-ATRP lift {d * 100:+.1f}pp t {t:.1f}  eras {row}')
