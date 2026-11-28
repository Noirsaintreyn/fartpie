"""516b: robustness for the two features that cleared the strict bar in 516 (HEALTH RND_hi, TECH MARGIN_hi) + sector contagion with a LABEL shuffle (keeps the market-wide timing of runs, so only the sector-specific part of clustering is tested) + apply the health flags to the actual sub-$5 flip history."""
import numpy as np, pandas as pd, io, contextlib, warnings; warnings.filterwarnings('ignore')
src = open('xs/report_516_sector.py').read().split("TRm = (P.yr")[0]
with contextlib.redirect_stdout(io.StringIO()): exec(src)
TRm = (P.yr >= 2019) & (P.yr <= 2021); P['px'] = P.c; P['pb'] = np.select([P.px < 1, P.px < 2.5], [0, 1], 2)
def bins(G): return pd.qcut(G.ATRP.rank(method='first'), 10, labels=False)
def lift2(G, flag, key):
    d = G.assign(f=flag.loc[G.index]); d = d[d.f.isin([0, 1])]; r = []
    for (m, b, p), gp in d.groupby(['mon', key, 'pb']):
        a, c = gp[gp.f == 1].y, gp[gp.f == 0].y
        if len(a) >= 3 and len(c) >= 3: r.append((m, a.mean() - c.mean(), len(a)))
    if len(r) < 4: return np.nan, np.nan, np.nan, np.nan
    o = pd.DataFrame(r, columns=['m', 'x', 'n']); mm = o.groupby('m').apply(lambda z: np.average(z.x, weights=z.n)); sd = mm.std(); return mm.mean(), (mm.mean() / (sd / np.sqrt(len(mm))) if sd and sd > 0 else np.nan), d[d.f == 1].y.mean(), d[d.f == 0].y.mean()
def flagq(G, c):
    q = G.loc[TRm.reindex(G.index).fillna(False), c].quantile([.33, .67]); return pd.Series(np.where(G[c] > q.iloc[1], 1, np.where(G[c] <= q.iloc[0], 0, np.nan)), index=G.index)
print('== (b) finer control: ATRP DECILE x price bucket, month-clustered; by year')
for sc, c in (('HEALTH', 'RND'), ('HEALTH', 'CASHTA'), ('TECH', 'MARGIN'), ('OTHER', 'MARGIN')):
    G = P[P.sector == sc].copy(); G['ab10'] = bins(G); f = flagq(G, c); d, t, p1, p0 = lift2(G, f, 'ab10'); print(f'{sc} {c}_hi pooled finer-control lift {d*100:+.1f}pp t {t:.1f} (raw P1 {p1:.3f} P0 {p0:.3f})')
    print('   by year:', {y: f'{lift2(G[G.yr == y], f, "ab10")[0] * 100:+.1f}' for y in sorted(G.yr.unique()) if (G.yr == y).sum() > 1500 and flagq is not None})
    # one observation per name-month (cut overlap): take first stock-day of each name-month
# leave-one-NAME-out style: how many distinct names carry the flag-high doubles
G = P[P.sector == 'HEALTH'].copy(); f = flagq(G, 'RND'); hi = G[(f == 1) & (G.y == 1)]; print('HEALTH RND_hi: names with a double', hi.tk.nunique(), 'of', G[f == 1].tk.nunique(), '| top-5 names share of doubling stock-days', round(hi.tk.value_counts().head(5).sum() / len(hi), 2))
# ---- (a) contagion with label shuffle
E = pd.read_csv('xs/report_515_episodes.csv'); E = E.merge(S[['tk', 'sector']], on='tk', how='left'); E['sector'] = E.sector.fillna('OTHER'); E['start'] = pd.to_datetime(E.start); rng = np.random.default_rng(1)
dd = E.start.values.astype('datetime64[ns]'); tt = E.tk.astype(str).to_numpy(); W = (np.abs((dd[:, None] - dd[None, :]) / np.timedelta64(1, 'D')) <= 20) & (tt[:, None] != tt[None, :])
def same(lab): L = (lab[:, None] == lab[None, :]); return (W & L)
SEC = E.sector.astype(str).to_numpy(); obs = same(SEC); res = []
sims = [same(rng.permutation(SEC)) for _ in range(200)]
for sc in sorted(E.sector.unique()):
    m = (SEC == sc)
    if m.sum() < 25: continue
    o = obs[m].sum() / m.sum(); s = np.array([x[m].sum() / m.sum() for x in sims]); res.append((sc, int(m.sum()), round(o, 2), round(s.mean(), 2), round(o / s.mean(), 2), round(float((s >= o).mean()), 3)))
print('\n== (a) SECTOR-SPECIFIC contagion (episode sector labels shuffled across episodes, dates fixed: removes the market-wide timing of runs)'); print(pd.DataFrame(res, columns=['sector', 'episodes', 'obs_same_sector_neighbors', 'label_shuffle_mean', 'ratio', 'p']).to_string(index=False))
# ---- (c) apply to the actual sub-$5 flips
H5 = pd.read_csv('flip_history.csv', parse_dates=['d']); H5['d'] = H5.d.astype('datetime64[ns]'); H5 = H5.merge(S[['tk', 'sector']], on='tk', how='left'); H5['sector'] = H5.sector.fillna('OTHER')
FD2 = FD.copy(); H5 = pd.merge_asof(H5.sort_values('d'), FD2[['tk', 'filed', 'RND', 'CASHTA', 'MARGIN']], left_on='d', right_on='filed', by='tk', direction='backward', tolerance=pd.Timedelta(days=540))
Hh = H5[H5.sector == 'HEALTH'].copy(); print('\n== (c) HEALTH sub-$5 vector flips (flip_history), n =', len(Hh), 'with RND', Hh.RND.notna().sum())
qq = P[(P.sector == 'HEALTH') & TRm].RND.quantile([.33, .67]); Hh['RNDhi'] = np.where(Hh.RND > qq.iloc[1], 1, np.where(Hh.RND <= qq.iloc[0], 0, np.nan)); Hh['yr'] = Hh.d.dt.year; Hh['era4'] = np.select([Hh.yr == 2018, Hh.yr <= 2021, Hh.yr == 2022], ['2018', '2019-21', '2022'], '2023-26')
for lab, gx in (('all', Hh), ('2019-21', Hh[Hh.era4 == '2019-21']), ('2022', Hh[Hh.era4 == '2022']), ('2023-26', Hh[Hh.era4 == '2023-26'])):
    a, b = gx[gx.RNDhi == 1], gx[gx.RNDhi == 0]; print(f'{lab:8s} RND_hi n={len(a):3d} reach {a.hit.mean():.3f} mean r {a.r.mean():+.3f} med {a.r.median():+.3f} | RND_lo n={len(b):3d} reach {b.hit.mean():.3f} mean r {b.r.mean():+.3f} med {b.r.median():+.3f}')
