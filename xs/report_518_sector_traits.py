"""Report 518 (Nov 27 2026): the HEALTH recipe generalised. Rey: "this is how we trace health, this gives us an idea how to look at other sectors".
RECIPE (declared): (1) the sector's trigger = what the news/headlines say moves it; (2) an EXPOSURE trait that measures how much a name is exposed to that trigger (HEALTH: R&D/assets = pipeline of binary readouts); (3) lift within ATRP-decile x price bucket, month-clustered, eras; (4) 'stale news' filter; (5) flips.
Exposure traits per sector (declared before results):
  HEALTH  R&D/assets (done, 516)                                  TECH   R&D/assets, revenue growth, capex/assets (build-out exposure: AI/data-centre theme)
  ENERGY  oil beta (60d beta of the stock to WTI-proxy XLE), capex/assets (drilling/exploration spend)       MINING  XLB beta, capex/assets (exploration/development spend)
  FIN     XLF beta, leverage                                       UTIL/OTHER skipped (n)
  ALL     SECTOR BETA = rolling 60d beta of daily return to the sector ETF (XLV, XLK, XLE, XLB, XLF), lagged 1 day.
Macro x exposure: within the HIGH-beta third, does the sector driver (ETF 20d rel-SPY up; oil up for ENERGY; USD down for MINING) lift P(double)? The macro driver should only matter for names exposed to it (516 showed it does not pooled).
Same within-ATRP-decile x price month-clustered lift; PASS pooled |t|>=3 and sign agrees >=3 of 4 eras; lead |t| 2-3."""
import numpy as np, pandas as pd, json, io, contextlib, warnings; warnings.filterwarnings('ignore')
src = open('xs/report_516_sector.py').read().split("TRm = (P.yr")[0]
with contextlib.redirect_stdout(io.StringIO()): exec(src)
TRm = (P.yr >= 2019) & (P.yr <= 2021); P['pb'] = np.select([P.c < 1, P.c < 2.5], [0, 1], 2); P['era4'] = np.select([P.yr == 2018, P.yr <= 2021, P.yr == 2022], ['2018', '2019-21', '2022'], '2023-26')
D = pd.read_pickle(X + 'wide_daily_ohlcv.pkl'); ETF = {'HEALTH': 'XLV', 'TECH': 'XLK', 'ENERGY': 'XLE', 'MINING': 'XLB', 'FIN': 'XLF'}; er = mp[list(ETF.values())].pct_change()
# ---- sector beta (60d, lagged 1d) per ticker
B = []
for tk, sc in P.groupby('tk').sector.first().items():
    if sc not in ETF or tk not in D: continue
    c = D[tk].close.dropna(); c.index = pd.DatetimeIndex(c.index).astype('datetime64[ns]'); r = c.pct_change().clip(-.5, 1.5); e = er[ETF[sc]].reindex(c.index).ffill(); cov = r.rolling(60, min_periods=40).cov(e); var = e.rolling(60, min_periods=40).var()
    B.append(pd.DataFrame({'tk': tk, 'd': c.index, 'SBETA': (cov / var).shift(1).values}))
B = pd.concat(B); P = P.merge(B, on=['tk', 'd'], how='left'); print('beta coverage', round(P.SBETA.notna().mean(), 2))
# ---- capex / assets (PIT 10-K)
rows = []
for tk in sorted(P[P.sector.isin(['TECH', 'ENERGY', 'MINING'])].tk.unique()):
    try: g = json.load(open(f'data/fund/sec_small/{tk}.json'))['facts']['us-gaap']
    except Exception: continue
    a = ann(g, ['Assets'], 1); cx = ann(g, ['PaymentsToAcquirePropertyPlantAndEquipment', 'PaymentsToAcquireProductiveAssets', 'PaymentsToExploreAndDevelopOilAndGasProperties', 'PaymentsToAcquireOilAndGasProperty'], 0)
    if a is None or cx is None: continue
    F = pd.concat({'TA': a.val, 'CX': cx.val, 'filed': a.filed}, axis=1).dropna(); F = F[F.TA > 0]; F['CAPEX'] = F.CX / F.TA; F['tk'] = tk; rows.append(F[['tk', 'filed', 'CAPEX']])
CX = pd.concat(rows).sort_values('filed'); CX['filed'] = CX.filed.astype('datetime64[ns]'); P = pd.merge_asof(P.sort_values('d'), CX.rename(columns={'filed': 'filed_cx'}), left_on='d', right_on='filed_cx', by='tk', direction='backward', tolerance=pd.Timedelta(days=540))
def lift2(G, flag):
    d = G.assign(f=flag.loc[G.index]); d = d[d.f.isin([0, 1])]; r = []
    for (m, b, p), gp in d.groupby(['mon', 'ab10', 'pb']):
        a, c = gp[gp.f == 1].y, gp[gp.f == 0].y
        if len(a) >= 3 and len(c) >= 3: r.append((m, a.mean() - c.mean(), len(a)))
    if len(r) < 4: return np.nan, np.nan
    o = pd.DataFrame(r, columns=['m', 'x', 'n']); mm = o.groupby('m').apply(lambda z: np.average(z.x, weights=z.n)); s = mm.std(); return mm.mean(), (mm.mean() / (s / np.sqrt(len(mm))) if s and s > 0 else np.nan)
def tert(G, c):
    q = G.loc[TRm.reindex(G.index).fillna(False), c].quantile([.33, .67]); return pd.Series(np.where(G[c] > q.iloc[1], 1, np.where(G[c] <= q.iloc[0], 0, np.nan)), index=G.index) if q.notna().all() else pd.Series(np.nan, index=G.index)
TR = {'HEALTH': ['RND', 'SBETA'], 'TECH': ['RND', 'REVG', 'CAPEX', 'SBETA'], 'ENERGY': ['SBETA', 'CAPEX'], 'MINING': ['SBETA', 'CAPEX'], 'FIN': ['SBETA', 'LEV']}; OUT = []
def add(sc, name, G, f):
    r = dict(sector=sc, trait=name, n1=int((f == 1).sum()), n0=int((f == 0).sum()))
    for e in ('2018', '2019-21', '2022', '2023-26'): r[e] = lift2(G[G.era4 == e], f)[0]
    d, t = lift2(G, f); r['pooled'] = d; r['t'] = t; ev = [r[e] for e in ('2018', '2019-21', '2022', '2023-26') if r[e] == r[e]]; r['agree'] = max(sum(x > 0 for x in ev), sum(x < 0 for x in ev))
    r['verdict'] = 'PASS' if (abs(t) >= 3 and r['agree'] >= 3) else ('lead' if (abs(t) >= 2 and r['agree'] >= 3) else '-'); OUT.append(r)
for sc, cols in TR.items():
    G = P[P.sector == sc].copy(); G['ab10'] = pd.qcut(G.ATRP.rank(method='first'), 10, labels=False)
    for c in cols: add(sc, c + '_hi', G, tert(G, c))
    # macro driver within the exposed (high sector-beta) third
    hb = tert(G, 'SBETA'); G2 = G[hb.loc[G.index] == 1]
    drv = {'HEALTH': ['XLV_rel'], 'TECH': ['XLK_rel'], 'ENERGY': ['oil_up', 'XLE_rel'], 'MINING': ['usd_dn', 'XLB_rel', 'real_dn'], 'FIN': ['XLF_rel', 'steep']}[sc]
    for c in drv: add(sc, f'{c} | high-beta names', G2, G2[c].where(G2[c].isin([0, 1])))
    LOb = G[hb.loc[G.index] == 0]
    for c in drv: add(sc, f'{c} | low-beta names', LOb, LOb[c].where(LOb[c].isin([0, 1])))
R = pd.DataFrame(OUT); R.to_csv('xs/report_518_sector_traits.csv', index=False); F2 = R.copy()
for c in ('2018', '2019-21', '2022', '2023-26', 'pooled'): F2[c] = (F2[c] * 100).round(1)
F2['t'] = F2.t.round(1); pd.set_option('display.width', 220); print(F2.to_string(index=False))
