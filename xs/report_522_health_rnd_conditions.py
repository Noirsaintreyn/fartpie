"""Report 522 (Nov 27 2026): Rey's second attachment (Consensus literature summary): R&D-intensity premium is strongest in healthcare and is CONDITIONAL on firm traits: size (small), value (book-to-market), ownership concentration (Bhattacharjee & Chatterjee 2020), reinvestment effectiveness / runway (Milano et al. 2017). Declared: within HEALTH, the RND_hi-vs-RND_lo lift (within ATRP decile x price) should be LARGER in: small market cap, high book-to-market, concentrated ownership (13D present or >=2 new 13G filers in the last 365d), cash runway > 12 months. Template as 516-521, eras, month-clustered t. Literature text is a secondary AI summary, used only to pick conditioning variables."""
import numpy as np, pandas as pd, json, io, contextlib, warnings; warnings.filterwarnings('ignore')
src = open('xs/report_516_sector.py').read().split("TRm = (P.yr")[0]
with contextlib.redirect_stdout(io.StringIO()): exec(src)
TRm = (P.yr >= 2019) & (P.yr <= 2021); P['pb'] = np.select([P.c < 1, P.c < 2.5], [0, 1], 2); P['era4'] = np.select([P.yr == 2018, P.yr <= 2021, P.yr == 2022], ['2018', '2019-21', '2022'], '2023-26')
G = P[P.sector == 'HEALTH'].copy(); G['ab10'] = pd.qcut(G.ATRP.rank(method='first'), 10, labels=False)
sh = pd.read_pickle(X + 'shares_pit.pkl'); sh['filed'] = pd.to_datetime(sh.filed).astype('datetime64[ns]'); sh = sh.sort_values('filed'); G['d'] = G.d.astype('datetime64[ns]'); G = pd.merge_asof(G.sort_values('d'), sh[['tk', 'filed', 'val']].rename(columns={'filed': 'fs', 'val': 'SH'}), left_on='d', right_on='fs', by='tk', direction='backward', tolerance=pd.Timedelta(days=730)); G['MCAP'] = G.SH * G.c
eq = []
for tk in G.tk.unique():
    try: gg = json.load(open(f'data/fund/sec_small/{tk}.json'))['facts']['us-gaap']
    except Exception: continue
    a = ann(gg, ['StockholdersEquity'], 1); ca = ann(gg, ['CashAndCashEquivalentsAtCarryingValue'], 1)
    if a is None: continue
    e = pd.DataFrame({'EQ': a.val, 'filed': a.filed}); e['CASHQ'] = ca.val.reindex(e.index) if ca is not None else np.nan; e['tk'] = tk; eq.append(e.dropna(subset=['filed']))
EQ = pd.concat(eq).sort_values('filed'); EQ['filed'] = EQ.filed.astype('datetime64[ns]'); G = pd.merge_asof(G.sort_values('d'), EQ.rename(columns={'filed': 'fe'}), left_on='d', right_on='fe', by='tk', direction='backward', tolerance=pd.Timedelta(days=540))
G['BM'] = G.EQ / G.MCAP; G.loc[G.BM <= 0, 'BM'] = np.nan
A = pd.read_csv(X + '13d_wide.csv'); A['filed'] = pd.to_datetime(A.filed).astype('datetime64[ns]'); f = A.form.str.upper(); A = A[f.str.contains('13D|13G')]; A = A.assign(isD=f.loc[A.index].str.contains('13D'))
gA = {t: (np.sort(x.filed.values), x.isD.values[np.argsort(x.filed.values)]) for t, x in A.groupby('ticker')}; conc = np.full(len(G), np.nan); idx = pd.Series(range(len(G))).groupby(G.tk.values).indices
for tk, ii in idx.items():
    if tk not in gA: continue
    a, isd = gA[tk]; d = G.d.values[ii]; lo = np.searchsorted(a, d - np.timedelta64(365, 'D'), side='left'); hi = np.searchsorted(a, d, side='left'); cs = np.concatenate([[0], np.cumsum(isd)]); nD = cs[hi] - cs[lo]; n = hi - lo; conc[ii] = ((nD >= 1) | (n >= 4)).astype(float)
G['CONC'] = conc
x = G.loc[TRm.reindex(G.index).fillna(False), 'RND'].dropna(); q = x.quantile([.33, .67]); G['RNDhi'] = np.where(G.RND > q.iloc[1], 1, np.where(G.RND <= q.iloc[0], 0, np.nan))
def tert2(c, hi_label):
    x = G.loc[TRm.reindex(G.index).fillna(False), c].dropna(); med = x.median(); return np.where(G[c] > med, 1, np.where(G[c].notna(), 0, np.nan))
G['BIG'] = tert2('MCAP', ''); G['HIBM'] = tert2('BM', ''); G['RUN12'] = np.where(G.CASHQ.notna() & G.RND.notna(), (G.CASHQ > 0).astype(float), np.nan)
def lift2(Gx):
    d = Gx[Gx.RNDhi.isin([0, 1])]; r = []
    for (m, b, p), gp in d.groupby(['mon', 'ab10', 'pb']):
        a, c = gp[gp.RNDhi == 1].y, gp[gp.RNDhi == 0].y
        if len(a) >= 3 and len(c) >= 3: r.append((m, a.mean() - c.mean(), len(a)))
    if len(r) < 4: return np.nan, np.nan
    o = pd.DataFrame(r, columns=['m', 'x', 'n']); mm = o.groupby('m').apply(lambda z: np.average(z.x, weights=z.n)); s = mm.std(); return mm.mean(), (mm.mean() / (s / np.sqrt(len(mm))) if s and s > 0 else np.nan)
print('HEALTH stock-days', len(G), '| coverage MCAP', round(G.MCAP.notna().mean(), 2), 'BM', round(G.BM.notna().mean(), 2), 'CONC', round(G.CONC.notna().mean(), 2), '| CONC=1 share', round(G.CONC.mean(), 2))
pd.set_option('display.width', 200); rows = []
for lab, m in (('ALL', G.RNDhi.notna()), ('small cap (below train median)', G.BIG == 0), ('large cap (above)', G.BIG == 1), ('high book-to-market', G.HIBM == 1), ('low book-to-market', G.HIBM == 0), ('concentrated holders (13D or >=4 13D/G in 365d)', G.CONC == 1), ('dispersed holders', G.CONC == 0)):
    Gx = G[m]; r = dict(group=lab, n=len(Gx))
    for e in ('2018', '2019-21', '2022', '2023-26'): d, t = lift2(Gx[Gx.era4 == e]); r[e] = None if d != d else round(d * 100, 1)
    d, t = lift2(Gx); r['pooled_pp'] = None if d != d else round(d * 100, 1); r['t'] = None if t != t else round(t, 1); rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
