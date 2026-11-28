"""Report 521 (Nov 27 2026): Rey's attached sector 'structural intensity' framework applied per sector (hypotheses taken from the document, declared before results).
 TECH/software: S&M intensity (S&M/revenue), R&D intensity (R&D/revenue), Rule of 40 (rev growth + FCF margin), gross margin level and its change, operating-leverage inflection (revenue growth > opex growth)
 INDUSTRIALS/defense (SIC 15-17, 30-34, 37, 39, 40-47): capex intensity (capex/revenue) and its DROP after a build-out, gross margin, revenue growth, operating leverage
 SEMIS/hardware (SIC 36, 35, 38-ish inside TECH): R&D intensity x gross margin (R&D/revenue high AND gross margin > 55%)
 FIN: ROTCE (net income / tangible common equity), tangible equity / assets, tangible book growth
 CONSUMER (SIC 20-23, 52-59, 70-72, 79): advertising/revenue, gross margin, revenue growth
 HEALTH: R&D intensity on revenue where revenue exists, OCF positive
 Composite 'accounting-distortion' screen (doc step 1-3): high intensity (R&D or S&M > 15% of revenue) AND GAAP loss (EBIT<0) AND (gross margin up OR OCF>0) AND cash runway > 12 months  -> tested in every sector.
Caveat declared: the document describes $300M-$2B compounders; our universe is sub-$5 lottery names, so the mechanism (mispricing of intangible investment) may not map to reach to +100%.
Template: within-ATRP-decile x price month-clustered lift, eras 2018 / 2019-21 / 2022 / 2023-26, top third vs bottom third (train quantiles) or boolean; PASS pooled |t|>=3 and sign agrees >=3 of 4 eras, lead |t| 2-3."""
import numpy as np, pandas as pd, json, io, contextlib, warnings; warnings.filterwarnings('ignore')
src = open('xs/report_516_sector.py').read().split("TRm = (P.yr")[0]
with contextlib.redirect_stdout(io.StringIO()): exec(src)
TRm = (P.yr >= 2019) & (P.yr <= 2021); P['pb'] = np.select([P.c < 1, P.c < 2.5], [0, 1], 2); P['era4'] = np.select([P.yr == 2018, P.yr <= 2021, P.yr == 2022], ['2018', '2019-21', '2022'], '2023-26')
S2 = S.set_index('tk').s2
def sec2(r):
    s = S2.get(r); 
    if s != s or s is None: return 'OTHER'
    if s in (28, 80, 87) or s == 38: return 'HEALTH'
    if s in (35, 36, 73, 48): return 'TECH'
    if s in (13, 29): return 'ENERGY'
    if s in (10, 12, 14): return 'MINING'
    if 60 <= s <= 64 or s == 67: return 'FIN'
    if s == 49: return 'UTIL'
    if s in (15, 16, 17, 30, 31, 32, 33, 34, 37, 39, 40, 41, 42, 44, 45, 47): return 'INDUS'
    if s in (20, 21, 22, 23) or 52 <= s <= 59 or s in (70, 72, 79): return 'CONSUMER'
    return 'OTHER'
P['sector'] = P.tk.map({t: sec2(t) for t in P.tk.unique()}); print('sectors:', P.groupby('sector').tk.nunique().to_dict(), '| stock-days', P.sector.value_counts().to_dict())
rows = []
def T(g, tags, inst, extra=None):
    return ann(g, tags, inst)
for tk in sorted(P.tk.unique()):
    try: g = json.load(open(f'data/fund/sec_small/{tk}.json'))['facts']['us-gaap']
    except Exception: continue
    ta = ann(g, ['Assets'], 1)
    if ta is None: continue
    items = {'REV': (['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'SalesRevenueNet', 'RevenueFromContractWithCustomerIncludingAssessedTax'], 0), 'RD': (['ResearchAndDevelopmentExpense'], 0), 'SM': (['SellingAndMarketingExpense'], 0), 'AD': (['AdvertisingExpense', 'MarketingAndAdvertisingExpense'], 0),
             'GP': (['GrossProfit'], 0), 'OPX': (['OperatingExpenses', 'CostsAndExpenses'], 0), 'OCF': (['NetCashProvidedByUsedInOperatingActivities'], 0), 'CX': (['PaymentsToAcquirePropertyPlantAndEquipment', 'PaymentsToAcquireProductiveAssets'], 0), 'NI': (['NetIncomeLoss', 'ProfitLoss'], 0),
             'GW': (['Goodwill'], 1), 'IA': (['IntangibleAssetsNetExcludingGoodwill'], 1), 'EQ': (['StockholdersEquity'], 1), 'EBIT': (['OperatingIncomeLoss'], 0), 'CASH': (['CashAndCashEquivalentsAtCarryingValue'], 1)}
    F = pd.DataFrame({'TA': ta.val, 'filed': ta.filed})
    for k, (tags, inst) in items.items():
        a = ann(g, tags, inst); F[k] = a.val.reindex(F.index) if a is not None else np.nan
    F = F[F.TA > 0].dropna(subset=['filed'])
    if not len(F): continue
    R = F.REV.where(F.REV > 0); F['RDI'] = F.RD / R; F['SMI'] = F.SM / R; F['ADI'] = F.AD / R; F['GM'] = F.GP / R; F['dGM'] = F.GM.diff(); F['REVG'] = F.REV.where(F.REV > 0).pct_change(); F['FCFM'] = (F.OCF - F.CX.fillna(0)) / R; F['R40'] = F.REVG + F.FCFM
    F['OPLEV'] = (F.REVG - F.OPX.pct_change()).where(F.REVG.notna()); F['CAPINT'] = F.CX / R; F['dCAPINT'] = F.CAPINT.diff(); F['OCFPOS'] = (F.OCF > 0).astype(float).where(F.OCF.notna())
    tang = F.EQ - F.GW.fillna(0) - F.IA.fillna(0); F['ROTCE'] = np.where(tang > 0, F.NI / tang, np.nan); F['TANGTA'] = tang / F.TA; F['TBVG'] = tang.pct_change().where(tang.shift() > 0)
    burn = -F.OCF.where(F.OCF < 0); F['RUNWAY12'] = np.where(F.OCF >= 0, 1.0, np.where(F.CASH / burn * 12 > 12, 1.0, 0.0)); F['RDGM'] = ((F.RDI > .15) & (F.GM > .55)).astype(float).where(F.RDI.notna() & F.GM.notna())
    F['COMP'] = (((F.RDI > .15) | (F.SMI > .15)) & (F.EBIT < 0) & (((F.dGM > 0)) | (F.OCF > 0)) & (F.RUNWAY12 == 1)).astype(float).where((F.RDI.notna() | F.SMI.notna()) & F.EBIT.notna()); F['tk'] = tk
    rows.append(F[['tk', 'filed', 'RDI', 'SMI', 'ADI', 'GM', 'dGM', 'REVG', 'R40', 'OPLEV', 'CAPINT', 'dCAPINT', 'OCFPOS', 'ROTCE', 'TANGTA', 'TBVG', 'RDGM', 'COMP']])
XF = pd.concat(rows).sort_values('filed'); XF['filed'] = XF.filed.astype('datetime64[ns]'); P['d'] = P.d.astype('datetime64[ns]'); P = P.drop(columns=[c for c in XF.columns if c in P.columns and c not in ('tk',)], errors='ignore'); P = pd.merge_asof(P.sort_values('d'), XF.rename(columns={'filed': 'fx'}), left_on='d', right_on='fx', by='tk', direction='backward', tolerance=pd.Timedelta(days=540))
for c in ('RDI', 'SMI', 'GM', 'R40', 'CAPINT', 'ROTCE', 'COMP'): print(f'coverage {c}: {P[c].notna().mean():.2f}', end=' | ')
print()
def lift2(G, f):
    d = G.assign(f=f.loc[G.index]); d = d[d.f.isin([0, 1])]; r = []
    for (m, b, p), gp in d.groupby(['mon', 'ab10', 'pb']):
        a, c = gp[gp.f == 1].y, gp[gp.f == 0].y
        if len(a) >= 3 and len(c) >= 3: r.append((m, a.mean() - c.mean(), len(a)))
    if len(r) < 4: return np.nan, np.nan
    o = pd.DataFrame(r, columns=['m', 'x', 'n']); mm = o.groupby('m').apply(lambda z: np.average(z.x, weights=z.n)); s = mm.std(); return mm.mean(), (mm.mean() / (s / np.sqrt(len(mm))) if s and s > 0 else np.nan)
def tert(G, c):
    x = G.loc[TRm.reindex(G.index).fillna(False), c].dropna()
    if len(x) < 300: return pd.Series(np.nan, index=G.index)
    q = x.quantile([.33, .67]); return pd.Series(np.where(G[c] > q.iloc[1], 1, np.where(G[c] <= q.iloc[0], 0, np.nan)), index=G.index)
FE = {'TECH': ['SMI', 'RDI', 'R40', 'GM', 'dGM', 'OPLEV', 'RDGM', 'COMP'], 'INDUS': ['CAPINT', 'dCAPINT', 'GM', 'OPLEV', 'REVG', 'COMP'], 'FIN': ['ROTCE', 'TANGTA', 'TBVG'], 'CONSUMER': ['ADI', 'GM', 'REVG', 'COMP'], 'HEALTH': ['RDI', 'OCFPOS', 'dGM', 'COMP'], 'ENERGY': ['COMP', 'OCFPOS', 'REVG'], 'MINING': ['COMP', 'OCFPOS']}; BOOL = {'OCFPOS', 'RDGM', 'COMP'}; OUT = []
for sc, cols in FE.items():
    G = P[P.sector == sc].copy()
    if len(G) < 3000: print(sc, 'too small'); continue
    G['ab10'] = pd.qcut(G.ATRP.rank(method='first'), 10, labels=False)
    for c in cols:
        f = G[c].where(G[c].isin([0, 1])) if c in BOOL else tert(G, c); r = dict(sector=sc, feature=c + ('' if c in BOOL else '_hi'), n1=int((f == 1).sum()), n0=int((f == 0).sum()))
        for e in ('2018', '2019-21', '2022', '2023-26'): r[e] = lift2(G[G.era4 == e], f)[0]
        d, t = lift2(G, f); r['pooled'] = d; r['t'] = t; ev = [r[e] for e in ('2018', '2019-21', '2022', '2023-26') if r[e] == r[e]]; r['agree'] = max(sum(x > 0 for x in ev), sum(x < 0 for x in ev))
        r['verdict'] = 'PASS' if (abs(t) >= 3 and r['agree'] >= 3) else ('lead' if (abs(t) >= 2 and r['agree'] >= 3) else '-'); OUT.append(r)
R = pd.DataFrame(OUT); R.to_csv('xs/report_521_intensity_framework.csv', index=False); F2 = R.copy()
for c in ('2018', '2019-21', '2022', '2023-26', 'pooled'): F2[c] = (F2[c] * 100).round(1)
F2['t'] = F2.t.round(1); pd.set_option('display.width', 220); print(F2.to_string(index=False))
