"""Report 516 (Nov 27 2026): per-SECTOR drivers of 100%+ runs in sub-$5 stocks. Rey: "small caps move more in their sectors, do the different fundamentals and macro research per sector".
DECLARED BEFORE RESULTS. Same unit/label as 515 (sub-$5 stock-day, y = doubles within 40d; panel = xs 515 panel2). Sector = SIC (ext/sic.csv) -> HEALTH (28,80,38,87), TECH (35,36,73,48,37), ENERGY (13,29), MINING (10,12,14,33), FIN (60-64,67), UTIL (49), OTHER.
Mechanism per sector (why this driver should matter for THAT sector):
  ENERGY  oil up (WTI 20d), XLE beats SPY          -> revenue/reserve value up, levered equity is a call on the commodity
  MINING  USD down, real yield (DFII10) down, XLB rel -> metals priced in USD / opportunity cost of holding metal
  FIN     curve steepening, HY OAS tightening, XLF rel -> NIM and credit spreads
  HEALTH  XLV rel, IWM rel, 10y yield down          -> biotech = long-duration, financing-dependent
  TECH    XLK rel, 10y yield down, IWM rel          -> growth duration
  UTIL    10y yield down, XLU rel                   -> bond proxy / power-demand theme
  ALL     IWM rel SPY (small-cap risk appetite), HY OAS tightening
  (all 20d changes, shifted 1 day = point-in-time)
Contagion: SECWAVE = leave-one-out share of the other same-sector sub-$5 names with r20>+30% that day (>=6 names). Hidden-state check: do episode start dates cluster within sector more than a within-sector date shuffle?
Fundamentals (PIT latest 10-K filed <=540d before d, SEC companyfacts): Z'' (Altman), REVG (revenue growth y/y), CASHTA (cash/assets), RUNWAY (months of cash at the latest operating cash burn, burners only), LEV (liabilities/assets), MARGIN (EBIT/revenue), RND (R&D/ assets for HEALTH/TECH).
Flag = top third (1) vs bottom third (0) by SECTOR train (2019-21) quantiles. Metric = within-ATRP-quintile lift in P(y), month-clustered t. Eras 2018 / 2019-21 / 2022 / 2023-26.
PASS (declared, ~60 tests so strict): sign agrees in >=3 of 4 eras AND pooled |t|>=3 ; LEAD if pooled |t| 2-3 and sign agrees >=3 of 4."""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; X = SP + 'ext/'
P = pd.read_pickle(X + 'runs515_panel2.pkl'); P = P[['d', 'tk', 'c', 'ATRP', 'r20', 'y', 'yr', 'era', 'mon']].copy(); P['ab'] = pd.qcut(P.ATRP.rank(method='first'), 5, labels=False)
S = pd.read_csv(X + 'sic.csv'); S['s2'] = pd.to_numeric(S.sic.astype(str).str[:2], errors='coerce')
def sec(s):
    if s in (28, 80, 38, 87): return 'HEALTH'
    if s in (35, 36, 73, 48, 37): return 'TECH'
    if s in (13, 29): return 'ENERGY'
    if s in (10, 12, 14, 33): return 'MINING'
    if 60 <= s <= 64 or s == 67: return 'FIN'
    if s == 49: return 'UTIL'
    return 'OTHER'
S['sector'] = S.s2.apply(lambda s: sec(s) if s == s else 'OTHER'); P = P.merge(S[['tk', 'sector']], on='tk', how='left'); P['sector'] = P.sector.fillna('OTHER'); P['up'] = (P.r20 > .3).astype(float)
print('panel', len(P), '| stock-days by sector:', P.sector.value_counts().to_dict(), '| names:', P.groupby('sector').tk.nunique().to_dict())
# ---- sector wave (leave-one-out)
g = P.groupby(['sector', 'd']).up; n = g.transform('size'); s = g.transform('sum'); P['SECWAVE'] = np.where(n >= 6, (s - P.up) / (n - 1), np.nan)
# ---- macro drivers (PIT)
fr = pd.read_parquet(SP + 'fred.parquet'); fr.index = pd.to_datetime(fr.index).astype('datetime64[ns]'); fr = fr.ffill(); mp = pd.read_parquet(SP + 'macro_px.parquet'); mp.index = pd.to_datetime(mp.index).astype('datetime64[ns]'); mp = mp.ffill()
rel = lambda e: (mp[e].pct_change(20) - mp.SPY.pct_change(20))
Dm = pd.DataFrame({'oil_up': (fr.DCOILWTICO.pct_change(20) > 0), 'XLE_rel': rel('XLE') > 0, 'usd_dn': fr.DTWEXBGS.pct_change(20) < 0, 'real_dn': (fr.DFII10 - fr.DFII10.shift(20)) < 0, 'XLB_rel': rel('XLB') > 0,
                   'steep': (fr.T10Y2Y - fr.T10Y2Y.shift(20)) > 0, 'hy_tight': (fr.BAMLH0A0HYM2 - fr.BAMLH0A0HYM2.shift(20)) < 0, 'XLF_rel': rel('XLF') > 0, 'XLV_rel': rel('XLV') > 0, 'IWM_rel': rel('IWM') > 0,
                   'y_dn': (fr.DGS10 - fr.DGS10.shift(20)) < 0, 'XLK_rel': rel('XLK') > 0, 'XLU_rel': rel('XLU') > 0}).astype(float).shift(1)
Dm = Dm.reindex(Dm.index.union(P.d.unique())).ffill().reindex(P.d.values);
for c in Dm: P[c] = Dm[c].values
DRV = {'ENERGY': ['oil_up', 'XLE_rel'], 'MINING': ['usd_dn', 'real_dn', 'XLB_rel'], 'FIN': ['steep', 'hy_tight', 'XLF_rel'], 'HEALTH': ['XLV_rel', 'IWM_rel', 'y_dn'], 'TECH': ['XLK_rel', 'y_dn', 'IWM_rel'], 'UTIL': ['y_dn', 'XLU_rel'], 'OTHER': ['IWM_rel', 'hy_tight', 'y_dn']}
# ---- fundamentals (PIT 10-K)
def ann(g, tags, instant):
    for t in tags:
        x = g.get(t)
        if x and 'USD' in x.get('units', {}):
            r = pd.DataFrame(x['units']['USD']); r = r[(r.form == '10-K') & (r.fp == 'FY')]
            if not instant and 'start' in r: r = r[(pd.to_datetime(r.end) - pd.to_datetime(r.start)).dt.days > 300]
            if len(r): r['filed'] = pd.to_datetime(r.filed); return r.sort_values('filed').drop_duplicates('end', keep='last').set_index('end')[['val', 'filed']]
    return None
rows = []
for tk in sorted(P.tk.unique()):
    try: gg = json.load(open(f'data/fund/sec_small/{tk}.json'))['facts']['us-gaap']
    except Exception: continue
    T = {'TA': ann(gg, ['Assets'], 1), 'TL': ann(gg, ['Liabilities'], 1), 'CA': ann(gg, ['AssetsCurrent'], 1), 'CL': ann(gg, ['LiabilitiesCurrent'], 1), 'RE': ann(gg, ['RetainedEarningsAccumulatedDeficit'], 1), 'EQ': ann(gg, ['StockholdersEquity'], 1),
         'EBIT': ann(gg, ['OperatingIncomeLoss'], 0), 'CASH': ann(gg, ['CashAndCashEquivalentsAtCarryingValue', 'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents'], 1), 'REV': ann(gg, ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'SalesRevenueNet'], 0),
         'OCF': ann(gg, ['NetCashProvidedByUsedInOperatingActivities'], 0), 'RND': ann(gg, ['ResearchAndDevelopmentExpense'], 0)}
    if T['TA'] is None: continue
    F = pd.concat({k: v.val for k, v in T.items() if v is not None}, axis=1); F['filed'] = T['TA'].filed.reindex(F.index); F = F.dropna(subset=['filed', 'TA']); F = F[F.TA > 0]
    if not len(F): continue
    for k in ('CA', 'CL', 'RE', 'EQ', 'TL', 'EBIT', 'CASH', 'REV', 'OCF', 'RND'):
        if k not in F: F[k] = np.nan
    TLv = F.TL.where(F.TL.notna(), F.TA - F.EQ)
    F['Z'] = 6.56 * (F.CA - F.CL) / F.TA + 3.26 * F.RE / F.TA + 6.72 * F.EBIT / F.TA + 1.05 * F.EQ / TLv; F['REVG'] = F.REV.pct_change(); F['CASHTA'] = F.CASH / F.TA; F['LEV'] = TLv / F.TA; F['MARGIN'] = F.EBIT / F.REV.replace(0, np.nan)
    F['RUNWAY'] = np.where(F.OCF < 0, F.CASH / (-F.OCF) * 12, np.nan); F['RND'] = F.RND / F.TA; F['tk'] = tk; rows.append(F[['tk', 'filed', 'Z', 'REVG', 'CASHTA', 'LEV', 'MARGIN', 'RUNWAY', 'RND']])
FD = pd.concat(rows).sort_values('filed'); FD['filed'] = FD.filed.astype('datetime64[ns]'); P['d'] = P.d.astype('datetime64[ns]')
P = pd.merge_asof(P.sort_values('d'), FD, left_on='d', right_on='filed', by='tk', direction='backward', tolerance=pd.Timedelta(days=540)); print('fundamental coverage', round(P.Z.notna().mean(), 2), {s: round(P[P.sector == s].Z.notna().mean(), 2) for s in P.sector.unique()})
FUND = {'HEALTH': ['RUNWAY', 'CASHTA', 'RND', 'Z'], 'TECH': ['REVG', 'MARGIN', 'RUNWAY', 'Z'], 'ENERGY': ['LEV', 'REVG', 'Z', 'MARGIN'], 'MINING': ['LEV', 'CASHTA', 'Z'], 'FIN': ['LEV', 'Z', 'REVG'], 'UTIL': ['LEV', 'Z', 'REVG'], 'OTHER': ['REVG', 'MARGIN', 'LEV', 'Z']}
TRm = (P.yr >= 2019) & (P.yr <= 2021); P['era4'] = np.select([P.yr == 2018, P.yr <= 2021, P.yr == 2022], ['2018', '2019-21', '2022', ], '2023-26')
def lift(d, flag):
    d = d.assign(f=flag.loc[d.index]); d = d[d.f.isin([0, 1])]; r = []
    for (m, b), gp in d.groupby(['mon', 'ab']):
        a, c = gp[gp.f == 1].y, gp[gp.f == 0].y
        if len(a) >= 3 and len(c) >= 3: r.append((m, a.mean() - c.mean(), len(a)))
    if len(r) < 4: return np.nan, np.nan, 0
    o = pd.DataFrame(r, columns=['m', 'x', 'n']); mm = o.groupby('m').apply(lambda z: np.average(z.x, weights=z.n)); return mm.mean(), mm.mean() / (mm.std() / np.sqrt(len(mm))), int((d.f == 1).sum())
OUT = []
print('\nSECTOR BASE RATES P(double<=40d) by era:'); print(P.pivot_table(index='sector', columns='era4', values='y', aggfunc='mean').round(3).to_string())
for sc, G in P.groupby('sector'):
    if len(G) < 4000: continue
    feats = {}
    G = G.copy()
    feats['SECWAVE_hi'] = pd.Series(np.where(G.SECWAVE > G[TRm.reindex(G.index).fillna(False)].SECWAVE.quantile(.67), 1, np.where(G.SECWAVE <= G[TRm.reindex(G.index).fillna(False)].SECWAVE.quantile(.33), 0, np.nan)), index=G.index)
    for c in DRV[sc]: feats[c] = G[c].where(G[c].isin([0, 1]))
    for c in FUND[sc]:
        q = G.loc[TRm.reindex(G.index).fillna(False), c].quantile([.33, .67]); feats[c + '_hi'] = pd.Series(np.where(G[c] > q.iloc[1], 1, np.where(G[c] <= q.iloc[0], 0, np.nan)), index=G.index) if q.notna().all() else pd.Series(np.nan, index=G.index)
    for k, f in feats.items():
        r = dict(sector=sc, feature=k, n1=int((f == 1).sum()), n0=int((f == 0).sum()))
        for e in ('2018', '2019-21', '2022', '2023-26'):
            d, t, n = lift(G[G.era4 == e], f); r[e] = d; r[e + '_t'] = t
        d, t, n = lift(G, f); r['pooled'] = d; r['pooled_t'] = t; ev = [r[e] for e in ('2018', '2019-21', '2022', '2023-26') if r[e] == r[e]]
        r['agree'] = max(sum(x > 0 for x in ev), sum(x < 0 for x in ev)); r['verdict'] = 'PASS' if (abs(r['pooled_t']) >= 3 and r['agree'] >= 3) else ('lead' if (abs(r['pooled_t']) >= 2 and r['agree'] >= 3) else '-')
        OUT.append(r)
R = pd.DataFrame(OUT); R.to_csv('xs/report_516_sector_features.csv', index=False)
pd.set_option('display.width', 250); F2 = R.copy();
for c in ('2018', '2019-21', '2022', '2023-26', 'pooled'): F2[c] = (F2[c] * 100).round(1)
for c in ('pooled_t',): F2[c] = F2[c].round(1)
print('\nWITHIN-ATRP LIFT (percentage points of P(double<=40d)), flag=1 minus flag=0, by era, month-clustered pooled t:'); print(F2[['sector', 'feature', 'n1', 'n0', '2018', '2019-21', '2022', '2023-26', 'pooled', 'pooled_t', 'agree', 'verdict']].to_string(index=False))
# ---- contagion check: episode clustering within sector vs within-sector date shuffle
E = pd.read_csv('xs/report_515_episodes.csv'); E = E.merge(S[['tk', 'sector']], on='tk', how='left'); E['sector'] = E.sector.fillna('OTHER'); E['start'] = pd.to_datetime(E.start); rng = np.random.default_rng(0); res = []
Pd = P.groupby('sector').d.apply(lambda x: np.sort(x.unique()))
def nb(dates, tks):
    c = 0
    for i, (d, t) in enumerate(zip(dates, tks)): c += ((np.abs((dates - d) / np.timedelta64(1, 'D')) <= 20) & (tks != t)).sum()
    return c / max(len(dates), 1)
for sc, g in E.groupby('sector'):
    if len(g) < 25: continue
    d = g.start.values.astype('datetime64[ns]'); t = g.tk.values; obs = nb(d, t); pool = Pd[sc]; sim = [nb(rng.choice(pool, len(d)).astype('datetime64[ns]'), t) for _ in range(100)]
    res.append((sc, len(g), round(obs, 2), round(float(np.mean(sim)), 2), round(obs / np.mean(sim), 2), round(float((np.array(sim) >= obs).mean()), 3)))
print('\nCONTAGION: avg # of OTHER same-sector episodes starting within +-20d of an episode (obs vs within-sector date shuffle, 100 sims; p = share of sims >= obs)'); print(pd.DataFrame(res, columns=['sector', 'episodes', 'obs_neighbors', 'shuffle_mean', 'ratio', 'p']).to_string(index=False))
