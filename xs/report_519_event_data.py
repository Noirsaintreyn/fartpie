"""Report 519 (Nov 27 2026): diversification beyond health. Rey: "try all, health is cool but it's not a lot of trades, we want it well diversified". Event/exposure data available to us per sector, tested with the 518 template (within-ATRP-decile x price month-clustered lift, eras, strict bar: pooled |t|>=3 and sign agrees >=3 of 4 eras = PASS, |t| 2-3 = lead).
Features (declared before results), all known before day d:
  D13_new   new Schedule 13D (activist / >5% holder with intent) filed in prior 30d        D13_amend  13D/A in prior 30d        G13_new  new 13G (passive 5% holder, institutions) in prior 30d
  PRE_REV   latest 10-K shows no/zero revenue (pre-production explorer / pre-revenue)       RPO_hi  backlog: remaining performance obligation / assets, top third (only names that report it)
  REVACC    revenue growth accelerating (rev growth this year > last year)
Sectors: all (HEALTH, TECH, ENERGY, MINING, FIN, OTHER); UTIL skipped (n)."""
import numpy as np, pandas as pd, json, io, contextlib, warnings; warnings.filterwarnings('ignore')
src = open('xs/report_516_sector.py').read().split("TRm = (P.yr")[0]
with contextlib.redirect_stdout(io.StringIO()): exec(src)
TRm = (P.yr >= 2019) & (P.yr <= 2021); P['pb'] = np.select([P.c < 1, P.c < 2.5], [0, 1], 2); P['era4'] = np.select([P.yr == 2018, P.yr <= 2021, P.yr == 2022], ['2018', '2019-21', '2022'], '2023-26')
A = pd.read_csv(X + '13d_wide.csv'); A['filed'] = pd.to_datetime(A.filed).astype('datetime64[ns]'); f = A.form.str.upper()
A['K'] = np.select([f.isin(['SC 13D', 'SCHEDULE 13D']), f.isin(['SC 13D/A', 'SCHEDULE 13D/A']), f.isin(['SC 13G', 'SCHEDULE 13G'])], ['D13_new', 'D13_amend', 'G13_new'], '')
def win(kind, back=30):
    g = {t: np.sort(x.filed.values) for t, x in A[A.K == kind].groupby('ticker')}; out = np.zeros(len(P), bool); idx = pd.Series(range(len(P))).groupby(P.tk.values).indices
    for tk, ii in idx.items():
        a = g.get(tk)
        if a is None: continue
        d = P.d.values[ii]; out[ii] = np.searchsorted(a, d, side='left') > np.searchsorted(a, d - np.timedelta64(back, 'D'), side='left')
    return out.astype(int)
covA = P.tk.isin(set(A.ticker))
for k in ('D13_new', 'D13_amend', 'G13_new'): P[k] = win(k); P.loc[~covA, k] = np.nan
# XBRL: pre-revenue flag, RPO/assets, revenue acceleration
rows = []
for tk in sorted(P.tk.unique()):
    try: g = json.load(open(f'data/fund/sec_small/{tk}.json'))['facts']['us-gaap']
    except Exception: continue
    ta = ann(g, ['Assets'], 1)
    if ta is None: continue
    rv = ann(g, ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'SalesRevenueNet', 'RevenueFromContractWithCustomerIncludingAssessedTax'], 0); rp = ann(g, ['RevenueRemainingPerformanceObligation'], 1)
    F = pd.concat({'TA': ta.val, 'filed': ta.filed}, axis=1); F['REV'] = rv.val.reindex(F.index) if rv is not None else np.nan; F['RPO'] = rp.val.reindex(F.index) if rp is not None else np.nan; F = F[F.TA > 0].dropna(subset=['filed'])
    F['PRE_REV'] = ((F.REV.isna()) | (F.REV <= 0)).astype(float); F['RPO_A'] = F.RPO / F.TA; gr = F.REV.pct_change(); F['REVACC'] = (gr > gr.shift(1)).astype(float).where(gr.notna() & gr.shift(1).notna()); F['tk'] = tk; rows.append(F[['tk', 'filed', 'PRE_REV', 'RPO_A', 'REVACC']])
XF = pd.concat(rows).sort_values('filed'); XF['filed'] = XF.filed.astype('datetime64[ns]'); P = pd.merge_asof(P.sort_values('d'), XF.rename(columns={'filed': 'fx'}), left_on='d', right_on='fx', by='tk', direction='backward', tolerance=pd.Timedelta(days=540))
print('coverage 13D/G', round(covA.mean(), 2), '| XBRL', round(P.PRE_REV.notna().mean(), 2), '| RPO', round(P.RPO_A.notna().mean(), 2), '| 13D_new share', round(P.D13_new.mean(), 3), '13G_new', round(P.G13_new.mean(), 3))
def lift2(G, flag):
    d = G.assign(f=flag.loc[G.index]); d = d[d.f.isin([0, 1])]; r = []
    for (m, b, p), gp in d.groupby(['mon', 'ab10', 'pb']):
        a, c = gp[gp.f == 1].y, gp[gp.f == 0].y
        if len(a) >= 3 and len(c) >= 3: r.append((m, a.mean() - c.mean(), len(a)))
    if len(r) < 4: return np.nan, np.nan
    o = pd.DataFrame(r, columns=['m', 'x', 'n']); mm = o.groupby('m').apply(lambda z: np.average(z.x, weights=z.n)); s = mm.std(); return mm.mean(), (mm.mean() / (s / np.sqrt(len(mm))) if s and s > 0 else np.nan)
OUT = []
for sc, G in P.groupby('sector'):
    if len(G) < 4000 or sc == 'UTIL': continue
    G = G.copy(); G['ab10'] = pd.qcut(G.ATRP.rank(method='first'), 10, labels=False); fe = {k: G[k] for k in ('D13_new', 'D13_amend', 'G13_new', 'PRE_REV', 'REVACC')}
    if G.RPO_A.notna().sum() > 2000:
        q = G.loc[TRm.reindex(G.index).fillna(False), 'RPO_A'].quantile([.33, .67]); fe['RPO_hi'] = pd.Series(np.where(G.RPO_A > q.iloc[1], 1, np.where(G.RPO_A <= q.iloc[0], 0, np.nan)), index=G.index) if q.notna().all() else pd.Series(np.nan, index=G.index)
    for k, fl in fe.items():
        r = dict(sector=sc, feature=k, n1=int((fl == 1).sum()), n0=int((fl == 0).sum()))
        for e in ('2018', '2019-21', '2022', '2023-26'): r[e] = lift2(G[G.era4 == e], fl)[0]
        d, t = lift2(G, fl); r['pooled'] = d; r['t'] = t; ev = [r[e] for e in ('2018', '2019-21', '2022', '2023-26') if r[e] == r[e]]; r['agree'] = max(sum(x > 0 for x in ev), sum(x < 0 for x in ev))
        r['verdict'] = 'PASS' if (abs(t) >= 3 and r['agree'] >= 3) else ('lead' if (abs(t) >= 2 and r['agree'] >= 3) else '-'); OUT.append(r)
R = pd.DataFrame(OUT); R.to_csv('xs/report_519_event_data.csv', index=False); F2 = R.copy()
for c in ('2018', '2019-21', '2022', '2023-26', 'pooled'): F2[c] = (F2[c] * 100).round(1)
F2['t'] = F2.t.round(1); pd.set_option('display.width', 220); print(F2.to_string(index=False))
