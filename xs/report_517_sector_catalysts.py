"""Report 517 (Nov 27 2026): per-sector CATALYST profile and per-sector lead tests. Rey: "go through the news, see what most sectors use, assign that to each group, test those leads per sector, then look for flips when they happen".
DECLARED BEFORE RESULTS.
Step 1 (news profile): for every clean 100%+ episode (xs/report_515_episodes, 689) take the 8-K item flags in the window [trigger day -2, trigger day] and compare, PER SECTOR, with the sector's base rate on random sub-$5 stock-days (same 3-day window). Plus the 32 hand-searched headlines (report_515e) tabulated by sector.
Step 2 (lead tests per sector): within-ATRP-decile x price-bucket lift in P(double<=40d), month-clustered t, eras 2018/2019-21/2022/2023-26, for: DEAL 1.01/2.01, PR 7.01/8.01, EARN 2.02, NEG dilution/termination, INS insider open-market buy in prior 30d, E14 earnings 8-K in next 14d, DTC_hi (days to cover top third), plus the 516 fundamentals RND_hi (HEALTH) and MARGIN_hi (TECH). Lead = pooled |t|>=2 and the sign agrees in >=3 of 4 eras; PASS = pooled |t|>=3.
Step 3 (flips): sub-$5 vector flips (flip_history, 1,504): reach / mean / median for flip WITH the sector's lead vs WITHOUT, by era.
"""
import numpy as np, pandas as pd, io, contextlib, warnings; warnings.filterwarnings('ignore')
src = open('xs/report_516_sector.py').read().split("TRm = (P.yr")[0]
with contextlib.redirect_stdout(io.StringIO()): exec(src)
TRm = (P.yr >= 2019) & (P.yr <= 2021); P['pb'] = np.select([P.c < 1, P.c < 2.5], [0, 1], 2); P['era4'] = np.select([P.yr == 2018, P.yr <= 2021, P.yr == 2022], ['2018', '2019-21', '2022'], '2023-26')
# ---- 8-K flags (known by close of day), 7d windows for the panel and a [-2,0] window for trigger days
K = pd.read_csv(X + '8k_wide.csv').dropna(subset=['items']); K['filed'] = pd.to_datetime(K.filed); K['acc'] = pd.to_datetime(K.accept, utc=True, errors='coerce'); K['known'] = pd.to_datetime(np.where(K.acc.dt.hour < 20, K.filed, K.filed + pd.Timedelta(days=1))).astype('datetime64[ns]')
it = K['items'].astype(str).str.split(','); K['DEAL'] = it.apply(lambda l: '1.01' in l or '2.01' in l); K['EARN'] = it.apply(lambda l: '2.02' in l); K['PR'] = it.apply(lambda l: ('7.01' in l or '8.01' in l) and '2.02' not in l); K['NEG'] = it.apply(lambda l: any(x in l for x in ('3.02', '1.02', '1.03', '2.05')))
KG = {t: g for t, g in K.groupby('ticker')}
def flagwin(col, tks, dates, back, fwd=0):
    out = np.zeros(len(tks), bool); idx = pd.Series(range(len(tks))).groupby(np.asarray(tks)).indices
    for tk, ii in idx.items():
        g = KG.get(tk)
        if g is None: continue
        a = np.sort(g.loc[g[col], 'known'].values)
        if not len(a): continue
        d = np.asarray(dates)[ii]; lo = np.searchsorted(a, d - np.timedelta64(back, 'D'), side='left'); hi = np.searchsorted(a, d + np.timedelta64(fwd, 'D'), side='right'); out[ii] = hi > lo
    return out
cov = P.tk.isin(KG.keys())
E = pd.read_csv('xs/report_515_episodes.csv'); E = E[(E.trig_gap < 2) & (E.px >= .05)].merge(S[['tk', 'sector']], on='tk', how='left'); E['sector'] = E.sector.fillna('OTHER'); E['td'] = pd.to_datetime(E.trig_day).astype('datetime64[ns]'); E = E[E.tk.isin(KG.keys())].copy()
print('== STEP 1: 8-K item in [trigger day-2, trigger day]  (share of episodes) vs same window on random sub-$5 stock-days of that sector')
rows = []
for sc in ['HEALTH', 'TECH', 'ENERGY', 'MINING', 'FIN', 'OTHER']:
    e = E[E.sector == sc]; b = P[(P.sector == sc) & cov]; r = dict(sector=sc, n_episodes=len(e))
    for col in ('DEAL', 'PR', 'EARN', 'NEG'):
        eo = flagwin(col, e.tk.values, e.td.values, 2).mean(); bo = flagwin(col, b.tk.values, b.d.values, 2).mean(); r[col] = f'{eo:.2f} vs {bo:.2f} ({eo / bo if bo > 0 else np.nan:.1f}x)'
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
N = pd.read_csv('xs/report_515e_news_sample.csv').merge(S[['tk', 'sector']], on='tk', how='left'); N['sector'] = N.sector.fillna('OTHER'); N['kind'] = N.cause_from_headlines.str.split(':').str[0].str.strip()
print('\n== hand-searched headline causes by sector (32-run sample, headline-level):'); print(pd.crosstab(N.sector, N.kind).to_string())
# ---- STEP 2: per-sector lead tests
P['DEAL7'] = 0; 
for col, nm in (('DEAL', 'DEAL'), ('PR', 'PR'), ('EARN', 'EARN'), ('NEG', 'NEG')): P[nm + '7'] = flagwin(col, P.tk.values, P.d.values, 7).astype(int)
P['E14'] = flagwin('EARN', P.tk.values, P.d.values, -1, 14).astype(int)
I = pd.read_pickle(X + 'insider_P_hist.pkl'); I = I[I.TRANS_CODE == 'P']; gI = {t: np.sort(x.fd.values.astype('datetime64[ns]')) for t, x in I.groupby('ISSUERTRADINGSYMBOL')}
ins = np.zeros(len(P), bool); idx = pd.Series(range(len(P))).groupby(P.tk.values).indices
for tk, ii in idx.items():
    a = gI.get(tk)
    if a is None: continue
    d = P.d.values[ii]; lo = np.searchsorted(a, d - np.timedelta64(30, 'D'), side='left'); hi = np.searchsorted(a, d, side='left'); ins[ii] = hi > lo
P['INS30'] = ins.astype(int)
sd = pd.read_pickle(X + 'runs515_panel2.pkl')[['d', 'tk', 'DTC']]; sd['d'] = sd.d.astype('datetime64[ns]'); P = P.merge(sd.drop_duplicates(['d', 'tk']), on=['d', 'tk'], how='left')
def lift2(G, flag):
    d = G.assign(f=flag.loc[G.index]); d = d[d.f.isin([0, 1])]; r = []
    for (m, b, p), gp in d.groupby(['mon', 'ab10', 'pb']):
        a, c = gp[gp.f == 1].y, gp[gp.f == 0].y
        if len(a) >= 3 and len(c) >= 3: r.append((m, a.mean() - c.mean(), len(a)))
    if len(r) < 4: return np.nan, np.nan
    o = pd.DataFrame(r, columns=['m', 'x', 'n']); mm = o.groupby('m').apply(lambda z: np.average(z.x, weights=z.n)); s = mm.std(); return mm.mean(), (mm.mean() / (s / np.sqrt(len(mm))) if s and s > 0 else np.nan)
OUT = []; LEAD = {}
for sc, G in P.groupby('sector'):
    if len(G) < 4000 or sc == 'UTIL': continue
    G = G.copy(); G['ab10'] = pd.qcut(G.ATRP.rank(method='first'), 10, labels=False); feats = {c: G[c].where(G.tk.isin(KG.keys())) for c in ('DEAL7', 'PR7', 'EARN7', 'NEG7', 'E14')}; feats['INS30'] = G.INS30
    q = G.loc[TRm.reindex(G.index).fillna(False), 'DTC'].quantile(.67); feats['DTC_hi'] = pd.Series(np.where(G.DTC > q, 1, np.where(G.DTC.notna(), 0, np.nan)), index=G.index)
    for c, hi in (('RND', 'HEALTH'), ('MARGIN', 'TECH')):
        if sc == hi: qq = G.loc[TRm.reindex(G.index).fillna(False), c].quantile([.33, .67]); feats[c + '_hi'] = pd.Series(np.where(G[c] > qq.iloc[1], 1, np.where(G[c] <= qq.iloc[0], 0, np.nan)), index=G.index)
    for k, f in feats.items():
        r = dict(sector=sc, feature=k, n1=int((f == 1).sum()))
        for e in ('2018', '2019-21', '2022', '2023-26'): d, t = lift2(G[G.era4 == e], f); r[e] = d
        d, t = lift2(G, f); r['pooled'] = d; r['t'] = t; ev = [r[e] for e in ('2018', '2019-21', '2022', '2023-26') if r[e] == r[e]]; r['agree'] = max(sum(x > 0 for x in ev), sum(x < 0 for x in ev))
        r['verdict'] = 'PASS' if (abs(t) >= 3 and r['agree'] >= 3) else ('lead' if (abs(t) >= 2 and r['agree'] >= 3) else '-'); OUT.append(r)
R = pd.DataFrame(OUT); R.to_csv('xs/report_517_sector_leads.csv', index=False); F2 = R.copy()
for c in ('2018', '2019-21', '2022', '2023-26', 'pooled'): F2[c] = (F2[c] * 100).round(1)
F2['t'] = F2.t.round(1); print('\n== STEP 2: within-ATRP-decile x price lift (pp of P(double<=40d)), flag=1 minus flag=0'); print(F2.to_string(index=False))
# ---- STEP 3: flips
H5 = pd.read_csv('flip_history.csv', parse_dates=['d']); H5['d'] = H5.d.astype('datetime64[ns]'); H5 = H5.merge(S[['tk', 'sector']], on='tk', how='left'); H5['sector'] = H5.sector.fillna('OTHER')
for col in ('DEAL', 'PR', 'EARN', 'NEG'): H5[col + '7'] = flagwin(col, H5.tk.values, H5.d.values, 7).astype(int)
H5['E14'] = flagwin('EARN', H5.tk.values, H5.d.values, -1, 14).astype(int); insf = np.zeros(len(H5), bool)
for i, (tk, d) in enumerate(zip(H5.tk.values, H5.d.values)):
    a = gI.get(tk)
    if a is not None: insf[i] = ((a < d) & (a >= d - np.timedelta64(30, 'D'))).any()
H5['INS30'] = insf.astype(int); H5 = pd.merge_asof(H5.sort_values('d'), FD[['tk', 'filed', 'RND', 'MARGIN']], left_on='d', right_on='filed', by='tk', direction='backward', tolerance=pd.Timedelta(days=540))
H5['yr'] = H5.d.dt.year; H5['era4'] = np.select([H5.yr == 2018, H5.yr <= 2021, H5.yr == 2022], ['2018', '2019-21', '2022'], '2023-26')
qh = P[(P.sector == 'HEALTH') & TRm].RND.quantile([.33, .67]); qt = P[(P.sector == 'TECH') & TRm].MARGIN.quantile([.33, .67])
H5['RND_hi'] = np.where((H5.sector == 'HEALTH') & (H5.RND > qh.iloc[1]), 1, np.where((H5.sector == 'HEALTH') & (H5.RND <= qh.iloc[0]), 0, np.nan)); H5['MARGIN_hi'] = np.where((H5.sector == 'TECH') & (H5.MARGIN > qt.iloc[1]), 1, np.where((H5.sector == 'TECH') & (H5.MARGIN <= qt.iloc[0]), 0, np.nan))
H5.to_pickle(SP + 'ext/flips517.pkl')
print('\n== STEP 3: sub-$5 vector FLIPS, lead present vs not (reach = peak>=+10% before exit; mean/median r of the flip trade)')
leads = R[R.verdict.isin(['PASS', 'lead'])][['sector', 'feature']].values.tolist(); print('leads carried to flips:', leads)
for sc, f in leads + [('HEALTH', 'E14'), ('ALL', 'E14')]:
    G = H5 if sc == 'ALL' else H5[H5.sector == sc]
    if f not in G: continue
    for lab, gx in (('all', G), ('2019-21', G[G.era4 == '2019-21']), ('2022', G[G.era4 == '2022']), ('2023-26', G[G.era4 == '2023-26'])):
        a, b = gx[gx[f] == 1], gx[gx[f] == 0]
        print(f'{sc:7s} {f:9s} {lab:8s} lead n={len(a):3d} reach {a.hit.mean():.2f} mean {a.r.mean():+.3f} med {a.r.median():+.3f} | none n={len(b):3d} reach {b.hit.mean():.2f} mean {b.r.mean():+.3f} med {b.r.median():+.3f}')
