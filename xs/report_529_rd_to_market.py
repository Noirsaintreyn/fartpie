"""Report 529 (Nov 28 2026): Rey's attachment #8 (Consensus): R&D effect strongest scaled by MARKET EQUITY (Chan-Lakonishok-Sougiannis 2001), concentrated in past losers; R&D-return relation sector/size dependent.
DECLARED BEFORE RESULTS. Within HEALTH and TECH sub-$5 stock-days (template 522: ATRP decile x price bucket x month, top vs bottom third by 2019-21 cut points, month-clustered t): 
 H1  RDME = latest-10-K R&D expense / market cap (shares_pit x close) lift beats RND (R&D/assets) lift (HEALTH RND: +4.2pp t 4.2).
 H2  RDME lift larger among past losers (DD252 below train median) than non-losers.
Eras 2018 / 2019-21 / 2022 / 2023-26. PASS = pooled |t|>=3 and same sign >=3 of 4 eras. Trials: 2 sectors x (RND, RDME, RDME|loser, RDME|nonloser) -> report all.
NOT testable (data missing): 13F short-term/dedicated institutional flow, proactive-trading score, 8-K filing-lag leakage, institutional insider-filing tracking.
"""
import numpy as np, pandas as pd, json, io, contextlib, warnings; warnings.filterwarnings('ignore')
src = open('xs/report_516_sector.py').read().split("TRm = (P.yr")[0]
with contextlib.redirect_stdout(io.StringIO()): exec(src)
TRm = (P.yr >= 2019) & (P.yr <= 2021); P['pb'] = np.select([P.c < 1, P.c < 2.5], [0, 1], 2); P['era4'] = np.select([P.yr == 2018, P.yr <= 2021, P.yr == 2022], ['2018', '2019-21', '2022'], '2023-26')
DDp = pd.read_pickle(X + 'runs515_panel.pkl')[['tk', 'd', 'DD252']]; DDp['d'] = DDp.d.astype('datetime64[ns]'); P['d'] = P.d.astype('datetime64[ns]'); P = P.merge(DDp, on=['tk', 'd'], how='left')
sh = pd.read_pickle(X + 'shares_pit.pkl'); sh['filed'] = pd.to_datetime(sh.filed).astype('datetime64[ns]'); sh = sh.sort_values('filed')
rd = []
for tk in P[P.sector.isin(['HEALTH', 'TECH'])].tk.unique():
    try: gg = json.load(open(f'data/fund/sec_small/{tk}.json'))['facts']['us-gaap']
    except Exception: continue
    a = ann(gg, ['ResearchAndDevelopmentExpense'], 0)
    if a is None: continue
    e = pd.DataFrame({'RDX': a.val, 'frd': a.filed}); e['tk'] = tk; rd.append(e.dropna(subset=['frd']))
RDX = pd.concat(rd).sort_values('frd'); RDX['frd'] = RDX.frd.astype('datetime64[ns]')
def lift2(Gx, col):
    d = Gx[Gx[col].isin([0, 1])]; r = []
    for (m, b, p), gp in d.groupby(['mon', 'ab10', 'pb']):
        a, c = gp[gp[col] == 1].y, gp[gp[col] == 0].y
        if len(a) >= 3 and len(c) >= 3: r.append((m, a.mean() - c.mean(), len(a)))
    if len(r) < 4: return np.nan, np.nan
    o = pd.DataFrame(r, columns=['m', 'x', 'n']); mm = o.groupby('m').apply(lambda z: np.average(z.x, weights=z.n)); s = mm.std(); return mm.mean(), (mm.mean() / (s / np.sqrt(len(mm))) if s and s > 0 else np.nan)
pd.set_option('display.width', 220)
for sec in ('HEALTH', 'TECH'):
    G = P[P.sector == sec].copy(); G['ab10'] = pd.qcut(G.ATRP.rank(method='first'), 10, labels=False); G['d'] = G.d.astype('datetime64[ns]')
    G = pd.merge_asof(G.sort_values('d'), sh[['tk', 'filed', 'val']].rename(columns={'filed': 'fs', 'val': 'SH'}), left_on='d', right_on='fs', by='tk', direction='backward', tolerance=pd.Timedelta(days=730))
    G = pd.merge_asof(G.sort_values('d'), RDX.rename(columns={'frd': 'fr'}), left_on='d', right_on='fr', by='tk', direction='backward', tolerance=pd.Timedelta(days=540)); G['RDME'] = G.RDX / (G.SH * G.c)
    tm = G.yr.between(2019, 2021)  # NB: index reset by merge_asof, so mask from G.yr (not TRm.reindex)
    for c in ('RND', 'RDME'):
        x = G.loc[tm, c].dropna(); q = x.quantile([.33, .67]); G[c + 'hi'] = np.where(G[c] > q.iloc[1], 1, np.where(G[c] <= q.iloc[0], 0, np.nan))
    med = G.loc[tm, 'DD252'].median(); G['LOSER'] = np.where(G.DD252 < med, 1, np.where(G.DD252.notna(), 0, np.nan))
    print(f'\n== {sec}: stock-days {len(G)} coverage RND {G.RND.notna().mean():.2f} RDME {G.RDME.notna().mean():.2f}')
    rows = []
    for lab, col, m in (('RND (R&D/assets)', 'RNDhi', G.RNDhi.notna()), ('RDME (R&D/mkt cap)', 'RDMEhi', G.RDMEhi.notna()), ('RDME | past losers', 'RDMEhi', G.LOSER == 1), ('RDME | non-losers', 'RDMEhi', G.LOSER == 0)):
        Gx = G[m]; r = dict(measure=lab, n=len(Gx))
        for e in ('2018', '2019-21', '2022', '2023-26'): d, t = lift2(Gx[Gx.era4 == e], col); r[e] = None if d != d else round(d * 100, 1)
        d, t = lift2(Gx, col); r['pooled_pp'] = None if d != d else round(d * 100, 1); r['t'] = None if t != t else round(t, 1); rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))
