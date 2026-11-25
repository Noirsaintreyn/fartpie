"""Report 497 (Nov 25 2026): Rey - run (1) insiders further down (Form 4 officer TITLE), (2) share-count / dilution from filings, (3) short interest, on the vector flips.
DECLARED BEFORE RESULTS. Same framework as 495/496: all daily flips with px<CAP (5, 10, 20), outcomes HIT (PEAK>=+10%) and r (winsorised), train 2019-21 / test 2022+, effect = flag coefficient controlling ATRP tercile x log-mcap tercile, SE clustered by month. LEAD = same sign train & test + pooled |t|>=2; REAL? = also test |t|>=2 with >=30 test events, at 2 of 3 caps, AND mechanism read.
(1) TITLES (Form 4 REPORTINGOWNER title joined on accession+owner CIK, buys code P in the prior 30d): BUY_TOP = CEO/President/Chair/CFO buyer; BUY_DOWN = officer whose title is NOT top (VP/EVP/SVP, controller, treasurer, secretary, GM, division, COO/CTO/other chief) - 'near the company'; BUY_DOWN_OPP = BUY_DOWN and not routine. Mechanism: operating insiders see orders/margins first, top bosses buy for signalling; hypothesis BUY_DOWN > BUY_TOP > none. Direction +.
(2) DILUTION (XBRL dei shares outstanding with filing dates, latest filed <= flip date vs the filing ~365d earlier, +/-60d): DIL_HI = growth >= +25%/yr, DIL_UP = growth >= +10% since the previous filing (recent issuance), DIL_FLAT = |growth| < 2%. Mechanism: serial issuers sell into any rally (ATM), capping upside and lowering reach; direction DIL_HI -, DIL_UP -, DIL_FLAT +.
(3) SHORT INTEREST (FINRA bi-monthly, usable 9 days after settlement): SI_DTC = days-to-cover >= 5; SI_PCT = short shares / XBRL shares >= 15%; SI_RISE = short position up >= 25% vs previous report. Mechanism: high short interest = squeeze fuel (forced covering on a flip) -> faster / higher reach (+); rising shorts = informed bearish flow (-), two-sided.
Mechanism read for any LEAD: ATRP/size, PEAK median, era split, top-ticker concentration (as 495)."""
import numpy as np, pandas as pd, json, os, glob, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
SPd = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
src = open('xs/report_489_informed_insiders_early_earnings.py').read(); exec(src[:src.index("for win, nm in")])
# ---- (1) titles
T = pd.read_pickle(SPd + 'ext/f345_owner_titles.pkl'); T['RPTOWNERCIK'] = T.RPTOWNERCIK.astype(str).str.lstrip('0'); T = T.drop_duplicates(['ACCESSION_NUMBER', 'RPTOWNERCIK'])
I['ck'] = I.RPTOWNERCIK.astype(str).str.lstrip('0'); I = I.merge(T[['ACCESSION_NUMBER', 'RPTOWNERCIK', 'RPTOWNER_TITLE']].rename(columns={'RPTOWNERCIK': 'ck'}), on=['ACCESSION_NUMBER', 'ck'], how='left')
ttl = I.RPTOWNER_TITLE.fillna('').str.lower()
top = ttl.str.contains(r'chief executive|\bceo\b|president|chairman|chair\b|chief financial|\bcfo\b|founder') & ~ttl.str.contains(r'vice president|vp\b|svp|evp')
I['isoff'] = I.rel.str.contains('Officer') | (ttl != ''); I['TOPT'] = top & I.isoff; I['DOWNT'] = I.isoff & ~top & (ttl != '')
print(f"title join: {ttl.ne('').mean():.0%} of buys have a title; top {I.TOPT.mean():.0%}, down {I.DOWNT.mean():.0%}; examples down: {I[I.DOWNT].RPTOWNER_TITLE.value_counts().head(6).to_dict()}")
gI = {t: x.sort_values('fd') for t, x in I.groupby('ISSUERTRADINGSYMBOL')}; b_top = np.zeros(len(O), bool); b_dn = np.zeros(len(O), bool); b_dno = np.zeros(len(O), bool)
for i, (tk, d) in enumerate(zip(O.tk.values, O.d.values)):
    x = gI.get(tk)
    if x is None: continue
    w = x[(x.fd < d) & (x.fd >= d - np.timedelta64(30, 'D'))]
    if len(w): b_top[i] = w.TOPT.any(); b_dn[i] = w.DOWNT.any(); b_dno[i] = (w.DOWNT & ~w.routine).any()
O['BUY_TOP'] = b_top; O['BUY_DOWN'] = b_dn; O['BUY_DOWN_OPP'] = b_dno
# ---- (2) dilution from XBRL shares outstanding (with filed dates)
sh = {}
for f in glob.glob('data/fund/sec_small/*.json') + glob.glob('data/fund/sec/*.json'):
    try: j = json.load(open(f))
    except Exception: continue
    tk = os.path.basename(f)[:-5]; u = j.get('facts', {}).get('dei', {}).get('EntityCommonStockSharesOutstanding', {}).get('units', {}).get('shares', [])
    if u: z = pd.DataFrame(u)[['filed', 'val']]; z['filed'] = pd.to_datetime(z.filed); sh[tk] = z.sort_values('filed').drop_duplicates('filed', keep='last').reset_index(drop=True)
print(f'XBRL shares for {len(sh)} tickers; flips covered: {O.tk.isin(sh).mean():.0%}')
dil = np.full(len(O), np.nan); rec = np.full(len(O), np.nan); sh_now = np.full(len(O), np.nan)
for i, (tk, d) in enumerate(zip(O.tk.values, O.d.values)):
    z = sh.get(tk)
    if z is None: continue
    a = z[z.filed <= d]
    if len(a) < 2: continue
    cur = a.iloc[-1]; sh_now[i] = cur.val; rec[i] = cur.val / a.iloc[-2].val - 1
    tgt = cur.filed - pd.Timedelta(days=365); near = a[(a.filed - tgt).abs() <= pd.Timedelta(days=60)]
    if len(near): dil[i] = cur.val / near.iloc[(near.filed - tgt).abs().argmin()].val - 1
O['DIL_HI'] = dil >= .25; O['DIL_UP'] = rec >= .10; O['DIL_FLAT'] = np.abs(dil) < .02
for c, v in (('DIL_HI', dil), ('DIL_UP', rec), ('DIL_FLAT', dil)): O.loc[np.isnan(v), c] = False
# ---- (3) short interest
SI = pd.read_pickle(SPd + 'ext/finra_si.pkl'); SI['symbolCode'] = SI.symbolCode.astype(str); SI = SI.sort_values('settle')
for c in ('currentShortPositionQuantity', 'previousShortPositionQuantity', 'daysToCoverQuantity'): SI[c] = pd.to_numeric(SI[c], errors='coerce')
SI['avail'] = SI.settle + pd.Timedelta(days=9); gS = {t: x for t, x in SI.groupby('symbolCode')}
dtc = np.full(len(O), np.nan); sip = np.full(len(O), np.nan); sir = np.full(len(O), np.nan)
for i, (tk, d) in enumerate(zip(O.tk.values, O.d.values)):
    x = gS.get(tk)
    if x is None: continue
    a = x[x.avail <= d]
    if not len(a): continue
    r = a.iloc[-1]
    if (pd.Timestamp(d) - r.avail).days > 25: continue
    dtc[i] = r.daysToCoverQuantity; sip[i] = r.currentShortPositionQuantity / sh_now[i] if sh_now[i] == sh_now[i] and sh_now[i] > 0 else np.nan
    sir[i] = r.currentShortPositionQuantity / r.previousShortPositionQuantity - 1 if r.previousShortPositionQuantity and r.previousShortPositionQuantity > 0 else np.nan
print(f'short interest matched for {np.mean(~np.isnan(dtc)):.0%} of flips; pct-of-shares for {np.mean(~np.isnan(sip)):.0%}')
O['SI_DTC'] = dtc >= 5; O['SI_PCT'] = sip >= .15; O['SI_RISE'] = sir >= .25
O['hit'] = (O.PEAK >= .10).astype(float); lo, hi = O.r.quantile([.01, .99]); O['rw'] = O.r.clip(lo, hi); O['mo_'] = O.d.dt.to_period('M').astype(str)
def eff(D, flag, y):
    if D[flag].sum() < 8: return (np.nan, np.nan, int(D[flag].sum()))
    m = smf.ols(f'{y} ~ {flag}.astype(float) + C(a3) + C(c3)', D).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D.mo_)[0]}); k = [n for n in m.params.index if n.startswith(flag)][0]
    return (m.params[k], m.tvalues[k], int(D[flag].sum()))
flags = ['BUY_TOP', 'BUY_DOWN', 'BUY_DOWN_OPP', 'DIL_HI', 'DIL_UP', 'DIL_FLAT', 'SI_DTC', 'SI_PCT', 'SI_RISE']
for CAP in (5, 10, 20):
    S = O[(O.px < CAP) & O.ATRP.notna() & O.LMC.notna()].copy(); tr = S.tr
    S['a3'] = pd.cut(S.ATRP, [-np.inf, *np.quantile(S.ATRP[tr], [1/3, 2/3]), np.inf], labels=False); S['c3'] = pd.cut(S.LMC, [-np.inf, *np.quantile(S.LMC[tr], [1/3, 2/3]), np.inf], labels=False)
    print(f'\n######## px<{CAP}: n {len(S)} train {int(tr.sum())} test {int((~tr).sum())} | base reach {S[tr].hit.mean():.0%}/{S[~tr].hit.mean():.0%}')
    print(f"{'flag':13s}| n tr/te | raw test reach flag vs rest | ADJ reach train test all (t) | ADJ mean r train test all (t) | verdict")
    for f in flags:
        te = S[~S.tr]; a = [eff(D, f, 'hit') for D in (S[S.tr], te, S)]; b = [eff(D, f, 'rw') for D in (S[S.tr], te, S)]
        raw = f"{te[te[f]].hit.mean():.0%} vs {te[~te[f]].hit.mean():.0%}" if te[f].sum() else 'n/a'
        same = lambda z: (np.sign(z[0][0]) == np.sign(z[1][0])) and abs(z[2][1]) >= 2; v = 'LEAD' if (same(a) or same(b)) else '-'
        if v == 'LEAD' and te[f].sum() >= 30 and (abs(a[1][1]) >= 2 or abs(b[1][1]) >= 2): v = 'REAL?'
        print(f"{f:13s}| {a[0][2]:3d}/{a[1][2]:3d} | {raw:26s} | {a[0][0]:+.2f} {a[1][0]:+.2f} {a[2][0]:+.2f} (t {a[2][1]:+.1f}) | {b[0][0]:+.3f} {b[1][0]:+.3f} {b[2][0]:+.3f} (t {b[2][1]:+.1f}) | {v}")
