"""Report 489 (Nov 25 2026): Rey - (a) 'find MORE INFORMED insider buying: someone who buys the same month 3 years running is routine, a unique/unusual buyer is different; size and role', (b) 'insiders further down near the company, not the top bosses', (c) 'maybe companies moving their earnings EARLIER'.
DECLARED BEFORE RESULTS. Same group as the live alert (4-feature base5: ATRP, F11, LTURN, -LMC; train-ECDF top 30% per price bucket), daily flips, test 2022+, px<20/<10/<5. Data: Form 4 open-market PURCHASES only (code P, 2018-Mar 2026; sales not in the file), owner CIK, relationship (Director / Officer / TenPercentOwner / Other; officer TITLE is not in the file so 'further down' cannot be split from CEO/CFO here), 8-K item 2.02 filing dates.
Window = purchases filed in the 30 days before the flip close. Per buyer-issuer pair: ROUTINE = the pair also bought in the SAME calendar month in each of the 2 prior years (Cohen-Malloy-Pomorski routine rule, relaxed from 3 years because history starts 2018 and sales are missing); OPP = everything else (opportunistic).
Flags: INS_OPP (>=1 opportunistic buy in window), INS_ROUT (buys only by routine pairs), FIRST (buyer's first purchase in this issuer for 24+ months or ever in the file), BIG (one buyer >= $100k in the window), CLUS (>=2 distinct buyers), OFF / DIR / TEN (an Officer / Director / 10%-owner bought), OPP_BIG_OFF (opportunistic officer buy >= $50k).
EARLY earnings: the latest 2.02 filing in the 8 days before the flip was filed >= 5 calendar days EARLIER than the same company's 2.02 for the same quarter a year before (aligned +365d, within +-40d); LATE = >= 5 days later; EARLY30 = same with a 30-day window.
Read each flag vs rest of group: n, REACH, mean, median, p10, era (2022 | 2023-26), month-clustered t of the mean gap. PASS = beats rest on mean AND REACH in both eras and t >= 2.0 (flags are few and correlated); n < 30 is called a lead at best."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
src = open('xs/report_464_catalyst_flips.py').read(); exec(src[:src.index('def mp_(')])
O['tr'] = O.d.dt.year <= 2021; O = O.reset_index(drop=True)
I = pd.read_pickle(SP + 'ext/insider_P_hist.pkl'); I = I[(I.TRANS_CODE == 'P') & (I.TRANS_ACQUIRED_DISP_CD == 'A')].copy(); I = I.dropna(subset=['ISSUERTRADINGSYMBOL', 'RPTOWNERCIK']).copy(); I['ISSUERTRADINGSYMBOL'] = I.ISSUERTRADINGSYMBOL.astype(object); I['RPTOWNERCIK'] = I.RPTOWNERCIK.astype(object); I['fd'] = pd.to_datetime(I.fd); I['usd'] = pd.to_numeric(I.TRANS_SHARES, errors='coerce') * pd.to_numeric(I.TRANS_PRICEPERSHARE, errors='coerce'); I['pair'] = I.ISSUERTRADINGSYMBOL.astype(str) + '|' + I.RPTOWNERCIK.astype(str)
I['y'] = I.fd.dt.year; I['m'] = I.fd.dt.month; months = I.groupby('pair').apply(lambda x: set(zip(x.y, x.m))).to_dict()
I['routine'] = [((y - 1, m) in months[p]) and ((y - 2, m) in months[p]) for p, y, m in zip(I.pair, I.y, I.m)]
firstd = I.groupby('pair').fd.apply(lambda x: np.sort(x.values)); I['prev_gap'] = [np.nan] * len(I)
pg = np.full(len(I), np.nan); ix = {p: [] for p in firstd.index}
for k, (p, f) in enumerate(zip(I.pair.values, I.fd.values)): ix[p].append((f, k))
for p, lst in ix.items():
    lst.sort(); last = None
    for f, k in lst:
        pg[k] = (f - last) / np.timedelta64(1, 'D') if last is not None else 9999; last = f
I['prev_gap'] = pg; I['rel'] = I.RPTOWNER_RELATIONSHIP.astype(str); gI = {t: x.sort_values('fd') for t, x in I.groupby('ISSUERTRADINGSYMBOL')}
F = {k: np.zeros(len(O), bool) for k in ('INS_OPP', 'INS_ROUT', 'FIRST', 'BIG', 'CLUS', 'OFF', 'DIR', 'TEN', 'OPP_BIG_OFF')}
for i, (tk, d) in enumerate(zip(O.tk.values, O.d.values)):
    x = gI.get(tk)
    if x is None: continue
    w = x[(x.fd < d) & (x.fd >= d - np.timedelta64(30, 'D'))]
    if not len(w): continue
    F['INS_OPP'][i] = (~w.routine).any(); F['INS_ROUT'][i] = w.routine.all(); F['FIRST'][i] = (w.prev_gap >= 730).any(); F['BIG'][i] = (w.groupby('pair').usd.sum() >= 1e5).any(); F['CLUS'][i] = w.RPTOWNERCIK.nunique() >= 2
    F['OFF'][i] = w.rel.str.contains('Officer').any(); F['DIR'][i] = w.rel.str.contains('Director').any(); F['TEN'][i] = w.rel.str.contains('TenPercent').any(); F['OPP_BIG_OFF'][i] = ((~w.routine) & w.rel.str.contains('Officer') & (w.usd >= 5e4)).any()
for k, v in F.items(): O[k] = v
E2 = K[K['items'].astype(str).str.contains('2.02')][['ticker', 'filed']].drop_duplicates(); E2['filed'] = pd.to_datetime(E2.filed); gE = {t: np.sort(x.filed.values) for t, x in E2.groupby('ticker')}
def early(tk, d, win):
    a = gE.get(tk)
    if a is None: return np.nan
    m = a[(a <= d) & (a > d - np.timedelta64(win, 'D'))]
    if not len(m): return np.nan
    cur = m[-1]; tgt = cur - np.timedelta64(365, 'D'); near = a[(np.abs(a - tgt) <= np.timedelta64(40, 'D')) & (a < cur - np.timedelta64(200, 'D'))]
    if not len(near): return np.nan
    p = near[np.argmin(np.abs(near - tgt))]; return (cur - p) / np.timedelta64(1, 'D') - 365
for win, nm in ((8, ''), (30, '30')):
    sh = np.array([early(t, d, win) for t, d in zip(O.tk.values, O.d.values)]); O['EARLY' + nm] = sh <= -5; O['LATE' + nm] = sh >= 5
B = {'ATRP': 1, 'F11': 1, 'LTURN': 1, 'LMC': -1}
def score(D):
    R = []
    for f, s in B.items():
        x = D[f].astype(float); ref = np.sort(x[D.tr & x.notna()].values); q = np.searchsorted(ref, x.values) / len(ref); q = np.where(x.isna(), np.nan, q); R.append(q if s > 0 else 1 - q)
    return np.nanmean(np.vstack(R), axis=0)
st = lambda x: f'n {len(x):4d} REACH {(x.PEAK >= .1).mean():.0%} win {(x.r > 0).mean():.0%} mean {x.r.mean():+6.1%} med {x.r.median():+.1%} p10 {x.r.quantile(.1):+.0%}' if len(x) else 'n 0'
FL = ['INS_OPP', 'INS_ROUT', 'FIRST', 'BIG', 'CLUS', 'OFF', 'DIR', 'TEN', 'OPP_BIG_OFF', 'EARLY', 'LATE', 'EARLY30', 'LATE30']
for cap in (20, 10, 5):
    S = O[O.px < cap].copy(); S['s_'] = score(S); cut = np.quantile(S[S.tr].s_.dropna(), .7); G = S[(~S.tr) & (S.s_ >= cut)].copy(); G['mo'] = G.d.dt.to_period('M').astype(str); anyins = G.INS_OPP | G.INS_ROUT
    print(f'\n######## sub-${cap} group test 2022+: {st(G)} | any insider buy in 30d: {st(G[anyins])}')
    for f in FL:
        a = G[G[f]]; b = G[~G[f]]
        if len(a) < 5: print(f'  {f:11s} n {len(a)} (too few)'); continue
        ma = a.groupby('mo').r.mean(); mb = b.groupby('mo').r.mean(); j = pd.concat([ma, mb], axis=1, keys=['a', 'b']).dropna(); d = j.a - j.b; t = d.mean() / (d.std() / np.sqrt(len(d))) if len(d) > 2 else np.nan
        e = lambda x, y: (x[x.d.dt.year == 2022].r.mean() if y == 22 else x[x.d.dt.year >= 2023].r.mean())
        print(f'  {f:11s} FLAG {st(a)} | REST mean {b.r.mean():+.1%} REACH {(b.PEAK >= .1).mean():.0%} | gap {d.mean():+.1%} t {t:+.2f} | 2022 {e(a, 22):+.1%} (n {(a.d.dt.year == 2022).sum()}) vs {e(b, 22):+.1%} | 2023-26 {e(a, 23):+.1%} vs {e(b, 23):+.1%}')
