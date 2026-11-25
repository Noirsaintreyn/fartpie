"""Report 470 (Nov 24 2026): Rey - the DEAL lead (8-K item 1.01/2.01 in prior 7d) lumps dilution financings with real commercial deals. Split it by MECHANISM and size, inside the high-vol base group (top 30% base5 score, train-cut per price bucket), px<20 / <10 / <5.
DECLARED BEFORE RESULTS. For each flip with a DEAL 8-K known before the flip day (latest one), the 8-K main document text is pulled from EDGAR (point-in-time: that filing) and classified by the EARLIEST keyword in the text of the item 1.01/2.01 section:
 FIN  (dilution/financing; predicted NOT to lead): securities purchase agreement, at-the-market, equity distribution, sales agreement, underwriting agreement, placement agent, registered direct, private placement, warrant, convertible, promissory note, loan agreement, credit agreement, security agreement, subscription, purchase of shares/common stock, securities exchange
 COM  (real business event; predicted to lead): license agreement, licensing, collaboration, supply agreement, distribution agreement, agreement and plan of merger, merger agreement, acquisition, acquire, asset purchase, joint venture, partnership, development agreement, master services, contract award, purchase order, definitive agreement to be acquired
 OTH  = neither. SIZE = first dollar figure in the item section / market cap (SEC diluted-shares-outstanding x flip price); BIGDEAL = COM with SIZE >= 10% of market cap (predicted to lead more).
Metrics: REACH (PEAK>=+10%) and REAL (r>=+10%) for COM / FIN / BIGDEAL vs the rest of the group (no DEAL), by era 2019-21 / 2022 / 2023-26, with pooled two-proportion z. PASS (declared) = COM beats no-DEAL on REACH in EVERY era (n>=8 each) AND pooled z >= 2 AND FIN does not. Small n is expected: pass is judged, power is reported."""
import numpy as np, pandas as pd, json, os, re, time, requests, html, warnings; warnings.filterwarnings('ignore')
src = open('xs/report_464_catalyst_flips.py').read(); exec(src[:src.index('def mp_(')])
O['tr'] = (O.d.dt.year <= 2021).values
H = {'User-Agent': 'research rey@example.com'}; CACHE = SP + 'ext/deal_docs/'; os.makedirs(CACHE, exist_ok=True)
CIK = {v['ticker']: v['cik_str'] for v in json.load(open('fartpie/data/fund/company_tickers.json' if os.path.exists('fartpie/data/fund/company_tickers.json') else 'data/fund/company_tickers.json')).values()}
def get(url):
    for a in range(3):
        try:
            r = requests.get(url, headers=H, timeout=40)
            if r.status_code == 200: return r
            if r.status_code == 404: return None
        except Exception: pass
        time.sleep(3 * (a + 1))
    return None
def submissions(tk):
    f = CACHE + f'sub_{tk}.pkl'
    if os.path.exists(f): return pd.read_pickle(f)
    c = CIK.get(tk)
    if c is None: return None
    r = get(f'https://data.sec.gov/submissions/CIK{c:010d}.json'); time.sleep(.15)
    if r is None: return None
    j = r.json(); fl = [j['filings']['recent']]
    for x in j['filings'].get('files', []):
        if x['filingTo'] >= '2019-01-01':
            q = get('https://data.sec.gov/submissions/' + x['name']); time.sleep(.15)
            if q is not None: fl.append(q.json())
    D = pd.concat([pd.DataFrame({k: v[k] for k in ('accessionNumber', 'filingDate', 'form', 'items', 'primaryDocument')}) for v in fl]); D = D[D.form.str.startswith('8-K')]; D['filingDate'] = pd.to_datetime(D.filingDate)
    pd.to_pickle(D, f); return D
# DEAL events -> one filing per flip
O['deal_known'] = pd.NaT; g = {t: x.sort_values('known') for t, x in K[K.DEAL].groupby('ticker')}; ev = []
for i, (tk, d) in enumerate(zip(O.tk.values, O.d.values)):
    if tk in g:
        x = g[tk]; x = x[(x.known.values <= d) & (x.known.values > d - np.timedelta64(7, 'D'))]
        if len(x): ev.append((i, tk, x.iloc[-1].filed))
print('flips with a DEAL event', len(ev))
FIN = ['securities purchase agreement', 'at-the-market', 'at the market', 'equity distribution', 'sales agreement', 'underwriting agreement', 'placement agent', 'registered direct', 'private placement', 'warrant', 'convertible', 'promissory note', 'loan agreement', 'credit agreement', 'security agreement', 'subscription', 'purchase of shares', 'purchase of common stock', 'securities exchange', 'exchange agreement', 'note purchase']
COM = ['license agreement', 'licensing', 'collaboration', 'supply agreement', 'distribution agreement', 'agreement and plan of merger', 'merger agreement', 'acquisition', 'acquire', 'asset purchase', 'joint venture', 'partnership', 'development agreement', 'master services', 'contract award', 'purchase order', 'to be acquired']
def text(tk, filed):
    f = CACHE + f'{tk}_{pd.Timestamp(filed).date()}.txt'
    if os.path.exists(f): return open(f).read()
    S = submissions(tk)
    if S is None: return None
    s = S[S.filingDate == pd.Timestamp(filed)]; s = s[s['items'].astype(str).str.contains('1.01|2.01')]
    if s.empty: return None
    r0 = s.iloc[0]; r = get(f'https://www.sec.gov/Archives/edgar/data/{CIK[tk]}/{r0.accessionNumber.replace("-", "")}/{r0.primaryDocument}'); time.sleep(.15)
    if r is None: return None
    t = re.sub(r'<[^>]+>', ' ', r.text); t = html.unescape(re.sub(r'\s+', ' ', t)); open(f, 'w').write(t); return t
cls = {}; size = {}
for n, (i, tk, filed) in enumerate(ev):
    t = text(tk, filed)
    if t is None: cls[i] = 'NA'; continue
    m = re.search(r'item\s+(1\.01|2\.01)', t, re.I); sec_ = t[m.start():m.start() + 6000] if m else t[:6000]; low = sec_.lower()
    pf = min([(low.find(k), 'FIN') for k in FIN if k in low] or [(10 ** 9, '')]); pc = min([(low.find(k), 'COM') for k in COM if k in low] or [(10 ** 9, '')])
    cls[i] = 'OTH' if min(pf[0], pc[0]) == 10 ** 9 else (pf[1] if pf[0] < pc[0] else pc[1])
    a = re.search(r'\$\s?([\d,]+(?:\.\d+)?)\s*(million|billion|thousand)?', sec_, re.I)
    if a: size[i] = float(a.group(1).replace(',', '')) * {'billion': 1e9, 'million': 1e6, 'thousand': 1e3, None: 1}[a.group(2).lower() if a.group(2) else None]
    if n % 40 == 0: print('..', n, flush=True)
O['dcls'] = pd.Series(cls); O['dsize'] = pd.Series(size)
# market cap from SEC shares
def shares(tk, d):
    f = f'data/fund/sec_small/{tk}.json'
    if not os.path.exists(f): return np.nan
    e = json.load(open(f))['facts'].get('dei', {}).get('EntityCommonStockSharesOutstanding', {}).get('units', {}).get('shares', [])
    x = [(pd.Timestamp(q['filed']), q['val']) for q in e if 'filed' in q and pd.Timestamp(q['filed']) <= d]
    return max(x)[1] if x else np.nan
O['mcap'] = [shares(tk, d) * px if i in cls else np.nan for i, (tk, d, px) in enumerate(zip(O.tk, O.d, O.px))]
O['ratio'] = O.dsize / O.mcap; O['BIG'] = (O.dcls == 'COM') & (O.ratio >= .1)
print('classified:', O.dcls.value_counts().to_dict(), '| size parsed', O.dsize.notna().sum(), '| ratio', O.ratio.notna().sum())
B = {'ATRP': 1, 'F11': 1, 'VOLX': 1, 'LTURN': 1, 'LMC': -1}
def score(D):
    R = pd.DataFrame(index=D.index)
    for f, s in B.items():
        x = D[f].astype(float); ref = np.sort(x[D.tr & x.notna()].values); q = np.searchsorted(ref, x.values) / len(ref); R[f] = q if s > 0 else 1 - q; R.loc[x.isna(), f] = np.nan
    return R.mean(axis=1)
def pz(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 5 or n2 < 5: return np.nan
    p1, p2 = a.mean(), b.mean(); p = (a.sum() + b.sum()) / (n1 + n2); se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)); return (p1 - p2) / se if se > 0 else np.nan
for lab, cap in (('px<20', 20), ('px<10', 10), ('px<5', 5)):
    S = O[O.px < cap].copy(); S['s_'] = score(S); cut = np.quantile(S[S.tr].s_.dropna(), .7); G = S[S.s_ >= cut]; rest = G[G.dcls.isna()]
    print(f'\n######## {lab}: all flips {len(S)} | base group {len(G)} | DEAL flips in group {G.dcls.notna().sum()} (COM {(G.dcls == "COM").sum()}, FIN {(G.dcls == "FIN").sum()}, OTH {(G.dcls == "OTH").sum()}, NA {(G.dcls == "NA").sum()}) | no-DEAL REACH {(rest.PEAK >= .1).mean():.1%} REAL {(rest.r >= .1).mean():.1%}')
    for nm, m in (('COM', G.dcls == 'COM'), ('FIN', G.dcls == 'FIN'), ('OTH', G.dcls == 'OTH'), ('BIGDEAL(COM,>=10% mcap)', G.BIG), ('COM+INS', (G.dcls == 'COM') & G.INS)):
        a = G[m]; s = ''
        for e in ('2019-21', '2022', '2023-26'):
            x = a[a.era3 == e]; y = rest[rest.era3 == e]; s += f' {(x.PEAK >= .1).mean() * 100:3.0f}% v {(y.PEAK >= .1).mean() * 100:3.0f}% [{len(x)}] /' if len(x) else ' - /'
        print(f'  {nm:24s} n {len(a):3d} | REACH {(a.PEAK >= .1).mean():.0%} v {(rest.PEAK >= .1).mean():.0%} (z {pz((a.PEAK >= .1).astype(float), (rest.PEAK >= .1).astype(float)):+.1f}) | REAL {(a.r >= .1).mean():.0%} v {(rest.r >= .1).mean():.0%} (z {pz((a.r >= .1).astype(float), (rest.r >= .1).astype(float)):+.1f}) | median r {a.r.median() * 100:+.1f}% v {rest.r.median() * 100:+.1f}% | era REACH:{s}')
    # all flips (not only group): does COM lead on the plain flip universe?
    allr = S[S.dcls.isna()]; a = S[S.dcls == 'COM']; b = S[S.dcls == 'FIN']
    print(f'  [all {lab} flips, no volatility base] COM n {len(a)} REACH {(a.PEAK >= .1).mean():.0%} v no-DEAL {(allr.PEAK >= .1).mean():.0%} (z {pz((a.PEAK >= .1).astype(float), (allr.PEAK >= .1).astype(float)):+.1f}) | FIN n {len(b)} {(b.PEAK >= .1).mean():.0%}')
print('\nexamples COM/FIN (tk date size ratio):'); print(O[O.dcls.isin(['COM', 'FIN'])].sample(12, random_state=1)[['tk', 'd', 'dcls', 'dsize', 'ratio', 'PEAK', 'r']].to_string())
