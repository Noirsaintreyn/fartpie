"""Report 491 (Nov 25 2026): Rey - 'companies often copy last year's report; the ones that change the words lagged by up to 7 percent' (Cohen-Malloy-Nathan 'Lazy Prices': firms that change their 10-K/10-Q language underperform the 'same as last year' firms, driven by risk-factor/legal text changes). Test on the LIVE sub-$5 group (4-feature top 30%, daily flips, test 2022+), the only bucket where the full text pull is feasible.
DECLARED BEFORE RESULTS. For each flip: the latest 10-K/10-Q filed in the 90 days before the flip close, and the same company's filing of the SAME form about a year earlier (filed within +-45d of cur-365d). Similarity = cosine of word-count vectors over the whole document (CMN's simplest measure). CHANGED = similarity in the lowest tercile of the sample's similarity distribution (text only, no outcomes used), SAME = the top tercile. Hypothesis (CMN): CHANGED flips do WORSE (mean and REACH) than SAME flips. Read n, REACH, mean, median, p10, era split, month-clustered t of (SAME - CHANGED). PASS = SAME beats CHANGED on mean AND REACH in both eras with t >= 2.0 (n will be small: 'lead' otherwise)."""
import numpy as np, pandas as pd, re, os, json, time, pickle, requests, warnings, collections, sys; warnings.filterwarnings('ignore'); sys.path.insert(0, '.')
from concurrent.futures import ThreadPoolExecutor
SP = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; H = {'User-Agent': 'research rey@example.com'}; CK = SP + 'lazy_cache.pkl'; cache = pickle.load(open(CK, 'rb')) if os.path.exists(CK) else {'sub': {}, 'sim': {}}
O = pd.read_pickle(SP + 'ext/structure_small.pkl'); O['d'] = pd.to_datetime(O.d); O['tr'] = O.d.dt.year <= 2021
B = {'ATRP': 1, 'F11': 1, 'LTURN': 1, 'LMC': -1}
def rank4(D):
    R = []
    for f, s in B.items():
        x = D[f].astype(float); ref = np.sort(x[D.tr & x.notna()].values); q = np.searchsorted(ref, x.values) / len(ref); q = np.where(x.isna(), np.nan, q); R.append(q if s > 0 else 1 - q)
    return np.nanmean(np.vstack(R), axis=0)
S = O[O.px < 5].dropna(subset=['r', 'PEAK']).copy(); S['s4'] = rank4(S); cut = np.quantile(S[S.tr].s4.dropna(), .7); G = S[(~S.tr) & (S.s4 >= cut)].reset_index(drop=True)
cikm = {v['ticker']: v['cik_str'] for v in json.load(open('data/fund/company_tickers.json')).values()}
def get(url):
    for k in range(3):
        try:
            r = requests.get(url, headers=H, timeout=40)
            if r.status_code == 200: return r
            time.sleep(1 + k)
        except Exception: time.sleep(1 + k)
    return None
def filings(tk):
    if tk in cache['sub']: return cache['sub'][tk]
    c = cikm.get(tk); out = []
    if c:
        r = get(f'https://data.sec.gov/submissions/CIK{c:010d}.json'); time.sleep(.12)
        if r is not None:
            j = r.json(); blocks = [j['filings']['recent']]
            for f in j['filings'].get('files', []):
                rr = get('https://data.sec.gov/submissions/' + f['name']); time.sleep(.12)
                if rr is not None: blocks.append(rr.json())
            for b in blocks:
                for fd, form, acc, pd_ in zip(b['filingDate'], b['form'], b['accessionNumber'], b['primaryDocument']):
                    if form in ('10-K', '10-Q'): out.append((fd, form, acc, pd_))
    cache['sub'][tk] = out; return out
def words(tk, acc, doc):
    c = cikm[tk]; r = get(f"https://www.sec.gov/Archives/edgar/data/{c}/{acc.replace('-', '')}/{doc}"); time.sleep(.12)
    if r is None: return None
    t = re.sub(r'<[^>]+>', ' ', r.text.lower()); return collections.Counter(re.findall(r'[a-z]{3,}', t))
def cos(a, b):
    k = set(a) & set(b); num = sum(a[w] * b[w] for w in k); return num / (np.sqrt(sum(v * v for v in a.values())) * np.sqrt(sum(v * v for v in b.values())))
def pair(tk, d):
    f = [(pd.Timestamp(x[0]), *x[1:]) for x in filings(tk)]
    cur = [x for x in f if d - pd.Timedelta(days=90) <= x[0] <= d]
    if not cur: return None
    cur = max(cur, key=lambda x: x[0]); prv = [x for x in f if x[1] == cur[1] and abs((x[0] - (cur[0] - pd.Timedelta(days=365))).days) <= 45]
    if not prv: return None
    prv = min(prv, key=lambda x: abs((x[0] - (cur[0] - pd.Timedelta(days=365))).days)); key = (tk, cur[2], prv[2])
    if key not in cache['sim']:
        a = words(tk, cur[2], cur[3]); b = words(tk, prv[2], prv[3]); cache['sim'][key] = cos(a, b) if a and b else np.nan
    return cache['sim'][key], cur[0]
t0 = time.time(); sims = []
for i, (tk, d) in enumerate(zip(G.tk, G.d)):
    try: p = pair(tk, d)
    except Exception as e: p = None
    sims.append(p[0] if p else np.nan)
    if i % 25 == 0: pickle.dump(cache, open(CK, 'wb')); print(f'  {i}/{len(G)} {time.time() - t0:.0f}s', flush=True)
    if time.time() - t0 > 540: print('  time budget hit; rerun resumes from cache', flush=True); break
pickle.dump(cache, open(CK, 'wb')); G = G.iloc[:len(sims)].copy(); G['sim'] = sims; G = G.dropna(subset=['sim']); G['mo'] = G.d.dt.to_period('M').astype(str)
st = lambda x: f'n {len(x):4d} REACH {(x.PEAK >= .1).mean():.0%} win {(x.r > 0).mean():.0%} mean {x.r.mean():+6.1%} med {x.r.median():+.1%} p10 {x.r.quantile(.1):+.0%}' if len(x) else 'n 0'
print(f'\nlive sub-$5 group test flips with a text pair: {len(G)} of {len(sims)} tried | similarity quartiles {np.round(G.sim.quantile([.25, .5, .75]).values, 3)}'); lo, hi = G.sim.quantile([1 / 3, 2 / 3]); C = G[G.sim <= lo]; Sm = G[G.sim >= hi]
ma = Sm.groupby('mo').r.mean(); mb = C.groupby('mo').r.mean(); j = pd.concat([ma, mb], axis=1, keys=['a', 'b']).dropna(); d = j.a - j.b; t = d.mean() / (d.std() / np.sqrt(len(d))) if len(d) > 2 else np.nan
print(f'  CHANGED (lowest tercile, sim <= {lo:.3f}): {st(C)}\n  MIDDLE: {st(G[(G.sim > lo) & (G.sim < hi)])}\n  SAME (top tercile, sim >= {hi:.3f}): {st(Sm)}\n  SAME - CHANGED mean gap {d.mean():+.1%} t {t:+.2f} ({len(d)} months)')
for y, nm in ((2022, '2022'), (2023, '2023-26')):
    f_ = (lambda x: x[x.d.dt.year == 2022]) if y == 2022 else (lambda x: x[x.d.dt.year >= 2023]); print(f'  {nm}: CHANGED {f_(C).r.mean():+.1%} (n {len(f_(C))}) vs SAME {f_(Sm).r.mean():+.1%} (n {len(f_(Sm))})')
print('  corr(sim, r):', round(G.sim.corr(G.r, method='spearman'), 3), '| corr(sim, REACH):', round(G.sim.corr((G.PEAK >= .1).astype(float), method='spearman'), 3))
