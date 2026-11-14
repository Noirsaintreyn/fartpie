"""[13D/13G version] All 8-K filings (items + acceptance time) since 2018 from SEC EDGAR submissions (recent + older pages) for the wide daily universe. Output scratchpad/ext/13d_wide.csv"""
import requests, pandas as pd, time, json
H = {'User-Agent': 'quant-research-bot research@example.com'}; OUT = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'
D = pd.read_pickle(OUT + 'wide_daily_ohlcv.pkl'); tick = requests.get('https://www.sec.gov/files/company_tickers.json', headers=H, timeout=30).json(); m = {v['ticker'].upper(): str(v['cik_str']).zfill(10) for v in tick.values()}; rows = []; nomap = 0; FORMS = {'SC 13D', 'SC 13D/A', 'SCHEDULE 13D', 'SCHEDULE 13D/A', 'SC 13G', 'SC 13G/A', 'SCHEDULE 13G', 'SCHEDULE 13G/A'}
def take(r, tk):
    for f, a, it, form in zip(r['filingDate'], r['acceptanceDateTime'], r['items'], r['form']):
        if form == '8-K' and f >= '2018-01-01': rows.append((tk, f, a, it))
def get(u):
    for _ in range(3):
        try:
            r = requests.get(u, headers=H, timeout=30)
            if r.status_code == 200: return r.json()
            if r.status_code == 429: time.sleep(2)
        except Exception: time.sleep(1)
    return None
import concurrent.futures as cf, threading
lock = threading.Lock()
def work(tk):
    local = []
    cik = m.get(tk.upper().replace('-', '.')) or m.get(tk.upper())
    if not cik: return tk, local, True
    def take2(r):
        for f, a, it, form in zip(r['filingDate'], r['acceptanceDateTime'], r['items'], r['form']):
            if form in FORMS and f >= '2018-01-01': local.append((tk, f, a, form))
    j = get(f'https://data.sec.gov/submissions/CIK{cik}.json'); time.sleep(0.6)
    if not j: return tk, local, False
    take2(j['filings']['recent'])
    for f in j['filings'].get('files', []):
        if f['filingTo'] >= '2018-01-01':
            x = get('https://data.sec.gov/submissions/' + f['name']); time.sleep(0.6)
            if x: take2(x)
    return tk, local, False
done_n = 0
with cf.ThreadPoolExecutor(6) as ex:
    for tk, local, nm_ in ex.map(work, list(D)):
        rows.extend(local); nomap += nm_; done_n += 1
        if done_n % 200 == 0: print(done_n, len(rows), flush=True)
for k, tk in []:
    cik = m.get(tk.upper().replace('-', '.')) or m.get(tk.upper())
    if not cik: nomap += 1; continue
    j = get(f'https://data.sec.gov/submissions/CIK{cik}.json')
    if not j: continue
    take(j['filings']['recent'], tk); time.sleep(0.11)
    for f in j['filings'].get('files', []):
        if f['filingTo'] >= '2018-01-01':
            x = get('https://data.sec.gov/submissions/' + f['name'])
            if x: take(x, tk)
            time.sleep(0.11)
    if k % 200 == 0: print(k, len(rows), flush=True)
pd.DataFrame(rows, columns=['ticker', 'filed', 'accept', 'form']).to_csv(OUT + '13d_wide.csv', index=False); print('done', len(rows), 'unmapped', nomap)
