"""CUSIP -> ticker for the 13F holdings. Step 1: exact normalized issuer-name match to SEC company_tickers (universe tickers only). Step 2: OpenFIGI (no key, 25 req/min x 10 ids) for unmatched CUSIPs, ordered by number of fund-quarters held. Cached to ext/cusip_map.pkl (resumable)."""
import pandas as pd, requests, re, time, os, pickle
X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; A = pd.read_pickle(X + 'f13_holdings.pkl'); A = A[A.putcall.isna()]
D = pd.read_pickle(X + 'wide_daily_ohlcv.pkl'); U = set(D.keys())
def norm(s): s = re.sub(r'[^A-Z0-9 ]', ' ', str(s).upper()); s = re.sub(r'\b(INC|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED|PLC|HOLDINGS|HLDGS|GROUP|CL|CLASS|COM|NEW|THE|DEL|DE|NV|SA|LLC|LP|ORD|SHS|A|B|C)\b', ' ', s); return ' '.join(s.split())
ct = requests.get('https://www.sec.gov/files/company_tickers.json', headers={'User-Agent': 'quant-research-bot research-script'}).json(); nm = {}
for v in ct.values():
    if v['ticker'] in U: nm.setdefault(norm(v['title']), v['ticker'])
M = pickle.load(open(X + 'cusip_map.pkl', 'rb')) if os.path.exists(X + 'cusip_map.pkl') else {}
cnt = A.groupby('cusip').size().sort_values(ascending=False); iss = A.groupby('cusip').issuer.first()
for c in cnt.index:
    if c not in M and norm(iss[c]) in nm: M[c] = nm[norm(iss[c])]
print('name-matched', len(M), 'of', len(cnt), flush=True)
todo = [c for c in cnt.index if c not in M][:int(os.environ.get('FIGI_MAX', 3000))]
for i in range(0, len(todo), 10):
    b = todo[i:i + 10]
    for _ in range(5):
        r = requests.post('https://api.openfigi.com/v3/mapping', json=[{'idType': 'ID_CUSIP', 'idValue': c} for c in b], timeout=60)
        if r.status_code == 429: time.sleep(30); continue
        break
    if r.status_code != 200: continue
    for c, res in zip(b, r.json()):
        t = [d['ticker'] for d in res.get('data', []) if d.get('securityType') == 'Common Stock' and d.get('exchCode') == 'US']
        M[c] = t[0] if t and t[0] in U else ''
    time.sleep(2.6); 
    if (i // 10) % 20 == 0: pickle.dump(M, open(X + 'cusip_map.pkl', 'wb')); print(i, len(M), flush=True)
pickle.dump(M, open(X + 'cusip_map.pkl', 'wb')); print('mapped', sum(1 for v in M.values() if v), 'of', len(M))
