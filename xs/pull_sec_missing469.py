import json, os, time, requests
U = [l.strip() for l in open('/tmp/miss469.txt') if l.strip()]
M = {v['ticker']: v['cik_str'] for v in json.load(open('data/fund/company_tickers.json')).values()}
H = {'User-Agent': 'research rey@example.com'}
n = 0
for t in U:
    f = f'data/fund/sec_small/{t}.json'
    if os.path.exists(f): continue
    for a in range(3):
        r = requests.get(f'https://data.sec.gov/api/xbrl/companyfacts/CIK{M[t]:010d}.json', headers=H, timeout=60)
        if r.status_code == 200: open(f, 'wb').write(r.content); n += 1; break
        if r.status_code == 404: break
        time.sleep(4 * (a + 1))
    time.sleep(0.25)
print('saved', n, 'of', len(U))
