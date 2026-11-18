"""Pull SEC companyfacts (point-in-time via 'filed') for the 753-name mid/small/sub-$5 universe. Free, no key. 5 req/s, SEC fair-access UA."""
import json, os, time, requests
U = [l.strip() for l in open('/mnt/user-data/uploads/fartpie/midsmall_universe.txt') if l.strip() and not l.startswith('#')]
M = {v['ticker']: v['cik_str'] for v in json.load(open('data/fund/company_tickers.json')).values()}
H = {'User-Agent': 'research rey@example.com'}
for t in U:
    f = f'data/fund/sec_small/{t}.json'
    if os.path.exists(f) or t not in M: continue
    for a in range(4):
        r = requests.get(f'https://data.sec.gov/api/xbrl/companyfacts/CIK{M[t]:010d}.json', headers=H, timeout=60)
        if r.status_code == 200: open(f, 'wb').write(r.content); break
        if r.status_code == 404: break
        time.sleep(5 * (a + 1))
    print(t, r.status_code, flush=True); time.sleep(0.25)
