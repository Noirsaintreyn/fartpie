"""Point-in-time shares outstanding for the wide universe from SEC companyfacts (dei:EntityCommonStockSharesOutstanding cover-page counts, filed dates). Output scratch ext/shares_pit.pkl: tk, end, filed, val. Ticker->CIK from SEC company_tickers.json (current listings only)."""
import requests, time, pandas as pd, concurrent.futures as cf
X = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; H = {'User-Agent': 'quant-research-bot research-script'}
U = set(pd.read_pickle(X + 'wide_daily_ohlcv.pkl').keys()); ct = requests.get('https://www.sec.gov/files/company_tickers.json', headers=H, timeout=60).json(); M = {v['ticker']: v['cik_str'] for v in ct.values() if v['ticker'] in U}; print('universe', len(U), 'with cik', len(M), flush=True)
def get(tk):
    for _ in range(3):
        try:
            r = requests.get(f'https://data.sec.gov/api/xbrl/companyfacts/CIK{int(M[tk]):010d}.json', headers=H, timeout=90)
            if r.status_code == 404: return tk, []
            r.raise_for_status(); f = r.json().get('facts', {}); rows = []
            for tg in ('EntityCommonStockSharesOutstanding',):
                for unit, arr in f.get('dei', {}).get(tg, {}).get('units', {}).items():
                    for x in arr: rows.append((tk, x['end'], x['filed'], x['val']))
            time.sleep(.1); return tk, rows
        except Exception: time.sleep(2)
    return tk, None
out = []; bad = []
with cf.ThreadPoolExecutor(4) as ex:
    for k, (tk, rows) in enumerate(ex.map(get, list(M))):
        if rows is None: bad.append(tk)
        else: out += rows
        if k % 100 == 0: print(k, len(out), flush=True)
pd.DataFrame(out, columns=['tk', 'end', 'filed', 'val']).to_pickle(X + 'shares_pit.pkl'); print('saved', len(out), 'bad', len(bad))
