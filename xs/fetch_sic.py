"""SIC industry code + description per ticker from SEC submissions (one request each). Output scratchpad/ext/sic.csv"""
import requests, pandas as pd, time, concurrent.futures as cf
H = {'User-Agent': 'quant-research-bot research@example.com'}; OUT = '/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'
D = pd.read_pickle(OUT + 'wide_daily_ohlcv.pkl'); tick = requests.get('https://www.sec.gov/files/company_tickers.json', headers=H, timeout=30).json(); m = {v['ticker'].upper(): str(v['cik_str']).zfill(10) for v in tick.values()}
def work(tk):
    cik = m.get(tk.upper().replace('-', '.')) or m.get(tk.upper())
    if not cik: return None
    for _ in range(3):
        try:
            r = requests.get(f'https://data.sec.gov/submissions/CIK{cik}.json', headers=H, timeout=30); time.sleep(0.6)
            if r.status_code == 200: j = r.json(); return (tk, j.get('sic'), j.get('sicDescription'))
        except Exception: time.sleep(1)
    return None
with cf.ThreadPoolExecutor(6) as ex: res = [x for x in ex.map(work, list(D)) if x]
pd.DataFrame(res, columns=['tk', 'sic', 'desc']).to_csv(OUT + 'sic.csv', index=False); print('done', len(res))
