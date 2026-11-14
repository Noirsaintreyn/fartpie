"""Pull 8-K filings (items + acceptance time) from SEC EDGAR submissions for every ticker in live/midsmall_bars.pkl (2025-01 onward). Output xs/ext/8k_items_midsmall.csv in the same format as 8k_items.csv."""
import requests, pandas as pd, time, sys
H = {'User-Agent': 'quant-research-bot research@example.com'}
tick = requests.get('https://www.sec.gov/files/company_tickers.json', headers=H, timeout=30).json(); m = {v['ticker'].upper(): str(v['cik_str']).zfill(10) for v in tick.values()}
D = pd.read_pickle('live/midsmall_bars.pkl')['d']; rows = []; miss = []
for k, tk in enumerate(D):
    cik = m.get(tk.upper().replace('-', '.')) or m.get(tk.upper())
    if not cik: miss.append(tk); continue
    try:
        j = requests.get(f'https://data.sec.gov/submissions/CIK{cik}.json', headers=H, timeout=30).json(); r = j['filings']['recent']
        for f, a, it, form in zip(r['filingDate'], r['acceptanceDateTime'], r['items'], r['form']):
            if form == '8-K' and f >= '2025-01-01': rows.append(dict(cik=cik, ticker=tk, filed=f, accept=a, items=it))
    except Exception as e: miss.append(tk + ':' + str(e)[:30])
    time.sleep(0.12)
pd.DataFrame(rows).to_csv('/home/claude/fartpie/xs/ext/8k_items_midsmall.csv', index=False); print('names', len(D), 'filings', len(rows), 'tickers with filings', len({r['ticker'] for r in rows}), 'missing', len(miss), miss[:15])
