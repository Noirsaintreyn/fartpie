"""Pull the full filing-form history (form, filingDate) from EDGAR submissions for every ticker in flip_history.csv. Output /tmp scratchpad parquet. Run time ~2 min (10 req/s polite)."""
import requests, pandas as pd, json, time, sys
H = {'User-Agent': 'quant-research research@example.com'}
out = sys.argv[1]
tick = json.load(open('data/fund/company_tickers.json')); m = {v['ticker'].upper(): str(v['cik_str']).zfill(10) for v in (tick.values() if isinstance(tick, dict) else tick)}
tks = sorted(set(pd.read_csv('flip_history.csv').tk)); rows = []; miss = []
def take(j, tk):
    for f, form, it in zip(j['filingDate'], j['form'], j.get('items', [''] * len(j['form']))): rows.append((tk, f, form, it))
for tk in tks:
    cik = m.get(tk.upper().replace('-', '.')) or m.get(tk.upper())
    if not cik: miss.append(tk); continue
    try:
        j = requests.get(f'https://data.sec.gov/submissions/CIK{cik}.json', headers=H, timeout=30).json(); take(j['filings']['recent'], tk)
        for fx in j['filings'].get('files', []):
            time.sleep(.12); take(requests.get('https://data.sec.gov/submissions/' + fx['name'], headers=H, timeout=30).json(), tk)
    except Exception as e: miss.append(tk + ':' + str(e)[:30])
    time.sleep(.12)
D = pd.DataFrame(rows, columns=['tk', 'filed', 'form', 'items']); D.to_parquet(out); print(len(D), D.tk.nunique(), 'missing', len(miss), miss[:10]); print(D.form.value_counts().head(15))
