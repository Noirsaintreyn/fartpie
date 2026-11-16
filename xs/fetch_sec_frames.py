"""Fetch SEC XBRL 'frames' (one call = every filer's value for one concept at one quarter-end) for balance-sheet concepts, 2018Q2..2026Q3, cached as a long table. Caveat: frames carry the latest reported value for each period-end (restatements included), so availability is lagged 90 days after the period end in the analysis."""
import requests, time, json, os, pandas as pd
H={'User-Agent':'quant-research-bot research-script'}; OUT='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'
C=[('us-gaap','Liabilities','USD'),('us-gaap','LiabilitiesAndStockholdersEquity','USD'),('us-gaap','StockholdersEquity','USD'),('us-gaap','AssetsCurrent','USD'),('us-gaap','Assets','USD'),('us-gaap','CashAndCashEquivalentsAtCarryingValue','USD'),('dei','EntityCommonStockSharesOutstanding','shares')]
Q=[f'CY{y}Q{q}I' for y in range(2018,2027) for q in (1,2,3,4)][1:-1]; rows=[]
for tax,tag,u in C:
    for q in Q:
        try:
            r=requests.get(f'https://data.sec.gov/api/xbrl/frames/{tax}/{tag}/{u}/{q}.json',headers=H,timeout=40)
            if r.status_code!=200: print(tag,q,r.status_code); time.sleep(.3); continue
            for x in r.json()['data']: rows.append((tag,x['cik'],x['end'],x['val']))
        except Exception as e: print(tag,q,str(e)[:60])
        time.sleep(.15)
    print(tag,len(rows),flush=True)
F=pd.DataFrame(rows,columns=['tag','cik','end','val']); F.to_pickle(OUT+'sec_frames.pkl')
t=requests.get('https://www.sec.gov/files/company_tickers.json',headers=H,timeout=30).json(); json.dump({v['ticker'].upper():int(v['cik_str']) for v in t.values()},open(OUT+'tk_cik.json','w')); print('done',len(F))
