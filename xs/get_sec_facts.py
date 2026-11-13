"""Pull SEC companyfacts for the PIT S&P ticker->CIK map and keep only the fields needed for a point-in-time per-share value anchor: annual revenue, net income, diluted weighted shares, stockholders equity, with filed dates. Saves xs/data/sec_facts.parquet (ticker, tag, start, end, filed, form, fy, fp, val)."""
import json, time, requests, pandas as pd, concurrent.futures as cf
M=json.load(open('/home/claude/fartpie/sp500_ticker_cik_map.json'))
TAGS={'rev':['Revenues','RevenueFromContractWithCustomerExcludingAssessedTax','SalesRevenueNet','RevenueFromContractWithCustomerIncludingAssessedTax'],
      'ni':['NetIncomeLoss'],'sh':['WeightedAverageNumberOfDilutedSharesOutstanding'],'eq':['StockholdersEquity']}
H={'User-Agent':'research-client'}
def get(tk):
    cik=M[tk]; url=f'https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json'
    for a in range(3):
        try:
            r=requests.get(url,headers=H,timeout=60)
            if r.status_code==404: return tk,[]
            r.raise_for_status(); f=r.json().get('facts',{}).get('us-gaap',{}); rows=[]
            for k,tags in TAGS.items():
                for tg in tags:
                    for unit,arr in f.get(tg,{}).get('units',{}).items():
                        for x in arr: rows.append((tk,k,tg,x.get('start'),x['end'],x['filed'],x.get('form'),x.get('fy'),x.get('fp'),x['val']))
            time.sleep(.12); return tk,rows
        except Exception as e: time.sleep(2)
    return tk,None
out=[]; bad=[]
with cf.ThreadPoolExecutor(6) as ex:
    for i,(tk,rows) in enumerate(ex.map(get,sorted(M))):
        if rows is None: bad.append(tk)
        else: out+=rows
        if i%50==0: print(i,len(out),len(bad),flush=True)
df=pd.DataFrame(out,columns=['tk','kind','tag','start','end','filed','form','fy','fp','val']); df.to_parquet('/home/claude/fartpie/xs/data/sec_facts.parquet'); print('saved',df.shape,'bad',bad[:20],len(bad))
