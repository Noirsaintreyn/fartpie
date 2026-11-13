import requests, pickle, yfinance as yf, pandas as pd, warnings, time; warnings.filterwarnings('ignore')
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
s=requests.Session(); s.headers['User-Agent']='Mozilla/5.0'; s.get('https://fc.yahoo.com',timeout=15); cr=s.get('https://query1.finance.yahoo.com/v1/test/getcrumb',timeout=15).text
ex=[{"operator":"EQ","operands":["exchange",e]} for e in ("NMS","NYQ","ASE","NCM")]
T=[]
for off in range(0,2500,250):
    b={"size":250,"offset":off,"sortField":"intradayprice","sortType":"ASC","quoteType":"EQUITY","query":{"operator":"AND","operands":[{"operator":"or","operands":ex},{"operator":"BTWN","operands":["intradayprice",0.3,20]},{"operator":"GT","operands":["avgdailyvol3m",100000]}]}}
    q=s.post('https://query1.finance.yahoo.com/v1/finance/screener?crumb='+cr,json=b,timeout=30).json()['finance']['result'][0]['quotes']
    if not q: break
    T+=[x['symbol'] for x in q]
T=sorted(set(T)); print('tickers',len(T))
out={}
for i in range(0,len(T),150):
    try:
        d=yf.download(T[i:i+150],period='2y',interval='1d',group_by='ticker',auto_adjust=False,progress=False,threads=True)
        for t in T[i:i+150]:
            try:
                x=d[t].dropna(subset=['Close']); 
                if len(x)>200: out[t]=x
            except Exception: pass
    except Exception as e: print('chunk err',i,e)
    print(i,len(out),flush=True)
pickle.dump(out,open(SC+'lowpx_daily.pkl','wb')); print('saved',len(out))
