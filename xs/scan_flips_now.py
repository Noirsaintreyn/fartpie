"""Live scan: names whose 1H vector flipped bull on a 1H bar that has CLOSED today. 1H bars are built on the clock hour like TradingView (09:30-10:00, 10:00-11:00, ...) from Yahoo 30m bars; the forming bar is dropped. Universes: S&P 500 list and the 299 sub-$5 names. Yahoo intraday can lag ~15 min."""
import json, pickle, numpy as np, pandas as pd, yfinance as yf, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
E='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'
sp=json.load(open('/home/claude/fartpie/sp500_tickers.json')); sp=list(sp.keys()) if isinstance(sp,dict) else [s if isinstance(s,str) else s.get('ticker') for s in sp]
low=list(pickle.load(open(E+'lowpx_1h_3y.pkl','rb')).keys()); now=pd.Timestamp.now(tz='America/New_York'); print('now',now.strftime('%Y-%m-%d %H:%M ET'),'| S&P',len(sp),'sub-$5',len(low),flush=True); res=[]; ok=0
for nm,tks in (('S&P 500',sp),('sub-$5',low)):
    for i in range(0,len(tks),200):
        ch=[t.replace('.','-') for t in tks[i:i+200]]
        try: H=yf.download(ch,period='60d',interval='30m',group_by='ticker',auto_adjust=False,progress=False,threads=True)
        except Exception as e: print('err',str(e)[:80]); continue
        for t in ch:
            try:
                x=H[t].dropna(); x.columns=[c.lower() for c in x.columns]; x.index=x.index.tz_convert('America/New_York'); m=x.index.hour*60+x.index.minute; x=x[(m>=570)&(m<960)]
                h=x.groupby(x.index.floor('h')).agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('volume','sum'))
                h=h[h.index+pd.Timedelta(hours=1)<=now]
                if len(h)<150 or h.index[-1].date()!=now.date(): continue
                ok+=1; reg=np.asarray(run_state_machine(h.assign(datetime=h.index.tz_localize(None)))[0])
                for j in np.where(h.index.date==now.date())[0]:
                    if reg[j]==1 and reg[j-1]!=1: res.append((nm,t,(h.index[j]+pd.Timedelta(hours=1)).strftime('%H:%M'),round(float(h.close.iloc[j]),2),'bull'))
                    if reg[j]==-1 and reg[j-1]!=-1: res.append((nm,t,(h.index[j]+pd.Timedelta(hours=1)).strftime('%H:%M'),round(float(h.close.iloc[j]),2),'bear'))
            except Exception: pass
print('names with a completed 1H bar today:',ok)
R=pd.DataFrame(res,columns=['universe','ticker','bar_close','close_px','flip']); print(R.sort_values(['flip','universe','ticker']).to_string(index=False) if len(R) else 'no flips on completed 1H bars today'); R.to_csv('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/flips_now.csv',index=False)
