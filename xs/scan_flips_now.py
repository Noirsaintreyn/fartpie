"""Live scan: names whose 1H vector flipped bull on a 1H bar that has CLOSED today. 1H bars on the clock hour like TradingView (09:30-10:00, 10:00-11:00, ...) from Yahoo 30m bars (60 days = the most Yahoo serves); forming bar dropped. A flip is kept only if it shows with BOTH the full 60-day warm-up and a 30-day warm-up (the ratchet is path-dependent, so a short history can put a flip on the wrong bar). Adds where price sits in its 60-day range. Universes: S&P 500 list and the sub-$5 set."""
import json, pickle, numpy as np, pandas as pd, yfinance as yf, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
E='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'
sp=json.load(open('/home/claude/fartpie/sp500_tickers.json')); sp=list(sp.keys()) if isinstance(sp,dict) else [s if isinstance(s,str) else s.get('ticker') for s in sp]
low=list(pickle.load(open(E+'lowpx_1h_3y.pkl','rb')).keys()); now=pd.Timestamp.now(tz='America/New_York'); print('now',now.strftime('%Y-%m-%d %H:%M ET'),flush=True); res=[]; ok=0
def flips(h):
    reg=np.asarray(run_state_machine(h.assign(datetime=h.index.tz_localize(None)))[0]); return reg
for nm,tks in (('S&P 500',sp),('sub-$5 set',low)):
    for i in range(0,len(tks),200):
        ch=[t.replace('.','-') for t in tks[i:i+200]]
        try: H=yf.download(ch,period='60d',interval='30m',group_by='ticker',auto_adjust=False,progress=False,threads=True)
        except Exception as e: continue
        for t in ch:
            try:
                x=H[t].dropna(); x.columns=[c.lower() for c in x.columns]; x.index=x.index.tz_convert('America/New_York'); m=x.index.hour*60+x.index.minute; x=x[(m>=570)&(m<960)]
                h=x.groupby(x.index.floor('h')).agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('volume','sum')); h=h[h.index+pd.Timedelta(hours=1)<=now]
                if len(h)<250 or h.index[-1].date()!=now.date(): continue
                ok+=1; r60=flips(h); h30=h.iloc[len(h)//2:]; r30=flips(h30); j=len(h)-1; j30=len(h30)-1
                f60=r60[j]==1 and r60[j-1]!=1; f30=r30[j30]==1 and r30[j30-1]!=1
                if f60 or f30:
                    # bars since the state last changed under each warm-up
                    hi,lo=h.high.max(),h.low.min(); c=float(h.close.iloc[-1]); res.append((nm,t,h.index[-1].strftime('%H:%M')+'-'+(h.index[-1]+pd.Timedelta(hours=1)).strftime('%H:%M'),round(c,2),'both' if (f60 and f30) else ('60d only' if f60 else '30d only'),round((c-lo)/(hi-lo)*100),round((c/lo-1)*100),round((c/hi-1)*100)))
            except Exception: pass
R=pd.DataFrame(res,columns=['universe','ticker','bar','close','agree','pct_of_60d_range','above_60d_low_%','below_60d_high_%']); print('names scanned',ok); print(R.sort_values(['agree','pct_of_60d_range']).to_string(index=False) if len(R) else 'no bull flips on the last completed 1H bar')
