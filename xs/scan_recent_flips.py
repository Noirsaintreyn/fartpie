"""Live scan: recent 1H bull flips that are still bull and not yet extended, ranked by the cuts that held in research (reports 221-223): daily ATRP in the top third (>= 7.25%; middle third >= 4.86%), flip-bar volume in the top third (>= VOLX cut), daily vector bull. Insider data ends May 2026 so it is not used. 1H bars on the clock hour from Yahoo 30m bars (60 days), forming bar dropped. 'Recent' = flip within the last 14 completed 1H bars (two sessions). 'Not extended' = last price no more than 5% above the flip close."""
import json, pickle, sys, numpy as np, pandas as pd, yfinance as yf, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
E='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; VC=float(sys.argv[1]) if len(sys.argv)>1 else 1.5
sp=json.load(open('/home/claude/fartpie/sp500_tickers.json')); sp=list(sp.keys()) if isinstance(sp,dict) else [s if isinstance(s,str) else s.get('ticker') for s in sp]
low=list(pickle.load(open(E+'lowpx_1h_3y.pkl','rb')).keys()); now=pd.Timestamp.now(tz='America/New_York'); print('now',now.strftime('%Y-%m-%d %H:%M ET'),flush=True); res=[]
for nm,tks in (('small-cap set',low),('S&P 500',sp)):
    for i in range(0,len(tks),200):
        ch=[t.replace('.','-') for t in tks[i:i+200]]
        try: H=yf.download(ch,period='60d',interval='30m',group_by='ticker',auto_adjust=False,progress=False,threads=True)
        except Exception: continue
        for t in ch:
            try:
                x=H[t].dropna(); x.columns=[c.lower() for c in x.columns]; x.index=x.index.tz_convert('America/New_York'); m=x.index.hour*60+x.index.minute; x=x[(m>=570)&(m<960)]; last=float(x.close.iloc[-1])
                h=x.groupby(x.index.floor('h')).agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('volume','sum')); h=h[h.index+pd.Timedelta(hours=1)<=now]
                if len(h)<250 or h.index[-1].date()!=now.date(): continue
                out=run_state_machine(h.assign(datetime=h.index.tz_localize(None))); reg=np.asarray(out[0]); lev=np.asarray(out[1],float); n=len(h)
                if reg[-1]!=1: continue
                j=n-1
                while j>0 and reg[j-1]==1: j-=1
                if n-1-j>13: continue
                d=x.groupby(x.index.normalize()).agg(o=('open','first'),h=('high','max'),l=('low','min'),c=('close','last')); pc=d.c.shift(1); tr=pd.concat([d.h-d.l,(d.h-pc).abs(),(d.l-pc).abs()],axis=1).max(axis=1); atrp=float((tr.rolling(14).mean()/d.c).iloc[-2])
                dreg=np.asarray(run_state_machine(d.rename(columns={'o':'open','h':'high','l':'low','c':'close'}).assign(volume=1,datetime=d.index.tz_localize(None)))[0]); volx=float(h.volume.iloc[j]/max(h.volume.iloc[max(0,j-100):j].mean(),1)); fc=float(h.close.iloc[j]); ema=float(h.close.ewm(span=20).mean().iloc[-1])
                score=int(atrp>=.0725)+int(volx>=VC)+int(dreg[-2]==1)
                res.append(dict(universe=nm,ticker=t,flip_bar_close=(h.index[j]+pd.Timedelta(hours=1)).strftime('%a %H:%M'),bars_ago=n-1-j,flip_px=round(fc,2),last=round(last,2),from_flip_pct=round((last/fc-1)*100,1),stop_level=round(float(lev[-1]),2),risk_pct=round((last/lev[-1]-1)*100,1),atrp_pct=round(atrp*100,1),vol_x=round(volx,1),daily='bull' if dreg[-2]==1 else 'bear',vs_ema20_pct=round((last/ema-1)*100,1),score=score))
            except Exception: pass
R=pd.DataFrame(res); R=R[(R.from_flip_pct<=5)&(R.from_flip_pct>=-5)]; print('recent bull flips, still bull, within 5% of the flip:',len(R))
pd.set_option('display.width',250); top=R[R.atrp_pct>=4.86].sort_values(['score','atrp_pct'],ascending=False); print(top.head(25).to_string(index=False)); R.to_csv('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/recent_flips.csv',index=False)
