"""Maximin Dynamic Pivot (Rey's Pine): LB = highest(lowest(low,L),L), L=20, on 5m and 15m. 'Crossed' = ta.crossover/crossunder(close, LB) on the latest bar of that timeframe. Universe = S&P 500 + midsmall_universe.txt. Sessions: REGULAR (09:30-16:00 bars only) vs EXTENDED (Yahoo 04:00-20:00 pre+regular+post; Yahoo has no 20:00-04:00 overnight bars, so '24h' cannot be built from it)."""
import json, warnings, pickle, numpy as np, pandas as pd, yfinance as yf; warnings.filterwarnings('ignore')
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; L=20
sp=json.load(open('/home/claude/fartpie/sp500_tickers.json')); sp=list(sp.keys()) if isinstance(sp,dict) else [s if isinstance(s,str) else s.get('ticker') for s in sp]
ms=[l.strip() for l in open('/home/claude/fartpie/midsmall_universe.txt') if l.strip() and not l.startswith('#')]
U=sorted(set(t.replace('.','-') for t in sp+ms)); print('universe',len(U),flush=True)
D={}
for i in range(0,len(U),100):
    try:
        d=yf.download(U[i:i+100],period='5d',interval='5m',prepost=True,group_by='ticker',auto_adjust=False,progress=False,threads=True)
        for t in U[i:i+100]:
            try:
                x=d[t].dropna(subset=['Close'])
                if len(x)>60: D[t]=x
            except Exception: pass
    except Exception as e: print('err',i,e)
    print(i,len(D),flush=True)
pickle.dump(D,open(SC+'maximin_5m.pkl','wb'))
def lb(df):
    return df.low.rolling(L).min().rolling(L).max()
def cross(df):
    if len(df)<2*L+1: return None
    l=lb(df); c=df.close; 
    up=(c.iloc[-1]>l.iloc[-1]) and (c.iloc[-2]<=l.iloc[-2]); dn=(c.iloc[-1]<l.iloc[-1]) and (c.iloc[-2]>=l.iloc[-2])
    up2=(c.iloc[-2]>l.iloc[-2]) and (c.iloc[-3]<=l.iloc[-3]); 
    return dict(t=df.index[-1],close=float(c.iloc[-1]),lb=float(l.iloc[-1]),up=up,dn=dn,prev_up=up2)
rows=[]
for t,x in D.items():
    x=x.rename(columns=str.lower); x.index=pd.to_datetime(x.index); x.index=x.index.tz_convert('America/New_York') if x.index.tz else x.index.tz_localize('UTC').tz_convert('America/New_York')
    x=x[['open','high','low','close','volume']]
    for sess in ('regular','extended'):
        y=x.between_time('09:30','15:59') if sess=='regular' else x.between_time('04:00','19:59')
        for tf,rule in (('5m',None),('15m','15min')):
            z=y if rule is None else y.resample(rule,origin='start_day',closed='left',label='left').agg(dict(open='first',high='max',low='min',close='last',volume='sum')).dropna(subset=['close'])
            r=cross(z)
            if r: rows.append(dict(tkr=t,sess=sess,tf=tf,**r))
R=pd.DataFrame(rows); R.to_pickle(SC+'maximin_cross.pkl'); print('last bar overall',max(R.t)); 
for sess in ('regular','extended'):
    for tf in ('5m','15m'):
        s=R[(R.sess==sess)&(R.tf==tf)]; print(f'\n== {sess} {tf}: names {len(s)}, latest bar {s.t.max()} | crossed UP now {s.up.sum()} | DOWN now {s.dn.sum()} | crossed up on previous bar {s.prev_up.sum()}')
        u=s[s.up].copy(); u['gap%']=(u.close/u.lb-1)*100; print('  UP:', ', '.join(f'{a} {b:.2f}(lb {c:.2f})' for a,b,c in zip(u.tkr,u.close,u.lb)) or '-')
        print('  prev-bar UP:', ', '.join(s[s.prev_up].tkr) or '-')
        print('  DOWN:', ', '.join(s[s.dn].tkr) or '-')
