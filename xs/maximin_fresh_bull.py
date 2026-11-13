"""Fresh-bull scan: names whose 15m VECTOR (xs/vector_pine.run_state_machine) flipped to BULL within the last N 15m bars AND whose price is above the 15m Maximin pivot (L=20) and above the vector level. Sessions: regular bars only vs extended (Yahoo 04:00-20:00). Last 15m bar is still forming."""
import sys, json, warnings, numpy as np, pandas as pd, yfinance as yf; warnings.filterwarnings('ignore'); sys.path.insert(0,'/home/claude/fartpie/xs')
from vector_pine import run_state_machine
L=20; NF=int(sys.argv[1]) if len(sys.argv)>1 else 6
sp=json.load(open('/home/claude/fartpie/sp500_tickers.json')); sp=list(sp.keys()) if isinstance(sp,dict) else [s if isinstance(s,str) else s.get('ticker') for s in sp]
ms=[l.strip() for l in open('/home/claude/fartpie/midsmall_universe.txt') if l.strip() and not l.startswith('#')]
U=sorted(set(t.replace('.','-') for t in sp+ms)); D={}
for i in range(0,len(U),100):
    try:
        d=yf.download(U[i:i+100],period='5d',interval='5m',prepost=True,group_by='ticker',auto_adjust=False,progress=False,threads=True)
        for t in U[i:i+100]:
            try:
                x=d[t].dropna(subset=['Close'])
                if len(x)>120: D[t]=x
            except Exception: pass
    except Exception as e: print('err',i,e)
rows=[]
for t,x in D.items():
    x=x.rename(columns=str.lower); x.index=pd.to_datetime(x.index); x.index=(x.index.tz_convert('America/New_York') if x.index.tz else x.index.tz_localize('UTC').tz_convert('America/New_York')).tz_localize(None); x=x[['open','high','low','close','volume']].astype(float)
    for sess in ('regular','extended'):
        y=x.between_time('09:30','15:59') if sess=='regular' else x.between_time('04:00','19:59')
        z=y.resample('15min',origin='start_day').agg(dict(open='first',high='max',low='min',close='last',volume='sum')).dropna(subset=['close'])
        if len(z)<60: continue
        reg,lev,_,_=run_state_machine(z.assign(datetime=z.index)); reg=np.asarray(reg); lev=np.asarray(lev,float)
        lb=z.low.rolling(L).min().rolling(L).max().values
        n=len(z)
        if reg[-1]!=1: continue
        k=n-1
        while k>0 and reg[k-1]==1: k-=1       # first bar of the current bull run
        age=n-1-k                              # 0 = flipped on the latest bar
        hi=z.high.max(); lo=z.low.min(); fc=z.close.iloc[k]; pos=(fc-lo)/(hi-lo) if hi>lo else np.nan; dd=fc/hi-1; rows.append(dict(tkr=t,sess=sess,age=age,pos=pos,dd=dd,flip_time=z.index[k],t=z.index[-1],close=z.close.iloc[-1],lvl=lev[-1],lb=lb[-1],above_lb=z.close.iloc[-1]>lb[-1],prior_bear=(k>0 and reg[k-1]==-1)))
R=pd.DataFrame(rows); R.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/maximin_fresh.pkl'); print('latest 15m bar',R.t.max(),'names',len(D))
for sess in ('regular','extended'):
    s=R[(R.sess==sess)&(R.age<=NF)&R.above_lb&(R.close>R.lvl)].sort_values(['age','tkr'])
    print(f'\n== {sess}: 15m vector BULL for <= {NF} bars and close > 15m Maximin pivot ({len(s)})')
    for a,g in s.groupby('age'):
        print(f'  age {a} (flipped {g.flip_time.iloc[0]:%H:%M}): '+', '.join(f'{r.tkr} {r.close:.2f}{"" if r.prior_bear else "*"}' for r in g.itertuples()))
print('\n* = no prior bear in the window (bull since the start of available data)')

print('\n=== FLIPPED AT A RELATIVE LOW: fresh 15m bull (<= %d bars), close > 15m pivot, flip-bar close in the bottom third of the 5-day range AND >= 3%% under the 5-day high ===' % NF)
for sess in ('regular','extended'):
    s=R[(R.sess==sess)&(R.age<=NF)&R.above_lb&(R.close>R.lvl)&(R.pos<=.34)&(R.dd<=-.03)&R.prior_bear].sort_values(['pos'])
    print(f'  {sess} ({len(s)}): '+', '.join(f'{r.tkr} {r.close:.2f} (flip {r.flip_time:%H:%M}, range pos {r.pos:.0%}, {r.dd:.0%} off high)' for r in s.itertuples()))
