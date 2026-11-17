"""Report 252: mean-reversion CEILING test. DECLARED before results.
Day type (EX-POST, hindsight, ceiling only): RTH 09:30-16:00 efficiency ER = |close-open|/(high-low); MR day = bottom third of ER within the split (tertiles over all days), TREND = top third.
Generic reversion model (no tuning beyond the declared 2x2 grid): when price (5m close) is k x ATR14(daily RTH range) away from session VWAP (k in 0.75, 1.0), after 10:00 and before 15:00, enter AGAINST at next 5m open; stop = entry-side extreme + s x ATR beyond entry (s in 0.5, 1.0) ; target = VWAP; flat 15:55; one trade per day per direction; cost 1 pt.
Report net pts/trade and t by day type and by split (train 2019-21, test 2018 + 2022-26). If it loses on MR days even with perfect day-type knowledge = payoff problem; if it wins = identification problem.
Also: main algo daily pts by ER tercile (shareable250.pkl daily, deploy size, NQ pts)."""
import pandas as pd, numpy as np, pickle
df=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); df['dt']=pd.to_datetime(df.dt); df=df.set_index('dt').sort_index()
df=df.between_time('09:30','15:55'); df['d']=df.index.normalize()
days=[]; trades=[]
D=df.groupby('d')
dd=pd.DataFrame({'o':D.open.first(),'h':D.high.max(),'l':D.low.min(),'c':D.close.last(),'n':D.size()}); dd=dd[dd.n>=70]
dd['rng']=dd.h-dd.l; dd['er']=(dd.c-dd.o).abs()/dd.rng; dd['atr']=dd.rng.rolling(14).mean().shift(1)
dd['type']=pd.qcut(dd.er,3,labels=['MR','MID','TREND'])
for k in (0.75,1.0):
  for sm in (0.5,1.0):
    for d,g in D:
        if d not in dd.index or np.isnan(dd.atr[d]): continue
        a=dd.atr[d]; h=g.high.values;l=g.low.values;c=g.close.values;o=g.open.values;v=g.volume.values; t=g.index
        tp=(h+l+c)/3; vw=np.cumsum(tp*v)/np.maximum(np.cumsum(v),1)
        done={1:False,-1:False}
        for i in range(6,len(g)-2):
            if t[i].hour>=15: break
            dev=(c[i]-vw[i])/a
            for dr in (1,-1):   # dr=1 long fade when below VWAP
                if done[dr] or dev*(-dr)<k: continue
                e=o[i+1]; stop=(l[i] if dr==1 else h[i]) - dr*sm*a; done[dr]=True
                res=None
                for j in range(i+1,len(g)):
                    if dr==1:
                        if l[j]<=stop: res=stop-e;break
                        if h[j]>=vw[i]+0*1 and h[j]>=vw[j]: res=vw[j]-e;break
                    else:
                        if h[j]>=stop: res=e-stop;break
                        if l[j]<=vw[j]: res=e-vw[j];break
                if res is None: res=dr*(c[-1]-e)
                trades.append((k,sm,d,dr,res-1.0))
T=pd.DataFrame(trades,columns=['k','sm','d','dr','pts']); T['type']=T.d.map(dd.type); T['yr']=T.d.dt.year
T['split']=np.where((T.yr>=2019)&(T.yr<=2021),'train','test')
def tt(x): return x.mean()/(x.std()/np.sqrt(len(x))) if len(x)>2 else np.nan
for (k,sm),g in T.groupby(['k','sm']):
    print(f'k={k} stop={sm}ATR')
    for ty in ['MR','MID','TREND']:
        for sp in ['train','test']:
            x=g[(g.type==ty)&(g.split==sp)].pts; print(f'   {ty:5s} {sp:5s} n {len(x):4d} net {x.mean():+6.1f} pts t {tt(x):+.1f} win {(x>0).mean():.0%}')
    x=g.pts; print(f'   ALL   n {len(x)} net {x.mean():+.1f} t {tt(x):+.1f}')
# main algo daily by type
Z=pickle.load(open('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/shareable250.pkl','rb')); daily=Z['daily']/20
x=pd.DataFrame({'p':daily}); x['type']=dd.type.reindex(x.index)
print('main algo NQ pts/day by ER tercile (deploy size)'); 
for sp,mask in [('train',(x.index.year>=2019)&(x.index.year<=2021)),('test',~((x.index.year>=2019)&(x.index.year<=2021)))]:
    print(sp, x[mask].groupby('type').p.agg(['mean','count']).round(2).to_dict('index'))
print('ER cutoffs',dd.er.quantile([1/3,2/3]).round(3).tolist())
pickle.dump(T,open('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/mr252.pkl','wb'))
