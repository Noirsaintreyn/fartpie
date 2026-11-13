"""Nov 12 2026: can we flag squeezes/pumps in mid/small caps (for DISCRETIONARY entry)? Declared before results.
Event (1h bar, ET-naive cache, non-BIG, px $1-20): rvol = vol / median vol of same hour-slot over prior 20 sessions >= 3, bar return >= 3 sigma (sigma = std of 1h log rets over prior 100 bars), close in top 30% of bar range.
Questions: (1) after the event, does it continue or fade (next-open entry, fwd 1/3/6 bars, +1 day close)? (2) does stage matter: run-up already done (ret over prior 20 bars), first-vs-later pump (prior events in 10 sessions), breakout of 20d high, close vs bar high.
Mechanism: squeezes/pumps = forced/FOMO flow; continuation if event is EARLY (little run-up, fresh 20d-high break), fade if LATE (already extended).  Train <2025-07-01 / test after. Costs 40bp ($<5) / 25bp ($5-20)."""
import pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data=pickle.load(open(SC+'uni_cache.pkl','rb')); SPLIT=pd.Timestamp('2025-07-01')
rows=[]
for tk,Y in data.items():
    Y=Y.dropna()
    if len(Y)<1500: continue
    dvm=(Y.close*Y.volume).groupby(Y.index.normalize()).sum().median()
    px0=Y.close.median()
    if dvm>=5e7 and px0>=10: continue
    idx=Y.index; c=Y.close.values; o=Y.open.values; h=Y.high.values; l=Y.low.values; v=Y.volume.astype(float)
    slot=pd.Series(idx.hour*60+idx.minute,index=idx)
    med=v.groupby(slot.values).transform(lambda s: s.shift(1).rolling(20,min_periods=10).median()); rv=(v/med).values
    r=np.log(Y.close).diff(); sg=r.rolling(100).std().shift(1).values; rt=r.values
    rng=np.maximum(h-l,1e-9); pos=(c-l)/rng
    hi20=pd.Series(h,index=idx).rolling(140).max().shift(1).values  # ~20 sessions
    run20=np.r_[np.full(20,np.nan),c[20:]/c[:-20]-1]
    day=idx.normalize(); last_of_day=pd.Series(np.arange(len(Y)),index=idx).groupby(day).transform('max').values
    last=-999; evs=[]
    for i in range(140,len(Y)-8):
        if not(np.isfinite(rv[i]) and np.isfinite(sg[i]) and sg[i]>0): continue
        px=c[i]
        if not(1<=px<=20): continue
        if rv[i]>=3 and rt[i]>=3*sg[i] and pos[i]>=.7 and i-last>=12:
            last=i; e=i+1; en=o[e]; cost=.004 if px<5 else .0025
            f={}
            for k in (1,3,6): f['f%d'%k]=c[min(e+k-1,len(Y)-1)]/en-1-cost
            f['fd']=c[last_of_day[e]]/en-1-cost
            nd=last_of_day[e]+1; f['fnd']=(c[min(last_of_day[min(nd,len(Y)-1)],len(Y)-1)]/en-1-cost) if nd<len(Y) else np.nan
            f['mae6']=l[e:e+6].min()/en-1
            evs.append(dict(tk=tk,t=idx[e],px=px,rv=rv[i],z=rt[i]/sg[i],ret=rt[i],run20=run20[i],brk=float(c[i]>hi20[i]),pos=pos[i],gap_up=float(en/c[i]-1),prior=sum(1 for q in evs if i-q['_i']<=70),_i=i,**f))
    rows+=evs
T=pd.DataFrame(rows); T['tr']=T.t<SPLIT; T.to_pickle(SC+'pumps_T.pkl'); print('events',len(T),'tickers',T.tk.nunique())
def st(g):
    if len(g)<15: return 'n%d'%len(g)
    f=lambda c:g[c].mean()*100
    return 'n%-4d f1 %+5.2f f3 %+5.2f f6 %+5.2f day %+5.2f next %+5.2f | win6 %2.0f%% med6 %+5.2f ex-top1(6) %+5.2f mae6 %+5.1f'%(len(g),f('f1'),f('f3'),f('f6'),f('fd'),g.fnd.mean()*100,(g.f6>0).mean()*100,g.f6.median()*100,g.f6.sort_values().iloc[:int(len(g)*.99)].mean()*100,g.mae6.mean()*100)
for nm,m in (('TRAIN',T.tr),('TEST',~T.tr)):
    D=T[m]; print('\n==',nm); print('all events      ',st(D))
    print('run20<10%       ',st(D[D.run20<.10])); print('run20 10-30%    ',st(D[(D.run20>=.10)&(D.run20<.30)])); print('run20>=30%      ',st(D[D.run20>=.30]))
    print('20d-high break  ',st(D[D.brk==1])); print('no break        ',st(D[D.brk==0]))
    print('1st (no prior)  ',st(D[D.prior==0])); print('2nd+ in 10 days ',st(D[D.prior>=1]))
    print('rv>=6           ',st(D[D.rv>=6])); print('z>=5            ',st(D[D.z>=5]))
    print('px<3            ',st(D[D.px<3])); print('px 3-20         ',st(D[D.px>=3]))
    print('EARLY(run20<10%,break,1st)',st(D[(D.run20<.10)&(D.brk==1)&(D.prior==0)])); print('LATE (run20>=30%)  ',st(D[D.run20>=.30]))
