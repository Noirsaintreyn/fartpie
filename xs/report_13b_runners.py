"""Runner days (day high >= +50% over prior close) in the cache: how early does the pump detector fire, how much is left after, and precision (P(runner | early event)). Declared before results. Survivor-cache caveat: true 100-300% names are mostly missing/delisted, so this UNDERSTATES runner frequency."""
import pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data=pickle.load(open(SC+'uni_cache.pkl','rb')); SPLIT=pd.Timestamp('2025-07-01')
R=[];E=[];nd=0
for tk,Y in data.items():
    Y=Y.dropna()
    if len(Y)<1500 or Y.close.median()>20: continue
    idx=Y.index; d=idx.normalize(); g=Y.groupby(d); pc=g.close.last().shift(1); dh=g.high.max(); dc=g.close.last(); nd+=len(pc)
    v=Y.volume.astype(float); slot=(idx.hour*60+idx.minute); med=v.groupby(slot).transform(lambda s:s.shift(1).rolling(20,min_periods=10).median()); rv=(v/med).values
    r=np.log(Y.close).diff(); sg=r.rolling(100).std().shift(1).values; rt=r.values; c=Y.close.values; o=Y.open.values; h=Y.high.values; l=Y.low.values
    pcb=pc.reindex(d).values; dhb=dh.reindex(d).values; dcb=dc.reindex(d).values
    pos=(c-l)/np.maximum(h-l,1e-9)
    ev=(rv>=3)&(rt>=3*sg)&(pos>=.7)&np.isfinite(pcb)&(c>=1)
    seen=set()
    for i in np.where(ev)[0]:
        if d[i] in seen or i+1>=len(Y) or d[i+1]!=d[i]: continue   # first event of day, not last bar
        seen.add(d[i]); so_far=c[i]/pcb[i]-1; en=o[i+1]
        after_hi=h[i+1:][d[i+1:]==d[i]].max()
        E.append(dict(tk=tk,t=idx[i+1],sofar=so_far,runner=float(dhb[i]/pcb[i]-1>=.5),left_hi=after_hi/en-1,left_cl=dcb[i]/en-1,px=c[i]))
E=pd.DataFrame(E); E['tr']=E.t<SPLIT
def st(g):
    if len(g)<10: return 'n%d'%len(g)
    return 'n%-4d P(runner>=+50%%) %4.1f%% | left to day-high mean %+5.1f%% med %+5.1f%% | to close mean %+5.1f%% med %+5.1f%% win %2.0f%%'%(len(g),g.runner.mean()*100,g.left_hi.mean()*100,g.left_hi.median()*100,g.left_cl.mean()*100,g.left_cl.median()*100,(g.left_cl>.0075).mean()*100)
print('ticker-days',nd,'first-of-day events',len(E),'| runner days among events',int(E.runner.sum()))
for nm,m in (('TRAIN',E.tr),('TEST',~E.tr)):
    D=E[m]; print('\n==',nm); print('all first-of-day events      ',st(D))
    for lo,hi in ((-1,.05),(.05,.15),(.15,.30),(.30,9)): print('day gain so far %3d-%3s%%      '%(lo*100,'inf' if hi==9 else int(hi*100)),st(D[(D.sofar>=lo)&(D.sofar<hi)]))
    print('px<3 & gain<15%              ',st(D[(D.px<3)&(D.sofar<.15)]))
