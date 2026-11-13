"""Rey: most alerts are gap-ups that don't hold after the volume spike; better to enter the DAY BEFORE if over the vector. Test: what at day t-1 CLOSE predicts a runner day t? Declared before results.
Universe: 2,024 $0.30-20 stocks (survivor caveat), daily bars 2y. Features at close t-1 (no look-ahead): V1 = vol_{t-1}/median prior 20d, G1 = day t-1 gain, daily vector regime (bull/bear; constants tuned on 1h, used as-is), dist above vector level in ATR, close vs prior 20d high (break), 20d range tightness.
Outcomes (enter day t-1 close, cost 1.5% round trip): P(day t high >= +50% over t-1 close) [runner], day t close-to-close net, and 3-day hold. Compare to base rate and to 'after the spike' entry.
Split: train < 2026-02-01 / test after. ~1M stock-days; tails dominate means so also report median / ex-top1 / hit rate."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'xs'); sys.path.insert(0,'.')
from vector_pine import run_state_machine
from vector_core import compute_atr_series
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data=pickle.load(open(SC+'lowpx_daily.pkl','rb')); SPLIT=pd.Timestamp('2026-02-01'); COST=.015; R=[]
for tk,X in data.items():
    X=X[~X.index.duplicated()]; X=X[X.Volume>0].copy()
    if len(X)<150: continue
    X.columns=[c.lower() for c in X.columns]; o,h,l,c,v=[X[k].values.astype(float) for k in ('open','high','low','close','volume')]; idx=X.index
    try: reg,lev,_,_=run_state_machine(X[['open','high','low','close','volume']].assign(datetime=idx)); atr=compute_atr_series(h,l,c,14)
    except Exception: continue
    lev=np.asarray(lev,float); med=pd.Series(v).shift(1).rolling(20).median().values; hi20=pd.Series(h).shift(1).rolling(20).max().values; rng20=(pd.Series(h).rolling(10).max()-pd.Series(l).rolling(10).min()).values/c
    for i in range(30,len(X)-3):
        if not(0.3<=c[i]<=20 and np.isfinite(med[i]) and med[i]>0 and np.isfinite(lev[i]) and np.isfinite(atr[i]) and atr[i]>0): continue
        n=i+1
        R.append(dict(tk=tk,t=idx[n],px=c[i],V1=v[i]/med[i],G1=c[i]/c[i-1]-1,bull=float(reg[i]==1),dv=(c[i]-lev[i])/atr[i],brk=float(c[i]>hi20[i]),tight=rng20[i],
          hi=h[n]/c[i]-1,cc=c[n]/c[i]-1-COST,c3=c[min(n+2,len(X)-1)]/c[i]-1-COST,oc=c[n]/o[n]-1,gap=o[n]/c[i]-1))
E=pd.DataFrame(R); E['tr']=E.t<SPLIT; E['run']=(E.hi>=.5).astype(float); E.to_pickle(SC+'prespike_E.pkl'); print('stock-days',len(E),'runner rate train %.2f%% test %.2f%%'%(E[E.tr].run.mean()*100,E[~E.tr].run.mean()*100))
def st(g):
    if len(g)<30: return 'n%d'%len(g)
    a=g.cc.sort_values(); return 'n%-6d P(next-day high>=+50%%) %5.2f%% | next-day c2c net mean %+6.2f%% med %+5.2f%% win %2.0f%% ex-top1%% %+5.2f | 3d mean %+6.2f%%'%(len(g),g.run.mean()*100,g.cc.mean()*100,g.cc.median()*100,(g.cc>0).mean()*100,a.iloc[:int(len(a)*.99)].mean()*100,g.c3.mean()*100)
G=[('ALL',lambda D:D.px>0),('V1 3-10x',lambda D:(D.V1>=3)&(D.V1<10)),('V1 10-30x',lambda D:(D.V1>=10)&(D.V1<30)),('V1>=30x',lambda D:D.V1>=30),('V1>=3x & G1<10% (quiet build)',lambda D:(D.V1>=3)&(D.G1<.10)&(D.G1>-.05)),
 ('V1>=3x & G1<10% & vector BULL',lambda D:(D.V1>=3)&(D.G1<.10)&(D.G1>-.05)&(D.bull==1)),('...& 20d-high break',lambda D:(D.V1>=3)&(D.G1<.10)&(D.G1>-.05)&(D.bull==1)&(D.brk==1)),('...& close within 1 ATR above vector',lambda D:(D.V1>=3)&(D.G1<.10)&(D.G1>-.05)&(D.bull==1)&(D.dv<=1.5)),
 ('V1>=3x & G1<10% & vector BEAR',lambda D:(D.V1>=3)&(D.G1<.10)&(D.G1>-.05)&(D.bull==0)),('G1 10-30% & V1>=5x (day-1 of move)',lambda D:(D.G1>=.10)&(D.G1<.30)&(D.V1>=5)),('G1 10-30% & V1>=5x & bull',lambda D:(D.G1>=.10)&(D.G1<.30)&(D.V1>=5)&(D.bull==1)),('bull & break & tight(<25%) any vol',lambda D:(D.bull==1)&(D.brk==1)&(D.tight<.25)),('V1>=3x & G1<10% & px<$3',lambda D:(D.V1>=3)&(D.G1<.10)&(D.G1>-.05)&(D.px<3))]
for nm,m in (('TRAIN',E.tr),('TEST',~E.tr)):
    D=E[m]; print('\n=====',nm)
    for lab,f in G: print('%-40s'%lab,st(D[f(D)]))
