"""Nov 12 2026: volume-explosion runner scan on 2,024 US $0.30-20 stocks (Yahoo daily, 2y, avg vol>100k NOW -> survivor/selection caveats, biased against names that ran >$20 or delisted). Declared before results.
Event day t: V = vol_t / median vol prior 20d >= 20, G = close_t/close_{t-1}-1, prior close $0.30-20.
A (no look-ahead): after the event close (G>=30% & V>=20; also G>=15%, G>=100%), enter next open: gap, open->close, open->high, close-to-close; cost 1.5% round trip.
B (volume-flag look-ahead, flagged): buy-stop at +15% over prior close on days with V>=20 (full-day vol, optimistic), exit that close.
C: continuation by yesterday-run size and today's gap (POAS case: yesterday +100%, today gaps +x%).
D: how early: among +50%-high days with V>=20, share where yesterday was NOT already a big mover.
Split: train < 2026-02-01 / test after."""
import pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data=pickle.load(open(SC+'lowpx_daily.pkl','rb')); SPLIT=pd.Timestamp('2026-02-01'); COST=.015
R=[]
for tk,X in data.items():
    X=X[~X.index.duplicated()].copy(); X=X[(X.Volume>0)]
    if len(X)<120: continue
    o,h,l,c,v=[X[k].values.astype(float) for k in ('Open','High','Low','Close','Volume')]; idx=X.index
    med=pd.Series(v).shift(1).rolling(20).median().values; V=v/med; pc=np.r_[np.nan,c[:-1]]; G=c/pc-1; HG=h/pc-1
    for i in range(21,len(X)-2):
        if not(np.isfinite(V[i]) and 0.3<=pc[i]<=20 and V[i]>=20): continue
        n=i+1
        R.append(dict(tk=tk,t=idx[i],V=V[i],G=G[i],HG=HG[i],pcl=pc[i],yG=G[i-1],yV=V[i-1] if np.isfinite(V[i-1]) else 0,
          gap=o[n]/c[i]-1,oc=c[n]/o[n]-1-COST,oh=h[n]/o[n]-1,cc=c[n]/c[i]-1,ol=l[n]/o[n]-1,
          bs=(c[i]/(1.15*pc[i])-1-COST) if h[i]>=1.15*pc[i] else np.nan, bs_lo=(l[i]/(1.15*pc[i])-1) if h[i]>=1.15*pc[i] else np.nan))
E=pd.DataFrame(R); E['tr']=E.t<SPLIT; print('V>=20 days',len(E),'tickers',E.tk.nunique(),'| +50% high days',int((E.HG>=.5).sum()))
def st(g,cols=('oc',)):
    if len(g)<10: return 'n%d'%len(g)
    a=g.oc.sort_values(); return 'n%-4d gap %+6.1f%% | o>c net mean %+6.2f%% med %+6.2f%% win %2.0f%% ex-top1%% %+6.2f | o>high mean %+5.1f%% med %+5.1f%% | worst-dd %+5.1f%%'%(len(g),g.gap.mean()*100,g.oc.mean()*100,g.oc.median()*100,(g.oc>0).mean()*100,a.iloc[:int(len(a)*.99)].mean()*100,g.oh.mean()*100,g.oh.median()*100,g.ol.mean()*100)
for nm,m in (('TRAIN',E.tr),('TEST',~E.tr)):
    D=E[m]; print('\n=====',nm,'(days with V>=20: %d)'%len(D))
    print('-- A. enter NEXT OPEN after the event close')
    for lab,f in (('G>=15%',D.G>=.15),('G>=30%',D.G>=.30),('G>=60%',D.G>=.60),('G>=100%',D.G>=1.0),('G 15-30% (modest)',(D.G>=.15)&(D.G<.30)),('G>=30% & first day (yG<10%)',(D.G>=.3)&(D.yG<.10)),('G>=30% & 2nd+ day (yG>=30%)',(D.G>=.3)&(D.yG>=.30)),('G>=30% close near high',(D.G>=.3)&(D.G>=.8*D.HG)),('G>=30% faded off high',(D.G>=.3)&(D.G<.6*D.HG))):
        print('  %-28s'%lab,st(D[f]))
    print('-- B. buy-stop at +15% on V>=20 days (optimistic volume flag), exit close, 1.5% cost')
    b=D.dropna(subset=['bs']); 
    if len(b)>10: print('  n%d mean %+.2f%% med %+.2f%% win %.0f%% | avg worst dip from entry %+.1f%% | ex-top1 %+.2f%%'%(len(b),b.bs.mean()*100,b.bs.median()*100,(b.bs>0).mean()*100,b.bs_lo.mean()*100,b.bs.sort_values().iloc[:int(len(b)*.99)].mean()*100))
    print('-- C. POAS case: yesterday big run (yG>=60%, yV>=20) -> today by gap')
    for lo,hi in ((-1,-.1),(-.1,.1),(.1,.3),(.3,9)):
        g=E[m&(E.yG>=.6)&(E.yV>=20)]; print('  today V>=20 & gap %+4.0f%% to %s%%'%(lo*100,'inf' if hi==9 else '%+.0f'%(hi*100)),end=' '); print('(use day-1 as event) ' if False else '',end=''); print('n/a - see below')
        break
    # day-after-run logic: event = day t with G>=60%, V>=20 (yesterday); next day gap bucket then open>close
    for lo,hi in ((-1,-.1),(-.1,.1),(.1,.3),(.3,9)):
        g=D[(D.G>=.6)&(D.gap>=lo)&(D.gap<hi)]; print('  runner day G>=60%%, next-day gap %+4.0f%%..%s%% :'%(lo*100,'inf' if hi==9 else '%+.0f'%(hi*100)),st(g))
Z=E[E.HG>=.5]; print('\n-- D. earliness: +50%%-high days with V>=20: %d; of them yesterday also G>=30%%: %d%% | yesterday V>=20: %d%%'%(len(Z),int(100*(Z.yG>=.3).mean()),int(100*(Z.yV>=20).mean())))
