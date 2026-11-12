"""Nov 12 2026: ADAPTIVE FAIR-VALUE z-score. DECLARED BEFORE RESULTS. Rey's point: a z-score off a stale mean says 'extended' after a move that has actually established NEW value (price accepted there with volume);
the fair value must migrate with accepted volume and the scale must be the value-area width / volume-weighted stdev, not just distance.
Measures at the 1h flip bar close (fi), $1-5 names, 1h bars, strict split train<2025-07-01 / test after, nothing fitted on test:
  zA   static: (c - rolling 20-session VWAP) / volume-weighted stdev of typical price          [classic VWAP-band z]
  zB   value-area: (c - POC) / half-width of the 70% value area, profile of the last 140 bars    [volume-profile z]
  zC   ADAPTIVE: min-|z| over VWAP windows 5/10/20 sessions (fair value = the window price has already migrated to; if a move was accepted, the short window catches up and z collapses)
  acc  acceptance: share of the last 20 bars' volume traded ABOVE the prior (bars fi-140..fi-20) value-area high   [new value being built]
  zH   (c - VWAP20)/(px*HAR daily sigma)  [vol-forecast scale instead of volume scale; the previous study's scale]
Hypotheses (sign = effect on forward return): zA,zB,zC high -> mean reversion (-), and zC should be the MOST monotone because it removes false 'extended' reads after accepted moves;
acc high -> continuation (+). Outcomes: trade return, first-bar return (the 'goes red instantly' complaint), 4-bar min (MAE), win rate, ex-top-1% mean (the tail check).
Also an ALL-BARS test (every 6th bar, $1-5 names, forward 6-bar return by z bucket, train cut-points) to ask the oversold/overbought question without the flip."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'xs'); sys.path.insert(0,'.')
from scipy.stats import spearmanr
from vector_pine import run_state_machine
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data=pickle.load(open(SC+'uni_cache.pkl','rb')); COST=.004; SPLIT=pd.Timestamp('2025-07-01'); WS={'5':35,'10':70,'20':140}
def vw(Y,n):
    tp=(Y.high+Y.low+Y.close)/3; v=Y.volume; pv=(tp*v).rolling(n).sum(); vv=v.rolling(n).sum(); m=pv/vv; var=(tp**2*v).rolling(n).sum()/vv-m**2
    return m,np.sqrt(var.clip(lower=1e-12))
def zfeat(Y):
    c=Y.close; out={}
    for k,n in WS.items():
        m,s=vw(Y,n); out['z'+k]=(c-m)/s.clip(lower=c*0.002)
    Z=pd.DataFrame(out); idx=np.abs(Z.values).argmin(axis=1); zC=pd.Series(Z.values[np.arange(len(Z)),idx],index=Z.index); zC[Z.isna().any(axis=1)]=np.nan
    return Z['z20'].rename('zA'),zC.rename('zC')
def profile(tp,lo,hi,v,nb=30):
    if hi<=lo: return np.nan,np.nan,np.nan
    e=np.linspace(lo,hi,nb+1); b=np.clip(np.searchsorted(e,tp)-1,0,nb-1); h=np.bincount(b,weights=v,minlength=nb); p=h.argmax(); tot=h.sum(); a=b_=p; cum=h[p]
    while cum<.7*tot and (a>0 or b_<nb-1):
        up=h[b_+1] if b_<nb-1 else -1; dn=h[a-1] if a>0 else -1
        if up>=dn: b_+=1; cum+=h[b_]
        else: a-=1; cum+=h[a]
    return (e[p]+e[p+1])/2,e[a],e[b_+1]
# ---- HAR for zH (train-only fit, as in report_12_zscore)
def daily(Y):
    d=Y.index.normalize(); G=Y.groupby(d); o=G.agg(open=('open','first'),close=('close','last')); lr=np.log(Y.close).diff(); o['rv']=(lr**2).groupby(d).sum(); return o.iloc[1:]
HD={}; X=[]
for tk,Y in data.items():
    Y=Y.dropna()
    if len(Y)<800: continue
    D=daily(Y); D=D[D.rv>0]; l=np.log(D.rv); h=pd.DataFrame({'d':l,'w':np.log(D.rv.rolling(5).mean()),'m':np.log(D.rv.rolling(22).mean())}); h['y']=np.log(D.rv.rolling(5).mean().shift(-5)); HD[tk]=(Y,D,h); X.append(h[h.index<SPLIT-pd.Timedelta(days=7)].dropna())
X=pd.concat(X); beta=np.linalg.lstsq(np.c_[np.ones(len(X)),X[['d','w','m']].values],X.y.values,rcond=None)[0]
rows=[]; allb=[]
for tk,(Y,D,h) in HD.items():
    zA,zC=zfeat(Y); sg=pd.Series(np.sqrt(np.exp(beta[0]+h[['d','w','m']].values@beta[1:])),index=h.index)
    m20,s20=vw(Y,140); idx=Y.index; days=idx.normalize(); o,hi,lo,c,vol=[Y[k].values for k in ('open','high','low','close','volume')]; tp=(hi+lo+c)/3
    # all-bars sample (every 6th), $1-5
    fw=pd.Series(c,index=idx).shift(-6)/pd.Series(c,index=idx)-1; ok=(Y.close>=1)&(Y.close<=5)
    S=pd.DataFrame({'zA':zA,'zC':zC,'fw':fw})[ok].iloc[::6].dropna(); S['t']=S.index; allb.append(S)
    reg,lev,_,fl=run_state_machine(Y.assign(datetime=Y.index)); pos=None
    for i,s in fl:
        if s==1 and pos is None: pos=i
        elif s==-1 and pos is not None:
            e,x=pos+1,i+1; fi=pos; pos=None
            if x>=len(Y) or fi<150: continue
            px=o[e]
            if px<1 or px>5: continue
            p=h.index[h.index<days[fi]]
            if len(p)<70: continue
            sgv=sg.get(p[-1],np.nan)
            if not np.isfinite(sgv) or not np.isfinite(zA.iat[fi]) or not np.isfinite(zC.iat[fi]): continue
            poc,val,vah=profile(tp[fi-140:fi+1],lo[fi-140:fi+1].min(),hi[fi-140:fi+1].max(),vol[fi-140:fi+1])
            _,pval,pvah=profile(tp[fi-140:fi-20],lo[fi-140:fi-20].min(),hi[fi-140:fi-20].max(),vol[fi-140:fi-20])
            if not np.isfinite(poc) or vah<=val: continue
            v20=vol[fi-19:fi+1]; acc=v20[tp[fi-19:fi+1]>pvah].sum()/max(v20.sum(),1)
            rows.append(dict(tk=tk,t=idx[e],ret=o[x]/o[e]-1-COST,fb=c[e]/o[e]-1,mae4=min(lo[e:e+4].min()/px-1,0),zA=zA.iat[fi],zB=(c[fi]-poc)/((vah-val)/2),zC=zC.iat[fi],acc=acc,zH=(c[fi]-m20.iat[fi])/(px*sgv)))
T=pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan).dropna(); T['tr']=T.t<SPLIT; T.to_pickle(SC+'zfair_T.pkl')
print('events',len(T),'train',T.tr.sum(),'test',(~T.tr).sum())
F=['zA','zB','zC','zH','acc']; SG={'zA':-1,'zB':-1,'zC':-1,'zH':-1,'acc':1}
print('\nSpearman IC vs: trade ret | first-bar ret | 4-bar MAE   (train / test)   [declared sign vs fwd return]')
for k in F:
    print('%-4s[%+d] '%(k,SG[k])+' | '.join('%s tr %+.3f te %+.3f'%(nm,spearmanr(T[T.tr][k],T[T.tr][nm])[0],spearmanr(T[~T.tr][k],T[~T.tr][nm])[0]) for nm in ('ret','fb','mae4')))
print('\nQuintiles (train cut points): win%, mean, ex-top1% mean, first-bar mean%, MAE4 mean%')
for k in F:
    q=np.unique(T[T.tr][k].quantile([.2,.4,.6,.8]).values); T['q']=np.searchsorted(q,T[k]); print(k)
    for nm,m in (('train',T.tr),('test ',~T.tr)):
        print(' '+nm+' '+' | '.join('Q%d n%d w%2.0f %+.1f x%+.1f fb%+.1f ma%+.1f'%(i,len(g),(g.ret>0).mean()*100,g.ret.mean()*100,g.ret.sort_values().iloc[:int(len(g)*.99)].mean()*100,g.fb.mean()*100,g.mae4.mean()*100) for i,g in T[m].groupby('q')))
B=pd.concat(allb); B['tr']=B.t<SPLIT.tz_localize(None) if B.t.dt.tz is None else B.t<SPLIT.tz_localize(B.t.dt.tz)
print('\nALL-BARS forward 6-bar return by z bucket (train cut points). n train %d test %d'%(B.tr.sum(),(~B.tr).sum()))
for k in ('zA','zC'):
    cuts=B[B.tr][k].quantile([.05,.2,.4,.6,.8,.95]).values; B['q']=np.searchsorted(cuts,B[k]); print(k,'cuts',np.round(cuts,2))
    for nm,m in (('train',B.tr),('test ',~B.tr)):
        print(' '+nm+' '+' | '.join('B%d n%d %+.2f%%'%(i,len(g),g.fw.mean()*100) for i,g in B[m].groupby('q')))
