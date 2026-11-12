"""Nov 12 2026: vol-normalised TIMING z-scores for the 1h vector bull flip. DECLARED BEFORE RESULTS.
Signal is fixed (1h bull flip, entry next open, exit next open after bear flip, 40bp). Question: does WHERE price is relative to the vol forecast tell us if it is the right moment?
Vol: HAR-RV (log RV_d, RV_w(5d), RV_m(22d) -> next-5d mean log RV), pooled OLS fitted on TRAIN (< 2025-07-01) only, applied everywhere; Yang-Zhang 5d/20d from daily OHLC built from the 1h bars.
Features (all in units of forecast daily sigma, so comparable across names/prices; each has a stated mechanism and a pre-declared SIGN for the hypothesis):
  ext   = (entry open - vector level)/(px*sig)      H: large = extended / late -> worse (-)
  run   = (flip close - min low last 20 bars)/(px*sig) H: big run already done -> worse (-)
  shock = flip-bar return / (sig/sqrt(6.5))          H: bigger shock bar = more informed -> better (+)
  emaz  = (close/EMA50 - 1)/sig                      H: stretched above mean -> worse (-)
  vsur  = log(YZ5 / HAR forecast)                    H: realised vol above forecast = vol expansion starting -> better (+)
  vlvl  = log HAR sigma annualised                   H: more vol = more room -> better (+)
  vcmp  = log(YZ5 / YZ60)                            H: compression before the flip -> better, so (-)
  gapz  = overnight gap into flip day / sig          H: gap-up context -> better (+)
Outputs: Spearman IC of each feature vs trade return (train, test), quintile table (train cut-points applied to test), composite of pre-declared signs, and a rule test vs random-skip null. Nothing is tuned on test."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'xs'); sys.path.insert(0,'.')
from scipy.stats import spearmanr
from vector_pine import run_state_machine
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data=pickle.load(open(SC+'uni_cache.pkl','rb')); COST=.004; SPLIT=pd.Timestamp('2025-07-01')
def daily(Y):
    d=Y.index.normalize(); G=Y.groupby(d); o=G.agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'))
    lr=np.log(Y.close).diff(); o['rv']=(lr**2).groupby(d).sum(); return o.iloc[1:]
def yz(D,n):
    o,h,l,c=[np.log(D[k]) for k in ('open','high','low','close')]; co=o-c.shift(); oc=c-o; rs=(h-c)*(h-o)+(l-c)*(l-o)
    k=.34/(1.34+(n+1)/(n-1)); s2=co.rolling(n).var()+k*oc.rolling(n).var()+(1-k)*rs.rolling(n).mean(); return np.sqrt(s2.clip(lower=1e-12))
# ---- daily frames + HAR fit on train
DD={}; X=[]
for tk,Y in data.items():
    Y=Y.dropna()
    if len(Y)<800: continue
    D=daily(Y); D=D[D.rv>0]; l=np.log(D.rv); har=pd.DataFrame({'d':l,'w':np.log(D.rv.rolling(5).mean()),'m':np.log(D.rv.rolling(22).mean())})
    har['y']=np.log(D.rv.rolling(5).mean().shift(-5)); DD[tk]=(Y,D,har)
    X.append(har[har.index<SPLIT-pd.Timedelta(days=7)].dropna())
X=pd.concat(X); A=np.c_[np.ones(len(X)),X[['d','w','m']].values]; beta=np.linalg.lstsq(A,X.y.values,rcond=None)[0]; print('HAR beta',beta.round(3),'n',len(X))
rows=[]
for tk,(Y,D,har) in DD.items():
    f=(beta[0]+har[['d','w','m']].values@beta[1:]); fc=pd.Series(np.sqrt(np.exp(f)),index=har.index)  # forecast daily sigma (5d mean var)
    y5,y60=yz(D,5),yz(D,60)
    reg,lev,_,fl=run_state_machine(Y.assign(datetime=Y.index)); idx=Y.index; o,h,l_,c=Y.open.values,Y.high.values,Y.low.values,Y.close.values; vol=Y.volume.values
    ema50=Y.close.ewm(span=50,adjust=False).mean().values; days=idx.normalize()
    ul=[(i,s) for i,s in fl]; pos=None
    for i,s in ul:
        if s==1 and pos is None: pos=i
        elif s==-1 and pos is not None:
            e,x=pos+1,i+1; pos=None
            if x>=len(Y): continue
            d0=days[pos if False else e-1]; prevd=D.index[D.index<d0]
            if len(prevd)<70: continue
            p=prevd[-1]; sg=fc.get(p,np.nan)
            if not np.isfinite(sg): continue
            px=o[e]; ret=o[x]/o[e]-1-COST
            if px<1 or px>5: continue
            fi=e-1; lo20=l_[max(0,fi-20):fi+1].min(); vm=vol[max(0,fi-20):fi].mean()
            gp=D.open.get(d0,np.nan)/D.close.get(p,np.nan)-1 if d0 in D.index else np.nan
            rows.append(dict(tk=tk,t=idx[e],ret=ret,px=px,ext=(px-lev[fi])/(px*sg),run=(c[fi]-lo20)/(px*sg),shock=(c[fi]/o[fi]-1)/(sg/np.sqrt(6.5)),emaz=(c[fi]/ema50[fi]-1)/sg,
                vsur=np.log(y5.get(p,np.nan)/sg),vlvl=np.log(sg*np.sqrt(252)),vcmp=np.log(y5.get(p,np.nan)/y60.get(p,np.nan)),gapz=gp/sg,vrat=vol[fi]/vm if vm>0 else np.nan))
T=pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan).dropna(); T.to_pickle(SC+'zscore_T.pkl'); T['tr']=T.t<SPLIT
print('events',len(T),'train',T.tr.sum(),'test',(~T.tr).sum(),'| mean ret train %.2f%% test %.2f%%'%(T[T.tr].ret.mean()*100,T[~T.tr].ret.mean()*100))
SG={'ext':-1,'run':-1,'shock':1,'emaz':-1,'vsur':1,'vlvl':1,'vcmp':-1,'gapz':1}
print('\nSpearman IC feature vs trade return  (pre-declared sign in brackets)')
for k,sgn in SG.items():
    a=spearmanr(T[T.tr][k],T[T.tr].ret); b=spearmanr(T[~T.tr][k],T[~T.tr].ret)
    print('%-6s [%+d]  train IC %+.3f (p %.3f) | test IC %+.3f (p %.3f)'%(k,sgn,a[0],a[1],b[0],b[1]))
print('\nQuintiles (train cut points) mean ret %, n')
for k in SG:
    q=np.unique(T[T.tr][k].quantile([.2,.4,.6,.8]).values); T['q']=np.searchsorted(q,T[k])
    r=T.groupby(['q',T.tr]).ret.agg(['mean','size']).unstack()
    print(k,' | '.join('Q%d tr %+.1f(%d) te %+.1f(%d)'%(i,r.loc[i,('mean',True)]*100,r.loc[i,('size',True)],r.loc[i,('mean',False)]*100,r.loc[i,('size',False)]) for i in r.index))
# composite with pre-declared signs, z-scored on train
Z=T[list(SG)].copy(); mu,sd=Z[T.tr].mean(),Z[T.tr].std(); Z=(Z-mu)/sd; T['comp']=sum(Z[k]*s for k,s in SG.items())/len(SG)
thr=T[T.tr].comp.median(); rng=np.random.default_rng(0)
for nm,m in (('train',T.tr),('test',~T.tr)):
    S=T[m]; K=S[S.comp>thr]; null=[S.sample(len(K),random_state=i).ret.mean() for i in range(2000)]
    print('\n%s: comp>train-median: n %d mean %+.2f%% win %.0f%% vs all %+.2f%% | p(random-skip) %.3f | IC %+.3f'%(nm,len(K),K.ret.mean()*100,(K.ret>0).mean()*100,S.ret.mean()*100,(np.array(null)>=K.ret.mean()).mean(),spearmanr(S.comp,S.ret)[0]))
