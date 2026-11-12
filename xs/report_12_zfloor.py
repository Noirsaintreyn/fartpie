"""Nov 12 2026: VECTOR-FLOOR z + CORRELATION MODEL. DECLARED BEFORE RESULTS.
Idea (Rey): reverse-engineer the vector: its ratchet level is the 'ground/floor' in a bull regime, so z_floor = (close - level)/scale is a fair-value-free stretch measure that is native to the signal.
Correlation layer: (a) trend corr = rolling corr(close, time) over 35/140 bars (how directional the recent path was); (b) market-residual stretch = 35-bar cumulative idiosyncratic return after rolling beta to the equal-weight universe return, / (resid sd*sqrt(35)); (c) AC = rolling lag-1 autocorrelation of 1h returns over 140 bars (negative = reverting tape, positive = trending tape) used to decide whether a stretch should be read as reversion or continuation.
Features: zfa=(c-level)/ATR14, zfs=(c-level)/(c*sigma_HAR), tc35, tc140, res35, ac140, rv. Only bars where 1h vector is bull (floor exists).
Hypotheses: zfa/zfs high -> worse near-term (-) only where ac140<0 (reverting tape) and neutral/(+) where ac140>0; res35 high -> reversion (-); tc high -> continuation (+). Anything else is exploratory.
Then a RIDGE model on train (all features + zfloor x ac interactions) -> forward 6-bar return; judged on test IC, test decile spread, and whether the top-decile bars beat the all-bars mean in test.
Bands $1-5 and $5-100, every 4th bar, horizons 1/6/24, split 2025-07-01. A cell counts only if sign agrees and |IC|>=.02 in BOTH halves."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'xs'); sys.path.insert(0,'.')
from scipy.stats import spearmanr
from vector_pine import run_state_machine
from vector_core import compute_atr_series
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data=pickle.load(open(SC+'uni_cache.pkl','rb')); SPLIT=pd.Timestamp('2025-07-01')
R=pd.concat({tk:np.log(Y.close).diff() for tk,Y in data.items() if len(Y)>800},axis=1); MKT=R.mean(axis=1)
def daily(Y):
    d=Y.index.normalize(); o=Y.groupby(d).agg(close=('close','last')); o['rv']=(np.log(Y.close).diff()**2).groupby(d).sum(); return o.iloc[1:]
HD={}; X=[]
for tk,Y in data.items():
    Y=Y.dropna()
    if len(Y)<800: continue
    D=daily(Y); D=D[D.rv>0]; h=pd.DataFrame({'d':np.log(D.rv),'w':np.log(D.rv.rolling(5).mean()),'m':np.log(D.rv.rolling(22).mean())}); h['y']=np.log(D.rv.rolling(5).mean().shift(-5)); HD[tk]=(Y,h); X.append(h[h.index<SPLIT-pd.Timedelta(days=7)].dropna())
X=pd.concat(X); beta=np.linalg.lstsq(np.c_[np.ones(len(X)),X[['d','w','m']].values],X.y.values,rcond=None)[0]
P=[]; tt=None
for tk,(Y,h) in HD.items():
    sgd=pd.Series(np.sqrt(np.exp(beta[0]+h[['d','w','m']].values@beta[1:])),index=h.index); days=Y.index.normalize()
    ix=np.searchsorted(sgd.index.values,days.values)-1; sg=np.where(ix>=0,sgd.values[np.clip(ix,0,None)],np.nan)
    c=Y.close; r=np.log(c).diff(); m=MKT.reindex(Y.index).fillna(0)
    reg,lev,_,fl=run_state_machine(Y.assign(datetime=Y.index)); atr=compute_atr_series(Y.high.values,Y.low.values,Y.close.values,14)
    F=pd.DataFrame(index=Y.index); F['px']=c.values; F['vb']=(reg==1); lv=pd.Series(lev,index=Y.index)
    F['zfa']=((c-lv)/pd.Series(atr,index=Y.index)).where(F.vb); F['zfs']=((c-lv)/(c*sg)).where(F.vb)
    t=pd.Series(np.arange(len(Y),dtype=float),index=Y.index)
    F['tc35']=c.rolling(35).corr(t); F['tc140']=c.rolling(140).corr(t)
    cv=r.rolling(140).cov(m); b=cv/m.rolling(140).var().replace(0,np.nan); res=r-b.shift()*m; F['res35']=res.rolling(35).sum()/(res.rolling(140).std()*np.sqrt(35))
    F['ac140']=r.rolling(140).corr(r.shift()); F['rv']=(Y.volume/Y.volume.rolling(20).mean().shift())
    for hz in (1,6,24): F['f%d'%hz]=(c.shift(-hz)/c-1).values
    P.append(F.iloc[::4].dropna(subset=['zfa','zfs','tc140','res35','ac140','rv','f24']))
B=pd.concat(P); B=B[(B.px>=1)&(B.px<=100)]; B['tr']=B.index<SPLIT; B['band']=np.where(B.px<=5,'1-5','5+'); print('VB bars',len(B),'train',B.tr.sum(),'test',(~B.tr).sum())
FE=['zfa','zfs','tc35','tc140','res35','ac140']; HZ=['f1','f6','f24']; hits=[]; n=0
print('\nSpearman IC train/test (VB bars) *=hit')
for band in ('1-5','5+'):
    for sn,fn in (('VB',lambda d:d.px>0),('VB&rv>=1.5',lambda d:d.rv>=1.5),('VB&ac<0',lambda d:d.ac140<0),('VB&ac>0',lambda d:d.ac140>0)):
        D=B[B.band==band]; D=D[fn(D)]; print('\n[%s %s] n=%d'%(band,sn,len(D)))
        for k in FE:
            cs=[]
            for hz in HZ:
                a=spearmanr(D[D.tr][k],D[D.tr][hz])[0]; b_=spearmanr(D[~D.tr][k],D[~D.tr][hz])[0]; n+=1; ok=np.sign(a)==np.sign(b_) and min(abs(a),abs(b_))>=.02
                if ok: hits.append((band,sn,k,hz,round(a,3),round(b_,3)))
                cs.append('%+.3f/%+.3f%s'%(a,b_,'*' if ok else ' '))
            print('%-6s'%k+''.join('%16s'%c_ for c_ in cs))
print('\ncells',n,'hits',len(hits),'(chance ~%d)'%round(n*.05)); [print('HIT',h) for h in hits]
print('\nfwd6 mean % by train-quintile of zfs x tape (ac<0 reverting | ac>0 trending), $1-5 and 5+')
for band in ('1-5','5+'):
    D=B[B.band==band]; q=np.unique(D[D.tr].zfs.quantile([.2,.4,.6,.8]).values); D=D.assign(q=np.searchsorted(q,D.zfs))
    for nm,m in (('train',D.tr),('test ',~D.tr)):
        print('%-3s %s '%(band,nm)+' | '.join('Q%d %+.2f %+.2f'%(i,g[g.ac140<0].f6.mean()*100,g[g.ac140>=0].f6.mean()*100) for i,g in D[m].groupby('q')))
# ridge correlation model
def design(D): 
    Z=D[FE+['rv']].copy(); Z['rv']=np.log1p(Z.rv.clip(upper=20)); 
    for k in ('zfa','zfs','res35'): Z[k+'_x_ac']=Z[k]*Z.ac140
    return Z
for band in ('1-5','5+'):
    D=B[B.band==band]; Dt,Ds=D[D.tr],D[~D.tr]; Zt,Zs=design(Dt),design(Ds); mu,sd=Zt.mean(),Zt.std(); Zt=((Zt-mu)/sd).clip(-5,5); Zs=((Zs-mu)/sd).clip(-5,5)
    for hz in HZ:
        y=Dt[hz].clip(Dt[hz].quantile(.01),Dt[hz].quantile(.99)); A=np.c_[np.ones(len(Zt)),Zt.values]; w=np.linalg.solve(A.T@A+len(A)*0.05*np.eye(A.shape[1]),A.T@(y-y.mean())) 
        pt,ps=Zt.values@w[1:],Zs.values@w[1:]; ic_t,ic_s=spearmanr(pt,Dt[hz])[0],spearmanr(ps,Ds[hz])[0]; cut=np.quantile(pt,.9); top=Ds[ps>=cut]; bot=Ds[ps<=np.quantile(pt,.1)]
        print('RIDGE %s %s: IC train %+.3f test %+.3f | test top-decile mean %+.2f%% (n %d) bottom %+.2f%% all %+.2f%%'%(band,hz,ic_t,ic_s,top[hz].mean()*100,len(top),bot[hz].mean()*100,Ds[hz].mean()*100))
