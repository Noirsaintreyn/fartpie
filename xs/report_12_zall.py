"""Nov 12 2026: TEST EVERYTHING - standalone mean-relative z-scores x relative volume x vector regime. DECLARED BEFORE RESULTS.
Rey's concern: a trend/breakout model that keeps buying while price is stretched from fair value (at different relative volume) may be paying for extension. Question is separate from the vector: does a fair-value z (several definitions) predict forward return, and does it matter inside vector-bull regimes and at high relative volume?
Features per 1h bar (sigma = HAR daily sigma of the PREVIOUS session, fit on train only): zv20/zv5 (VWAP 20/5 sessions over volume-weighted stdev), zvc (min-|z| over VWAP 5/10/20 sessions = migrating fair value), ze21/ze55/ze200 ((c-EMA)/(c*sigma)), zk (c vs causal Kalman level, same scale), zr1/zr5 (1-day / 5-day return in sigma), zdon (position in the 20-session high/low range, -1..1), rv (bar volume / prior-20-bar mean).
Horizons 1/6/24 bars close-to-close. Samples: every 4th bar, bands $1-5 and $5-100. Subsets: ALL | VB (1h vector bull) | VB & rv>=1.5. Train < 2025-07-01 / test after; a result counts only if sign agrees and |IC|>=0.02 in BOTH halves. Number of cells is printed so chance hits can be judged (expect ~5% by luck).
Event panel: 1h bull-flip entries ($1-5), mean/win/ex-top1% by z-quintile x relvol (flip bar rv<1.5 / >=1.5)."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'xs'); sys.path.insert(0,'.')
from scipy.stats import spearmanr
from vector_pine import run_state_machine
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data=pickle.load(open(SC+'uni_cache.pkl','rb')); COST=.004; SPLIT=pd.Timestamp('2025-07-01')
def vw(Y,n):
    tp=(Y.high+Y.low+Y.close)/3; v=Y.volume; vv=v.rolling(n).sum(); m=(tp*v).rolling(n).sum()/vv; var=(tp**2*v).rolling(n).sum()/vv-m**2; return m,np.sqrt(var.clip(lower=1e-12))
def kal(c):
    r=np.diff(c,prepend=c[0]); R=np.nanvar(r[1:]) if len(r)>2 else 1e-4; q=.05*R; m=np.zeros(len(c)); P=R; m[0]=c[0]
    for i in range(1,len(c)): P+=q; K=P/(P+R); m[i]=m[i-1]+K*(c[i]-m[i-1]); P*=(1-K)
    return m
def daily(Y):
    d=Y.index.normalize(); G=Y.groupby(d); o=G.agg(close=('close','last')); o['rv']=(np.log(Y.close).diff()**2).groupby(d).sum(); return o.iloc[1:]
HD={}; X=[]
for tk,Y in data.items():
    Y=Y.dropna()
    if len(Y)<800: continue
    D=daily(Y); D=D[D.rv>0]; h=pd.DataFrame({'d':np.log(D.rv),'w':np.log(D.rv.rolling(5).mean()),'m':np.log(D.rv.rolling(22).mean())}); h['y']=np.log(D.rv.rolling(5).mean().shift(-5)); HD[tk]=(Y,h); X.append(h[h.index<SPLIT-pd.Timedelta(days=7)].dropna())
X=pd.concat(X); beta=np.linalg.lstsq(np.c_[np.ones(len(X)),X[['d','w','m']].values],X.y.values,rcond=None)[0]
P=[]; EV=[]
for tk,(Y,h) in HD.items():
    sgd=pd.Series(np.sqrt(np.exp(beta[0]+h[['d','w','m']].values@beta[1:])),index=h.index); days=Y.index.normalize()
    ix=np.searchsorted(sgd.index.values,days.values)-1; ok=ix>=0; sg=np.where(ok,sgd.values[np.clip(ix,0,None)],np.nan)
    c=Y.close; F=pd.DataFrame(index=Y.index); F['px']=c.values
    for k,n in (('zv20',140),('zv5',35)):
        m,s=vw(Y,n); F[k]=((c-m)/s.clip(lower=c*.002)).values
    Zs=[((c-vw(Y,n)[0])/vw(Y,n)[1].clip(lower=c*.002)).values for n in (35,70,140)]; Zs=np.vstack(Zs); F['zvc']=Zs[np.abs(np.nan_to_num(Zs,nan=1e9)).argmin(0),np.arange(Zs.shape[1])]
    for k,sp in (('ze21',21),('ze55',55),('ze200',200)): F[k]=((c-c.ewm(span=sp,adjust=False).mean())/(c*sg)).values
    F['zk']=((c.values-np.exp(kal(np.log(c.values))))/(c.values*sg)); lc=np.log(c)
    F['zr1']=(lc-lc.shift(7)).values/sg; F['zr5']=(lc-lc.shift(35)).values/sg/np.sqrt(5)
    hh,ll=Y.high.rolling(140).max(),Y.low.rolling(140).min(); F['zdon']=((c-ll)/(hh-ll)*2-1).values
    F['rv']=(Y.volume/Y.volume.rolling(20).mean().shift()).values
    for hz in (1,6,24): F['f%d'%hz]=(c.shift(-hz)/c-1).values
    reg,_,_,fl=run_state_machine(Y.assign(datetime=Y.index)); F['vb']=(reg==1)
    F['tk']=tk; P.append(F.iloc[::4].dropna(subset=['zv20','zvc','ze200','zr5','rv','f24']))
    pos=None; o=Y.open.values
    for i,s in fl:
        if s==1 and pos is None: pos=i
        elif s==-1 and pos is not None:
            e,x=pos+1,i+1; fi=pos; pos=None
            if x<len(Y) and 1<=o[e]<=5 and fi>140 and np.isfinite(F.zv20.iat[fi]) and np.isfinite(F.rv.iat[fi]): EV.append({**F.iloc[fi][['zv20','zvc','ze55','zk','zr5','zdon','rv']].to_dict(),'t':Y.index[e],'ret':o[x]/o[e]-1-COST})
B=pd.concat(P); B['tr']=B.index<SPLIT; B['band']=np.where(B.px<=5,'1-5','5+'); B=B[B.px>=1]; B=B[B.px<=100]; print('bars',len(B),'train',B.tr.sum(),'test',(~B.tr).sum())
FE=['zv20','zv5','zvc','ze21','ze55','ze200','zk','zr1','zr5','zdon']; HZ=['f1','f6','f24']; hits=[]; ncell=0
print('\nSpearman IC (train / test) *=sign agrees & |IC|>=.02 both halves')
for band in ('1-5','5+'):
    for sn,sm in (('ALL',lambda d:d.tr|~d.tr),('VB',lambda d:d.vb),('VB&rv>=1.5',lambda d:d.vb&(d.rv>=1.5))):
        D=B[B.band==band]; D=D[sm(D)]; print('\n[%s %s] n=%d train %d test %d'%(band,sn,len(D),D.tr.sum(),(~D.tr).sum()))
        print('%-6s'%''+''.join('%16s'%h for h in HZ))
        for k in FE:
            cells=[]
            for hz in HZ:
                a=spearmanr(D[D.tr][k],D[D.tr][hz])[0]; b=spearmanr(D[~D.tr][k],D[~D.tr][hz])[0]; ncell+=1; ok=(np.sign(a)==np.sign(b)) and min(abs(a),abs(b))>=.02
                if ok: hits.append((band,sn,k,hz,a,b))
                cells.append('%+.3f/%+.3f%s'%(a,b,'*' if ok else ' '))
            print('%-6s'%k+''.join('%16s'%c_ for c_ in cells))
print('\ncells',ncell,'hits',len(hits)); [print('HIT',h) for h in hits]
# bucket view for VB bars, $1-5: fwd6 by z quintile x rv hi/lo
print('\nVB bars $1-5 fwd6 by train-quintile of z (mean %), rv<1.5 | rv>=1.5 ; train then test')
D=B[(B.band=='1-5')&B.vb]
for k in ('zv20','zvc','ze55','zk','zr5','zdon'):
    q=np.unique(D[D.tr][k].quantile([.2,.4,.6,.8]).values); D=D.assign(q=np.searchsorted(q,D[k]))
    for nm,m in (('train',D.tr),('test ',~D.tr)):
        print('%-5s %s '%(k,nm)+' | '.join('Q%d %+.2f %+.2f'%(i,g[g.rv<1.5].f6.mean()*100,g[g.rv>=1.5].f6.mean()*100) for i,g in D[m].groupby('q')))
E=pd.DataFrame(EV).replace([np.inf,-np.inf],np.nan).dropna(); E['tr']=E.t<SPLIT; print('\nEVENTS 1h bull flips $1-5:',len(E))
for k in ('zv20','zvc','ze55','zk','zr5','zdon'):
    q=np.unique(E[E.tr][k].quantile([.2,.4,.6,.8]).values); E['q']=np.searchsorted(q,E[k])
    for nm,m in (('train',E.tr),('test ',~E.tr)):
        o_=[]
        for i,g in E[m].groupby('q'):
            for lab,gg in (('lo',g[g.rv<1.5]),('hi',g[g.rv>=1.5])):
                r=gg.ret.sort_values(); o_.append('Q%d%s n%d w%2.0f %+.1f x%+.1f'%(i,lab,len(gg),(gg.ret>0).mean()*100 if len(gg) else 0,gg.ret.mean()*100 if len(gg) else 0,(r.iloc[:int(len(r)*.99)].mean()*100 if len(r)>20 else np.nan)))
        print('%-5s %s '%(k,nm)+' | '.join(o_))
