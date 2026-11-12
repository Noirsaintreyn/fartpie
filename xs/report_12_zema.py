"""Nov 12 2026: EMA-distance z-scores on 1h/4h/daily x EMA 50/100/200, mid/small book vs bigger-cap book. DECLARED BEFORE RESULTS.
z_{tf}_{n} = (close_now - EMA_n of the LAST COMPLETED tf bar) / (close*sigma_HAR); spreads sp_{tf} = (EMA50-EMA200)/(close*sigma). Causal: HTF EMA shifted one HTF bar.
Books: SMALL = bars with price $1-5. BIG = tickers whose median daily dollar volume >= $50M and price >= $10 (proxy for the large-cap screener; the cache is a survivor list, not the live universe).
Hypotheses: far above a slow EMA (z200 high) -> mean-reversion (-) at 24-120 bars; z50 high in an uptrend (vector bull) -> continuation at short horizon; each tf is exploratory, so hit counts are compared with chance (5%).
Horizons (1h bars): 1, 6, 24, 48, 120. Subsets ALL | VB (vector bull) | VB&rv>=1.5. Split 2025-07-01, every 4th bar. A cell is a hit only if the sign agrees and |IC|>=.02 in BOTH halves.
Event panel: 1h bull flips in each book, IC of each z vs trade return (entry next open, exit next open after bear flip, 40bp) and win-rate/ex-top-1% by top-vs-bottom tercile of the best candidates."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'xs'); sys.path.insert(0,'.')
from scipy.stats import spearmanr
from vector_pine import run_state_machine
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; data=pickle.load(open(SC+'uni_cache.pkl','rb')); SPLIT=pd.Timestamp('2025-07-01'); COST=.004
def daily(Y):
    d=Y.index.normalize(); o=Y.groupby(d).agg(close=('close','last')); o['rv']=(np.log(Y.close).diff()**2).groupby(d).sum(); return o.iloc[1:]
HD={}; X=[]; dv={}
for tk,Y in data.items():
    Y=Y.dropna()
    if len(Y)<1500: continue
    D=daily(Y); D=D[D.rv>0]; h=pd.DataFrame({'d':np.log(D.rv),'w':np.log(D.rv.rolling(5).mean()),'m':np.log(D.rv.rolling(22).mean())}); h['y']=np.log(D.rv.rolling(5).mean().shift(-5)); HD[tk]=(Y,h); X.append(h[h.index<SPLIT-pd.Timedelta(days=7)].dropna())
    dv[tk]=(Y.close*Y.volume).groupby(Y.index.normalize()).sum().median()
X=pd.concat(X); beta=np.linalg.lstsq(np.c_[np.ones(len(X)),X[['d','w','m']].values],X.y.values,rcond=None)[0]
big={t for t,v in dv.items() if v>=5e7}; print('tickers',len(HD),'median $vol>=50M:',len(big))
def htf_ema(Y,keys,n):
    G=Y.groupby(keys); cl=G.close.last(); e=cl.ewm(span=n,adjust=False).mean().shift(1); return e
P=[]; EV=[]
for tk,(Y,h) in HD.items():
    sgd=pd.Series(np.sqrt(np.exp(beta[0]+h[['d','w','m']].values@beta[1:])),index=h.index); idx=Y.index; days=idx.normalize()
    ix=np.searchsorted(sgd.index.values,days.values)-1; sg=np.where(ix>=0,sgd.values[np.clip(ix,0,None)],np.nan); c=Y.close; F=pd.DataFrame(index=idx); F['px']=c.values
    k4=pd.MultiIndex.from_arrays([days,(idx.hour>=13).astype(int)]); kd=days
    for nm,keys,ns in (('1h',None,(50,100,200)),('4h',k4,(50,100,200)),('d',kd,(50,100,200))):
        E={}
        for n in ns:
            if keys is None: e=c.ewm(span=n,adjust=False).mean().shift(1)   # 1h: previous bar's EMA
            else:
                s=htf_ema(Y,keys,n); e=pd.Series(s.reindex(keys).values,index=idx)
            E[n]=e; F['z%s_%d'%(nm,n)]=((c-e)/(c*sg)).values
        F['sp_%s'%nm]=((E[50]-E[200])/(c*sg)).values
    F['rv']=(Y.volume/Y.volume.rolling(20).mean().shift()).values
    for hz in (1,6,24,48,120): F['f%d'%hz]=(c.shift(-hz)/c-1).values
    reg,_,_,fl=run_state_machine(Y.assign(datetime=idx)); F['vb']=(reg==1); F['tk']=tk; F['big']=tk in big
    P.append(F.iloc[::4].dropna(subset=['zd_200','z4h_200','z1h_200','f120','rv']))
    o=Y.open.values; pos=None
    for i,s in fl:
        if s==1 and pos is None: pos=i
        elif s==-1 and pos is not None:
            e_,x=pos+1,i+1; fi=pos; pos=None
            if x<len(Y) and fi>200 and np.isfinite(F.zd_200.iat[fi]): EV.append({**F.iloc[fi].drop(['tk']).to_dict(),'t':idx[e_],'tk':tk,'ret':o[x]/o[e_]-1-COST,'px_e':o[e_]})
B=pd.concat(P); B['tr']=B.index<SPLIT; print('bars',len(B))
ZF=[c_ for c_ in B.columns if c_.startswith('z') or c_.startswith('sp_')]; HZ=['f1','f6','f24','f48','f120']
books={'SMALL $1-5':B[(B.px>=1)&(B.px<=5)],'BIG':B[B.big&(B.px>=10)]}
hits=[]; n=0
for bn,Bk in books.items():
    for sn,fn in (('ALL',lambda d:d.px>0),('VB',lambda d:d.vb),('VB&rv>=1.5',lambda d:d.vb&(d.rv>=1.5))):
        D=Bk[fn(Bk)]; print('\n[%s %s] n=%d train %d test %d'%(bn,sn,len(D),D.tr.sum(),(~D.tr).sum())); print('%-8s'%''+''.join('%15s'%h for h in HZ))
        for k in ZF:
            cs=[]
            for hz in HZ:
                a=spearmanr(D[D.tr][k],D[D.tr][hz],nan_policy='omit')[0]; b_=spearmanr(D[~D.tr][k],D[~D.tr][hz],nan_policy='omit')[0]; n+=1; ok=np.sign(a)==np.sign(b_) and min(abs(a),abs(b_))>=.02
                if ok: hits.append((bn,sn,k,hz,round(a,3),round(b_,3)))
                cs.append('%+.3f/%+.3f%s'%(a,b_,'*' if ok else ' '))
            print('%-8s'%k+''.join('%15s'%c_ for c_ in cs))
print('\ncells',n,'hits',len(hits),'chance ~%d'%round(n*.05)); [print('HIT',h) for h in hits]
E=pd.DataFrame(EV).replace([np.inf,-np.inf],np.nan); E['tr']=E.t<SPLIT
for bn,Ek in (('SMALL $1-5',E[(E.px_e>=1)&(E.px_e<=5)]),('BIG',E[E.big&(E.px_e>=10)])):
    Ek=Ek.dropna(subset=ZF); print('\nEVENT PANEL',bn,'n',len(Ek),'train',Ek.tr.sum(),'test',(~Ek.tr).sum(),'mean ret train %.2f%% test %.2f%%'%(Ek[Ek.tr].ret.mean()*100,Ek[~Ek.tr].ret.mean()*100))
    for k in ZF:
        a=spearmanr(Ek[Ek.tr][k],Ek[Ek.tr].ret)[0]; b_=spearmanr(Ek[~Ek.tr][k],Ek[~Ek.tr].ret)[0]
        q=Ek[Ek.tr][k].quantile([1/3,2/3]).values; Ek=Ek.assign(q=np.searchsorted(q,Ek[k]))
        def st(g): r=g.ret.sort_values(); return 'n%d w%2.0f %+.1f x%+.1f'%(len(g),(g.ret>0).mean()*100,g.ret.mean()*100,r.iloc[:int(len(r)*.99)].mean()*100)
        print('%-8s IC tr %+.3f te %+.3f%s | lowT tr [%s] te [%s] | highT tr [%s] te [%s]'%(k,a,b_,'*' if np.sign(a)==np.sign(b_) and min(abs(a),abs(b_))>=.03 else ' ',st(Ek[Ek.tr&(Ek.q==0)]),st(Ek[~Ek.tr&(Ek.q==0)]),st(Ek[Ek.tr&(Ek.q==2)]),st(Ek[~Ek.tr&(Ek.q==2)])))
