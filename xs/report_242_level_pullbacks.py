"""Report 242: PULLBACKS TO FIB / VOLUME-PROFILE LEVELS INSIDE THE 1H VECTOR REGIME, with flow / gamma / IV state, then a walk-forward regression (Rey Nov 17: mean reversion with fib or VP levels + vector context + regime classifier or a regression). DECLARED before results.
Idea: this is mean reversion TO a level WITH the 1H vector (not against it): the vector gives the trend context and the invalidation, the level gives the entry. Long (vector regime +1 at the last closed 1H bar): a 5m bar after 10:00 and before 15:00 whose LOW trades down to a level and CLOSES back above it -> long at the next open; stop = the 1H vector level (trend invalidation, R = entry - vector level >= 15 pts), target 1R or 2R, flat 15:55, cost 1 pt; mirror for shorts. Levels (from the PRIOR RTH day, known at the open): volume-profile POC, VAH, VAL (70% value area, 2-pt bins by typical price) and fib 38.2 / 50 / 61.8 of the prior day's range. Only levels between the vector level and price count (the pullback must stay inside the trend). First touch per level per day, one position at a time.
State at the trade: NEG gamma (prior-close net GEX < 0), IVHI (>= trailing median), ABS (cumulative est. delta of the 6 bars before the touch is AGAINST the trade = sellers pressed into a long level), ER (Kaufman efficiency of the prior 10 daily closes), distance to the vector in ATR14, hours since the open, level type.
PART 1 cells: pooled (first touch of any level per day and side), each level type, pooled x ABS, x NEG, x IVHI; exits 1R and 2R. PART 2: L2 logistic regression on 2019-21 touches (features above, standardised on train), label = trade made money after cost; OUT OF SAMPLE on 2018+2022-26: net pts of the top vs bottom half by predicted probability (threshold = train median). Bar ~3 for PART 1 (about 20 cells); PART 2 needs top > bottom in test with a sensible t. Train 2019-21 | test 2018+2022-26."""
import sys; src=open('/home/claude/fartpie/xs/report_232_vx_exits.py').read(); exec(src[:src.index("MODES=")])
dsrc=open('/home/claude/fartpie/xs/dte_greeks.py').read(); seg=dsrc[dsrc.index("U = '/mnt"):dsrc.index("DAYF =")].replace("f'{U}/xs/ext/gex/QQQ.csv'","'/home/claude/fartpie/xs/ext/gex/QQQ.csv'"); exec(seg)
def tstat(x):
    d=x.groupby('day').pts.sum(); return d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if len(d)>5 else np.nan
dl=v*(2*c-h-l)/np.maximum(h-l,1e-9); cd=np.cumsum(dl)
rth=(m>=570)&(m<=955); tpx=(h+l)/2
# daily levels from each RTH day, applied to the NEXT day
LV={}; rows=[]
for d,ix in pd.Series(np.arange(n)[rth]).groupby(day[rth]):
    ix=ix.values; H=h[ix].max(); L=l[ix].min(); px=np.round(tpx[ix]/2)*2; vol=pd.Series(v[ix]).groupby(px).sum().sort_index(); pocp=vol.idxmax(); tot=vol.sum(); lo=hi_=list(vol.index).index(pocp); acc=vol.iloc[lo]; idx_=vol.index
    while acc<.7*tot and (lo>0 or hi_<len(idx_)-1):
        up=vol.iloc[hi_+1] if hi_<len(idx_)-1 else -1; dn=vol.iloc[lo-1] if lo>0 else -1
        if up>=dn: hi_+=1; acc+=up
        else: lo-=1; acc+=dn
    LV[pd.Timestamp(d)]=dict(POC=pocp,VAH=idx_[hi_],VAL=idx_[lo],F382=L+.382*(H-L),F50=L+.5*(H-L),F618=L+.618*(H-L),rng=H-L,cl=c[ix[-1]],H=H,L=L)
days=sorted(LV); nxt={days[i]:days[i+1] for i in range(len(days)-1)}; LVD={nxt[d]:LV[d] for d in nxt}
rr=pd.Series({d:LV[d]['rng'] for d in days}); atr14=rr.rolling(14).mean().shift(0); ATRD={nxt[d]:atr14[d] for d in nxt}
cl_=pd.Series({d:LV[d]['cl'] for d in days}); er_=((cl_-cl_.shift(10)).abs()/cl_.diff().abs().rolling(10).sum()); ERD={nxt[d]:er_[d] for d in nxt}
TYPES=['POC','VAH','VAL','F382','F50','F618']; C=[]
for d in pd.unique(day):
    td=pd.Timestamp(d)
    if td not in LVD or td<pd.Timestamp('2018-01-01'): continue
    lv_=LVD[td]; ix=np.where((day==d))[0]; seen=set()
    for j in ix:
        if m[j]<600 or m[j]>900 or j+1>=n: continue
        k=hi1[j]; 
        if k<0 or not np.isfinite(lev[k]) or reg[k]==0: continue
        s=int(reg[k]); e=j+1
        for T in TYPES:
            if (T,s) in seen: continue
            lvl=lv_[T]
            if s==1 and l[j]<=lvl<c[j] and lvl>lev[k]: touch=True
            elif s==-1 and h[j]>=lvl>c[j] and lvl<lev[k]: touch=True
            else: touch=False
            if not touch: continue
            R=abs(o[e]-lev[k])
            if R<15: continue
            seen.add((T,s)); pre=cd[j-1]-cd[j-7]; a=G.iloc[G.index.searchsorted(td)-1] if G.index.searchsorted(td)>0 else np.nan; f=prior(td)
            C.append(dict(e=e,s=s,R=R,T=T,day=td,neg=bool(np.isfinite(a) and a<0),iv=bool(f is not None and np.isfinite(f.ivp) and f.ivp>=.5),absb=bool(np.sign(pre)==-s),er=ERD.get(td,np.nan),dist=R/ATRD.get(td,np.nan),hrs=(m[j]-570)/60))
C=pd.DataFrame(C); print('touch candidates',len(C),C.T.value_counts().to_dict() if False else C.groupby('T').size().to_dict())
def run(cs,k):
    out=[]; last=-1
    for _,r in cs.sort_values('e').iterrows():
        e=int(r.e)
        if e<=last or e>=n-2 or day[e]!=np.datetime64(r.day) and False: continue
        gp,mfe,j=trade(e,int(r.s),r.R,'fix',k); out.append(dict(day=r.day,pts=gp-1.0,idx=_)); last=j
    return pd.DataFrame(out)
def cell(nm,D):
    y=D.day.dt.year; res=[]
    for era,mk in (('train',(y>=2019)&(y<=2021)),('test',(y==2018)|(y>=2022))):
        x=D[mk]; res.append(f'{era}: n{len(x):4d} win{np.mean(x.pts>0)*100:3.0f}% net {x.pts.mean():+6.1f} t{tstat(x):+5.1f}' if len(x)>5 else f'{era}: n{len(x)}')
    print(f'{nm:28s}',' || '.join(res),flush=True)
POOL=C.sort_values('e').drop_duplicates(['day','s'])   # first touch of any level per day and side
for k in (1.0,2.0):
    print(f'\n=== PART 1, target {k}R, stop = 1H vector level ===')
    cell('pooled (first touch/day/side)',run(POOL,k))
    for T in TYPES: cell(f'level {T}',run(C[C['T']==T],k))
    cell('pooled & ABS',run(POOL[POOL.absb],k)); cell('pooled & NEG',run(POOL[POOL.neg],k)); cell('pooled & IVHI',run(POOL[POOL.iv],k)); cell('pooled & ABS & NEG',run(POOL[POOL.absb&POOL.neg],k))
# PART 2 regression
print('\n=== PART 2: walk-forward logistic regression (train 2019-21, test 2018+2022-26), target 1R ===')
D1=run(C,1.0); Cm=C.reset_index().merge(D1.rename(columns={'idx':'index'})[['index','pts']],on='index'); Cm['win']=(Cm.pts>0).astype(int); Cm['y']=Cm.day.dt.year
X=pd.concat([Cm[['neg','iv','absb']].astype(float),Cm[['er','dist','hrs']],pd.get_dummies(Cm['T']).astype(float)],axis=1); X['er']=X.er.fillna(X.er.median()); X['dist']=X.dist.clip(upper=X.dist.quantile(.99))
tr_=(Cm.y>=2019)&(Cm.y<=2021); te_=(Cm.y==2018)|(Cm.y>=2022); mu=X[tr_].mean(); sd=X[tr_].std().replace(0,1); Z=((X-mu)/sd).values
def fit(Z,y,lam=30.0,it=60):
    w=np.zeros(Z.shape[1]+1); A=np.c_[np.ones(len(Z)),Z]
    for _ in range(it):
        p=1/(1+np.exp(-A@w)); W=p*(1-p)+1e-6; g=A.T@(y-p)-lam*np.r_[0,w[1:]]; Hh=(A*W[:,None]).T@A+lam*np.diag(np.r_[0,np.ones(Z.shape[1])]); w=w+np.linalg.solve(Hh,g)
    return w
w=fit(Z[tr_.values],Cm.win[tr_].values); P=1/(1+np.exp(-(np.c_[np.ones(len(Z)),Z]@w))); thr=np.median(P[tr_.values])
print('standardised coefficients:',', '.join(f'{c_} {x:+.2f}' for c_,x in zip(X.columns,w[1:])))
from itertools import product
def auc(p,y):
    r=pd.Series(p).rank().values; n1=y.sum(); n0=len(y)-n1; return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
print(f'AUC train {auc(P[tr_.values],Cm.win[tr_].values):.3f} | test {auc(P[te_.values],Cm.win[te_].values):.3f}')
for era,mk in (('train',tr_),('test',te_)):
    for nm,sel_ in (('top half',P>=thr),('bottom half',P<thr),('top quarter',P>=np.quantile(P[tr_.values],.75))):
        x=Cm[mk&sel_]; D=pd.DataFrame(dict(day=x.day,pts=x.pts)); print(f'  {era} {nm:12s} n{len(x):4d} win{x.win.mean()*100:3.0f}% net {x.pts.mean():+6.1f} t{tstat(D):+5.1f}')
print('\n=== by year, pooled first touch, net pts per trade (n) ===')
for k in (1.0,2.0):
    D=run(POOL,k); D['y']=D.day.dt.year; print(f'{k}R',' '.join(f'{y}:{g.pts.mean():+.0f}({len(g)})' for y,g in D.groupby('y')))
D=run(C[C['T']=='F618'],2.0); D['y']=D.day.dt.year; print('F618 2R',' '.join(f'{y}:{g.pts.mean():+.0f}({len(g)})' for y,g in D.groupby('y')))
