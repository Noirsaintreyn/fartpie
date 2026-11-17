"""Report 246: THE STORY MODEL - all sources as CONTEXT with order flow, learned weights, causal (Rey Nov 17: stop judging levels one by one; the move tells the story - pivots as liquidity, poor highs/lows, a call wall on a 1-day HVN, a fib on the 5-day POC; options flow, volatility, fib and VP are context; ignore levels that do not matter). DECLARED before results.
Candidates = every stage-2 rejection (report 244/245 definition) at ANY level group: VPd/VP3/VPw/VPm, EXT, SWG, FIB, OPT, plus NEW groups LIQ (1H swing pivots not yet traded through = resting liquidity) and POOR (the prior day's high/low when the profile has no excess tail there: top / bottom two bins hold >= 0.8 x the mean bin volume = unfinished auction). OPTICS clusters dropped (report 245: no better than VP).
Features, all known at the trade: which groups are present at the bar (9 flags + count + number of levels touched), order flow (delta of the 3 bars into the level in the push direction, its strength, volume effort vs the prior 12 bars), gamma sign (prior close), IV >= median, chop (Kaufman ER), hour, side, stop size / ATR14, and day-structure context = distance of price to the day's high / low so far (THOD / TLOD, in ATR14) and whether the bar sits at the extreme.
Models trained ONLY on 2019-21: L2 logistic regression and a shallow gradient-boosted tree (depth 2, 150 trees); label = the fade makes money after cost at 2R (stop = bar extreme + 1, R >= 10, target 2R). Threshold = train quantile of the predicted probability. Judged OUT OF SAMPLE on 2018 + 2022-26: AUC; net pts per trade, trades/yr, t for the top 10 / 20 / 30%; the stage-3 rule as the baseline; by year; which features carry weight (= which level groups matter). One position at a time for the trade counts. Bar ~3 for the t-stats; the model needs a test AUC clearly above 0.5."""
import sys, warnings; warnings.filterwarnings('ignore'); src=open('/home/claude/fartpie/xs/report_232_vx_exits.py').read(); exec(src[:src.index("MODES=")])
dsrc=open('/home/claude/fartpie/xs/dte_greeks.py').read(); seg=dsrc[dsrc.index("U = '/mnt"):dsrc.index("DAYF =")].replace("f'{U}/xs/ext/gex/QQQ.csv'","'/home/claude/fartpie/xs/ext/gex/QQQ.csv'"); exec(seg)
from sklearn.cluster import OPTICS
def tstat(x):
    d=x.groupby('day').pts.sum(); return d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if len(d)>5 else np.nan
dl=v*(2*c-h-l)/np.maximum(h-l,1e-9); cd=np.cumsum(dl); rv=pd.Series(v).rolling(12).mean().shift(1).values
rth=(m>=570)&(m<=955); tpx=(h+l)/2
PROF={}; DH={}; DL={}; DC={}
for d,ix in pd.Series(np.arange(n)[rth]).groupby(day[rth]):
    ix=ix.values; td=pd.Timestamp(d); PROF[td]=pd.Series(v[ix]).groupby(np.round(tpx[ix]/2)*2).sum(); DH[td]=h[ix].max(); DL[td]=l[ix].min(); DC[td]=c[ix[-1]]
days=sorted(PROF); pos={d:i for i,d in enumerate(days)}
def vp_levels(vol):
    idx_=vol.index.values; pocp=vol.idxmax(); tot=vol.sum(); lo=hi_=int(np.where(idx_==pocp)[0][0]); acc=vol.iloc[lo]
    while acc<.7*tot and (lo>0 or hi_<len(idx_)-1):
        up=vol.iloc[hi_+1] if hi_<len(idx_)-1 else -1; dn=vol.iloc[lo-1] if lo>0 else -1
        if up>=dn: hi_+=1; acc+=up
        else: lo-=1; acc+=dn
    full=vol.reindex(np.arange(idx_.min(),idx_.max()+2,2)).fillna(0); sm=full.rolling(5,center=True,min_periods=1).mean(); a=sm.values; ps=sm.index.values; med=np.median(a[a>0])
    hv=sorted([(a[i],ps[i]) for i in range(2,len(a)-2) if a[i]>=a[i-1] and a[i]>=a[i+1] and a[i]>=a[i-2] and a[i]>=a[i+2] and a[i]>=1.25*med and abs(ps[i]-pocp)>10],reverse=True)[:3]
    lv=sorted([(a[i],ps[i]) for i in range(2,len(a)-2) if a[i]<=a[i-1] and a[i]<=a[i+1] and a[i]<=a[i-2] and a[i]<=a[i+2] and 0<a[i]<=.6*med])[:3]
    return [('POC',pocp),('VAH',idx_[hi_]),('VAL',idx_[lo])]+[('HVN',p) for _,p in hv]+[('LVN',p) for _,p in lv]
def optics_levels(vol):
    p=vol.index.values.astype(float); w=np.maximum((vol.values/vol.values.sum()*500).round().astype(int),0); X=np.repeat(p,w).reshape(-1,1)
    if len(X)<60: return []
    lab=OPTICS(min_samples=20,xi=.05,min_cluster_size=.06).fit(X).labels_; out=[]
    for k in set(lab)-{-1}:
        q=X[lab==k,0]; out+=[('c',float(q.mean())),('lo',float(np.percentile(q,3))),('hi',float(np.percentile(q,97)))]
    return out
# 1H swing points (3-bar fractals), confirmed 3 bars later
g=b.set_index('dt').resample('60min',label='right',closed='left').agg({'high':'max','low':'min'}).dropna(); GH,GL,GE=g.high.values,g.low.values,g.index.values
sw=[]
for i in range(3,len(g)-3):
    if GH[i]>GH[i-3:i].max() and GH[i]>=GH[i+1:i+4].max(): sw.append((GE[i+3],GH[i],1))
    if GL[i]<GL[i-3:i].min() and GL[i]<=GL[i+1:i+4].min(): sw.append((GE[i+3],GL[i],-1))
sw=sorted(sw); swt=np.array([x[0] for x in sw]); swp=np.array([x[1] for x in sw]); swk=np.array([x[2] for x in sw])
dayix={}
for i in range(n): dayix.setdefault(pd.Timestamp(day[i]),[]).append(i)
rr=pd.Series({d:DH[d]-DL[d] for d in days}); atr14=rr.rolling(14).mean(); cl_=pd.Series(DC); er_=((cl_-cl_.shift(10)).abs()/cl_.diff().abs().rolling(10).sum())
tr_er=er_[(er_.index>='2019-01-01')&(er_.index<'2022-01-01')].dropna(); q1=tr_er.quantile(1/3)
def comp(i,W): return pd.concat([PROF[days[k]] for k in range(max(0,i-W+1),i+1)]).groupby(level=0).sum()
LEV={}; t0=pd.Timestamp('2018-01-01')
for td in days:
    i=pos[td]
    if i<21: continue
    nd=days[i+1] if i+1<len(days) else None
    if nd is None or nd<t0: continue
    pc=DC[td]; at=atr14[td]; L_=[]
    for gname,W in (('VPd',1),('VP3',3),('VPw',5),('VPm',20)):
        for nm,p in vp_levels(comp(i,W)): L_.append((gname,p))
    pdh,pdl=DH[td],DL[td]; L_+=[('EXT',pdh),('EXT',pdl),('EXT',max(DH[days[k]] for k in range(i-4,i+1))),('EXT',min(DL[days[k]] for k in range(i-4,i+1))),('EXT',max(DH[days[k]] for k in range(i-19,i+1))),('EXT',min(DL[days[k]] for k in range(i-19,i+1)))]
    on=[j for j in dayix.get(td,[]) if m[j]>=1080]+[j for j in dayix.get(nd,[]) if m[j]<570]
    if len(on)>20: L_+=[('EXT',h[on].max()),('EXT',l[on].min())]
    for rng_,lab in ((pdh-pdl,'d'),(max(DH[days[k]] for k in range(i-4,i+1))-min(DL[days[k]] for k in range(i-4,i+1)),'w')):
        lo_=pdl if lab=='d' else min(DL[days[k]] for k in range(i-4,i+1))
        for fr in (.382,.5,.618): L_.append(('FIB',lo_+fr*rng_))
    cut=np.datetime64(pd.Timestamp(nd)+pd.Timedelta(hours=9,minutes=30)); a_=np.searchsorted(swt,cut); b_=np.searchsorted(swt,cut-np.timedelta64(7,'D'))
    L_+=[('SWG',float(x)) for x in swp[b_:a_]]
    t930=np.searchsorted(t5,cut)
    for q in range(b_,a_):
        i0=np.searchsorted(t5,swt[q])
        if i0<t930:
            mit=(h[i0:t930].max()>=swp[q]) if swk[q]==1 else (l[i0:t930].min()<=swp[q])
            if not mit: L_.append(('LIQ',float(swp[q])))
    pv=PROF[td]; bins=pv.sort_index(); mean_v=bins.mean()
    if bins.iloc[-2:].mean()>=.8*mean_v: L_.append(('POOR',float(bins.index[-1])))
    if bins.iloc[:2].mean()>=.8*mean_v: L_.append(('POOR',float(bins.index[0])))
    f=prior(nd)
    if f is not None:
        for nm in ('call_wall_nq','put_wall_nq'):
            if np.isfinite(f[nm]) and abs(f[nm]-pc)<3*at: L_.append(('OPT',float(f[nm])))
    L_=[(gn,float(p)) for gn,p in L_ if np.isfinite(p) and abs(p-pc)<=1.5*at]
    LEV[nd]=(np.array([p for _,p in L_]),np.array([gn for gn,_ in L_]),at)
print('days with levels',len(LEV),'| mean levels/day',np.mean([len(x[0]) for x in LEV.values()]).round(1),'| by group',pd.Series(np.concatenate([x[1] for x in LEV.values()])).value_counts().to_dict(),flush=True)
GRP=['VPd','VP3','VPw','VPm','EXT','SWG','FIB','OPT','LIQ','POOR']; rows=[]
for td,(P,GN,at) in LEV.items():
    tol=.1*at; a=G.iloc[G.index.searchsorted(td)-1] if G.index.searchsorted(td)>0 else np.nan; used={}; fv=prior(td)
    ivh=bool(fv is not None and np.isfinite(fv.ivp) and fv.ivp>=.5); neg=bool(np.isfinite(a) and a<0); erv=er_.get(td,np.nan)
    ixs=[j for j in dayix.get(td,[]) if 570<=m[j]<=955]; hod=-1e18; lod=1e18; hodj=lodj=0
    for j in ixs:
        if m[j]>=600 and m[j]<=900 and j+1<n and j>=14 and np.isfinite(hod):
            rg=max(h[j]-l[j],1e-9); e=j+1; up=(h[j]-max(o[j],c[j]))/rg>=.5; dn=(min(o[j],c[j])-l[j])/rg>=.5
            for s,ext,okw in ((-1,h[j],up),(1,l[j],dn)):
                if not okw: continue
                msk=(c[j-1]<P)&(P<=h[j])&(c[j]<P) if s==-1 else (c[j-1]>P)&(P>=l[j])&(c[j]>P)
                if not msk.any(): continue
                R=abs(o[e]-ext)+1
                if R<10: continue
                ids=[q for q in np.where(msk)[0] if (q,s) not in used]
                if not ids: continue
                for q in ids: used[(q,s)]=1
                dl3=cd[j-1]-cd[j-4]; vs3=v[j-3:j].sum(); gs=set(GN[ids]); ng=len(gs)
                r=dict(e=e,s=s,R=R,day=td,ng=ng,nl=len(ids),ab=bool(np.sign(dl3)==-s) and bool(np.isfinite(rv[j]) and v[j]>=1.3*rv[j]),ad=float(np.sign(dl3)==-s),dstr=float(abs(dl3)/max(vs3,1))*(1 if np.sign(dl3)==-s else -1),rvol=float(v[j]/rv[j]) if np.isfinite(rv[j]) and rv[j]>0 else 1.0,
                       neg=float(neg),iv=float(ivh),er=float(erv) if np.isfinite(erv) else .3,hrs=(m[j]-570)/60,side=float(s),rat=R/at,dhod=(hod-ext)/at if s==-1 else (hod-o[e])/at,dlod=(o[e]-lod)/at if s==1 else (ext-lod)/at,athod=float(s==-1 and h[j]>=hod),atlod=float(s==1 and l[j]<=lod))
                for gname in GRP: r['g_'+gname]=float(gname in gs)
                rows.append(r)
        if np.isfinite(h[j]): 
            if h[j]>hod: hod=h[j]
            if l[j]<lod: lod=l[j]
C=pd.DataFrame(rows).drop_duplicates(['e','s']).sort_values('e').reset_index(drop=True); print('candidates',len(C),'| with stage 3',int(C.ab.sum()),flush=True)
res=[trade(int(r.e),int(r.s),r.R,'fix',2.0) for r in C.itertuples()]; C['pts']=[x[0]-1.0 for x in res]; C['xj']=[x[2] for x in res]; C['win']=(C.pts>0).astype(int); C['y']=C.day.dt.year
FEA=['ng','nl','ad','dstr','rvol','neg','iv','er','hrs','side','rat','dhod','dlod','athod','atlod']+['g_'+x for x in GRP]
tr=(C.y>=2019)&(C.y<=2021); te=(C.y==2018)|(C.y>=2022); X=C[FEA].astype(float).fillna(0); X['rvol']=X.rvol.clip(upper=6); X['dhod']=X.dhod.clip(-5,5); X['dlod']=X.dlod.clip(-5,5); X['rat']=X.rat.clip(upper=1.5)
from sklearn.linear_model import LogisticRegression; from sklearn.ensemble import GradientBoostingClassifier; from sklearn.metrics import roc_auc_score
mu=X[tr].mean(); sd=X[tr].std().replace(0,1); Z=(X-mu)/sd
lr=LogisticRegression(C=.05,max_iter=500).fit(Z[tr],C.win[tr]); gb=GradientBoostingClassifier(max_depth=2,n_estimators=150,learning_rate=.05,subsample=.7,random_state=0).fit(X[tr],C.win[tr])
PR={'LOGIT':lr.predict_proba(Z)[:,1],'GBM':gb.predict_proba(X)[:,1]}
print('train candidates',int(tr.sum()),'win',round(C.win[tr].mean(),3),'| test candidates',int(te.sum()),'win',round(C.win[te].mean(),3))
for nm,p in PR.items(): print(f'{nm}: AUC train {roc_auc_score(C.win[tr],p[tr.values]):.3f} | test {roc_auc_score(C.win[te],p[te.values]):.3f}')
print('LOGIT largest standardised weights:',', '.join(f'{f} {w:+.2f}' for f,w in sorted(zip(FEA,lr.coef_[0]),key=lambda t:-abs(t[1]))[:10]))
print('GBM importances:',', '.join(f'{f} {w:.2f}' for f,w in sorted(zip(FEA,gb.feature_importances_),key=lambda t:-t[1])[:10]))
def tstat(x):
    d=x.groupby('day').pts.sum(); return d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if len(d)>5 else np.nan
def seqrun(M):
    out=[]; last=-1
    for r in C[M].itertuples():
        if int(r.e)<=last: continue
        out.append(dict(day=r.day,pts=r.pts)); last=int(r.xj)
    return pd.DataFrame(out)
def cell(nm,D):
    y=D.day.dt.year; res=[]
    for era,mk,yrs in (('train',(y>=2019)&(y<=2021),3),('test',(y==2018)|(y>=2022),6.55)):
        x=D[mk]; res.append(f'{era}: n{len(x):4d} ({len(x)/yrs:4.0f}/yr) win{np.mean(x.pts>0)*100:3.0f}% net {x.pts.mean():+6.1f} t{tstat(x):+5.1f}' if len(x)>5 else f'{era}: n{len(x)}')
    print(f'{nm:30s}',' || '.join(res),flush=True)
def byyear(nm,D): D=D.assign(y=D.day.dt.year); print(f'   {nm} by year:',' '.join(f'{y}:{g.pts.mean():+.0f}({len(g)})' for y,g in D.groupby('y')))
print('\n=== 2R fades, sequential, out-of-sample = test columns ===')
cell('(ref) all stage-2 rejections',seqrun(C.e>0)); cell('(ref) stage 3 rule (ab)',seqrun(C.ab))
for nm,p in PR.items():
    for q in (.9,.8,.7):
        thr=np.quantile(p[tr.values],q); cell(f'{nm} top {int((1-q)*100)}%',seqrun(p>=thr))
    byyear(f'{nm} top 20%',seqrun(p>=np.quantile(p[tr.values],.8)))
