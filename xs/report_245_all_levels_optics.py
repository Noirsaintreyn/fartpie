"""Report 245: EVERY level source + an OPTICS clustering level finder, three-stage reaction (Rey Nov 17: use everything - swing points, extremes, LVN, ledges, shelves, volume profiles on different time frames - and an auction-market clustering model like OPTICS instead of hand-picking VP levels; if VP does not work drop it; the aim is MORE TRADES). DECLARED before results.
Level groups (all from sessions BEFORE the trade day; only levels within 1.5 x ATR14 of the prior close): VPd/VP3/VPw/VPm = volume profile (2-pt bins) of the prior 1 / 3 / 5 / 20 RTH sessions: POC, VAH, VAL (70%), up to 3 HVN, up to 3 LVN; EXT = prior-day H/L, overnight H/L, 5-day H/L, 20-day H/L; SWG = 1H swing highs/lows (3-bar fractals, confirmed, last ~5 sessions) = ledges; FIB = 38.2/50/61.8 of the prior-day and the 5-day range; OPT = call / put wall; CLU = OPTICS clusters (min_samples 20, xi .05) of the volume-replicated price distribution of the last 3 and last 10 sessions: cluster centroid and the two edges (3rd / 97th pct) = acceptance zones and their shelves.
Reaction = report 244 stages: stage 2 = 5m bar after 10:00 touches the level from the near side and closes back with a >= 50% wick (fade at the next open, stop = bar extreme + 1, R >= 10, one trade at a time, 1 pt cost); stage 3 = delta of the 3 bars into the level in the push direction AND touch-bar volume >= 1.3 x prior-12 mean. Targets 1R / 2R.
Cells: stage 2+3 by level group (2R); pooled over ALL groups (trade count per year, net, t) and pooled WITHOUT any VP group, WITHOUT VP and CLU, VP groups only, CLU only; by confluence = number of distinct groups touched by the same bar (1, 2, 3+); chop gate; by year. The questions: does it add trades, does the edge hold, and which sources matter. Train 2019-21 | test 2018+2022-26; ~35 cells -> bar ~3."""
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
    if GH[i]>GH[i-3:i].max() and GH[i]>=GH[i+1:i+4].max(): sw.append((GE[i+3],GH[i]))
    if GL[i]<GL[i-3:i].min() and GL[i]<=GL[i+1:i+4].min(): sw.append((GE[i+3],GL[i]))
sw=sorted(sw); swt=np.array([x[0] for x in sw]); swp=np.array([x[1] for x in sw])
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
    f=prior(nd)
    if f is not None:
        for nm in ('call_wall_nq','put_wall_nq'):
            if np.isfinite(f[nm]) and abs(f[nm]-pc)<3*at: L_.append(('OPT',float(f[nm])))
    for W in (3,10):
        for nm,p in optics_levels(comp(i,W)): L_.append(('CLU',p))
    L_=[(gn,float(p)) for gn,p in L_ if np.isfinite(p) and abs(p-pc)<=1.5*at]
    LEV[nd]=(np.array([p for _,p in L_]),np.array([gn for gn,_ in L_]),at)
print('days with levels',len(LEV),'| mean levels/day',np.mean([len(x[0]) for x in LEV.values()]).round(1),'| by group',pd.Series(np.concatenate([x[1] for x in LEV.values()])).value_counts().to_dict(),flush=True)
GR=['VPd','VP3','VPw','VPm','EXT','SWG','FIB','OPT','CLU']; rows=[]
for td,(P,GN,at) in LEV.items():
    tol=.1*at; a=G.iloc[G.index.searchsorted(td)-1] if G.index.searchsorted(td)>0 else np.nan; used={}
    for j in dayix.get(td,[]):
        if m[j]<600 or m[j]>900 or j+1>=n or j<14: continue
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
            dl3=cd[j-1]-cd[j-4]; ab=bool(np.sign(dl3)==-s) and bool(np.isfinite(rv[j]) and v[j]>=1.3*rv[j]); gs=set(GN[ids]); ng=len(gs)
            for gname in gs: rows.append((e,s,R,td,ab,gname,ng,bool(np.isfinite(er_.get(td,np.nan)) and er_[td]<q1)))
C=pd.DataFrame(rows,columns=['e','s','R','day','ab','grp','ng','chop']); print('rejection rows',len(C),'| unique trades',len(C.drop_duplicates(['e','s'])),'| absorbed unique',int(C[C.ab].drop_duplicates(['e','s']).shape[0]),flush=True)
def run(cs,k):
    out=[]; last=-1
    for _,r in cs.sort_values('e').iterrows():
        e=int(r.e)
        if e<=last or e>=n-2 or not (575<=m[e]<=930): continue
        gp,mfe,j=trade(e,int(r.s),r.R,'fix',k); out.append(dict(day=r.day,pts=gp-1.0)); last=j
    return pd.DataFrame(out)
def cell(nm,D):
    y=D.day.dt.year; res=[]
    for era,mk,yrs in (('train',(y>=2019)&(y<=2021),3),('test',(y==2018)|(y>=2022),6.55)):
        x=D[mk]; res.append(f'{era}: n{len(x):4d} ({len(x)/yrs:4.0f}/yr) win{np.mean(x.pts>0)*100:3.0f}% net {x.pts.mean():+6.1f} t{tstat(x):+5.1f}' if len(x)>5 else f'{era}: n{len(x)}')
    print(f'{nm:32s}',' || '.join(res),flush=True)
def byyear(nm,D): D=D.assign(y=D.day.dt.year); print(f'   {nm} by year:',' '.join(f'{y}:{g.pts.mean():+.0f}({len(g)})' for y,g in D.groupby('y')))
A=C[C.ab]; uniq=lambda X:X.drop_duplicates(['e','s'])
for k in (1.0,2.0):
    print(f'\n=== target {k}R, stage 2+3 ===')
    for gname in GR: cell(f'group {gname}',run(A[A.grp==gname],k))
    cell('ALL groups pooled',run(uniq(A),k))
    cell('without any VP',run(uniq(A[~A.grp.isin(['VPd','VP3','VPw','VPm'])]),k)); cell('without VP and CLU',run(uniq(A[~A.grp.isin(['VPd','VP3','VPw','VPm','CLU'])]),k))
    cell('VP groups only',run(uniq(A[A.grp.isin(['VPd','VP3','VPw','VPm'])]),k)); cell('CLU only',run(uniq(A[A.grp=='CLU']),k)); cell('EXT+SWG+FIB+OPT only',run(uniq(A[A.grp.isin(['EXT','SWG','FIB','OPT'])]),k))
    U=uniq(A); cell('confluence 1 group',run(U[U.ng==1],k)); cell('confluence 2 groups',run(U[U.ng==2],k)); cell('confluence 3+ groups',run(U[U.ng>=3],k)); cell('ALL pooled & chop gate',run(U[U.chop],k))
    cell('(ref) stage 2 only, all groups',run(uniq(C),k))
    byyear('ALL pooled stage 3',run(uniq(A),k)); byyear('confluence 2+',run(U[U.ng>=2],k))
