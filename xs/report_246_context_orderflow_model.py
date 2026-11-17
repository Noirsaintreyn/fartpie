"""Report 246: CONTEXT + ORDER FLOW MODEL (Rey Nov 17: levels are context not edges; the story is confluence - call wall at the 1-day HVN, fib at the 5-day POC - plus pivots as liquidity, poor highs/lows, options flow, volatility and order flow; levels that do not matter should be ignored). DECLARED before results.
Candidates = every stage-2 rejection (report 244/245 definition: 5m bar after 10:00 touches a level from the near side and closes back with a >= 50% wick; fade at the next open; stop = bar extreme + 1, R >= 10; outcome = 2R target hit after a 1-pt cost). Levels (prior sessions only): VPd/VP3/VPw/VPm POC-VAH-VAL-HVN-LVN, EXT (prior-day, overnight, 5d, 20d H/L), SWG (1H swing highs/lows = pivot liquidity), EQ (equal highs / equal lows: two same-side swings within 0.05 ATR = liquidity pool), POOR (prior-day high/low with no volume taper in its top / bottom 2 bins = poor high / poor low), FIB (38.2/50/61.8 of prior-day and 5-day range), OPT (call / put wall). OPTICS dropped (report 245: no better than VP).
Features at the signal bar: which level groups sit within 0.1 ATR of the touched level (presence flags, count), order flow (delta of the 3 bars into the level in the push direction, normalised; touch-bar volume ratio; the report-244 absorption flag), state (prior-close net-gamma sign, IV high, chop ER, 1H vector regime aligned with the trade or against), hour, R / ATR. L1 logistic regression (the L1 penalty zeroes out levels and features that do not matter), standardised on the training years.
Validation: (a) train 2019-21, test 2018 + 2022-26; (b) expanding walk-forward: each test year y in 2022-26 is predicted by a model trained on 2019..y-1, 2018 by the 2019-21 model. Selection = candidates whose score is above the training-set 80th percentile (and 90th). Read: AUC, net pts per trade and t of the selected set vs all candidates vs the rule-based stage-3 set, trades per year, by year. Bar ~3 for a headline; consistency across years counts more."""
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
    pf=PROF[td]; q_=pf.reindex(np.arange(pf.index.min(),pf.index.max()+2,2)).fillna(0); md=np.median(q_[q_>0])
    if q_.iloc[-2:].mean()>=md: L_.append(('POOR',pdh))
    if q_.iloc[:2].mean()>=md: L_.append(('POOR',pdl))
    sh=[(float(x)) for x in swp[b_:a_]]; sh=sorted(sh)
    for u_ in range(len(sh)-1):
        if abs(sh[u_+1]-sh[u_])<=.05*at: L_.append(('EQ',(sh[u_]+sh[u_+1])/2))
    L_=[(gn,float(p)) for gn,p in L_ if np.isfinite(p) and abs(p-pc)<=1.5*at]
    LEV[nd]=(np.array([p for _,p in L_]),np.array([gn for gn,_ in L_]),at)
print('days with levels',len(LEV),'| mean levels/day',np.mean([len(x[0]) for x in LEV.values()]).round(1),'| by group',pd.Series(np.concatenate([x[1] for x in LEV.values()])).value_counts().to_dict(),flush=True)
GR=['VPd','VP3','VPw','VPm','EXT','SWG','FIB','OPT','POOR','EQ']; rows=[]
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
            dl3=cd[j-1]-cd[j-4]; vs3=max(v[j-3:j].sum(),1); ab=bool(np.sign(dl3)==-s) and bool(np.isfinite(rv[j]) and v[j]>=1.3*rv[j]); gs=set(GN[ids]); kk=hi1[j]
            rows.append(dict(e=e,s=s,R=R,day=td,ab=ab,ng=len(gs),dpush=float(-s*dl3/vs3),rvr=float(v[j]/rv[j]) if np.isfinite(rv[j]) and rv[j]>0 else 1.0,neg=bool(np.isfinite(a) and a<0),iv=bool(prior(td) is not None and np.isfinite(prior(td).ivp) and prior(td).ivp>=.5),chop=bool(np.isfinite(er_.get(td,np.nan)) and er_[td]<q1),er=float(er_.get(td,np.nan)) if np.isfinite(er_.get(td,np.nan)) else .3,vect=float(reg[kk]*s) if kk>=0 else 0.0,hr=(m[j]-570)/60,Rn=R/at,**{g_:(g_ in gs) for g_ in GR}))
C=pd.DataFrame(rows); print('candidates',len(C),'| absorbed',int(C.ab.sum()),flush=True); C.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/cand246.pkl')

def run(cs,k):
    out=[]; last=-1
    for _,r in cs.sort_values('e').iterrows():
        e=int(r.e)
        if e<=last or e>=n-2 or not (575<=m[e]<=930): continue
        gp,mfe,j=trade(e,int(r.s),r.R,'fix',k); out.append(dict(day=r.day,pts=gp-1.0)); last=j
    return pd.DataFrame(out)
def cell(nm,D,yrs_t=3,yrs_o=6.55):
    y=D.day.dt.year; res=[]
    for era,mk,yy in (('train',(y>=2019)&(y<=2021),yrs_t),('test',(y==2018)|(y>=2022),yrs_o)):
        x=D[mk]; res.append(f'{era}: n{len(x):4d} ({len(x)/yy:4.0f}/yr) win{np.mean(x.pts>0)*100:3.0f}% net {x.pts.mean():+6.1f} t{tstat(x):+5.1f}' if len(x)>5 else f'{era}: n{len(x)}')
    print(f'{nm:34s}',' || '.join(res),flush=True)
def byyear(nm,D): D=D.assign(y=D.day.dt.year); print(f'   {nm} by year:',' '.join(f'{y}:{g.pts.mean():+.0f}({len(g)})' for y,g in D.groupby('y')))
# outcome of every candidate (2R), individually
out=[]
for i_,r in C.iterrows():
    e=int(r.e)
    if e>=n-2 or not (575<=m[e]<=930): out.append(np.nan); continue
    gp,mfe,j=trade(e,int(r.s),r.R,'fix',2.0); out.append(gp-1.0)
C['pts']=out; C=C.dropna(subset=['pts']).reset_index(drop=True); C['win']=(C.pts>0).astype(int); C['y']=C.day.dt.year
FEATS=GR+['ng','dpush','rvr','neg','iv','chop','er','vect','hr','Rn','ab']
X=C[FEATS].astype(float).copy(); X['rvr']=X.rvr.clip(upper=6); X['dpush']=X.dpush.clip(-1,1); X['Rn']=X.Rn.clip(upper=X.Rn.quantile(.99))
from sklearn.linear_model import LogisticRegression
def fitpred(tr_mask,te_mask,Cc=.05):
    mu=X[tr_mask].mean(); sd=X[tr_mask].std().replace(0,1); Ztr=((X[tr_mask]-mu)/sd).values; Zte=((X[te_mask]-mu)/sd).values
    mdl=LogisticRegression(penalty='l1',solver='liblinear',C=Cc,max_iter=500).fit(Ztr,C.win[tr_mask].values)
    return mdl,mdl.decision_function(Ztr),mdl.decision_function(Zte)
def auc(p,y):
    r=pd.Series(p).rank().values; n1=y.sum(); n0=len(y)-n1; return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
tr=(C.y>=2019)&(C.y<=2021); te=(C.y==2018)|(C.y>=2022)
mdl,ptr,pte=fitpred(tr,te)
print('\n(a) TRAIN 2019-21 -> TEST 2018+2022-26: candidates train',int(tr.sum()),'test',int(te.sum()),'| base win rate train',round(C.win[tr].mean(),3),'test',round(C.win[te].mean(),3))
print('L1 coefficients kept:',', '.join(f'{f} {w:+.2f}' for f,w in zip(FEATS,mdl.coef_[0]) if abs(w)>1e-6) or 'none')
print(f'AUC train {auc(ptr,C.win[tr].values):.3f} | test {auc(pte,C.win[te].values):.3f}')
def sel_report(nm,mask_sel,mask_all):
    D=C[mask_all&mask_sel][['day','pts','e','s','R']]; return D
def evalsel(nm,thr_q,ptr,pte,trm,tem,full=True):
    thr=np.quantile(ptr,thr_q); S=C[tem].copy(); S['sc']=pte; S=S[S.sc>=thr]
    D=run(S,2.0); cell(f'{nm} top {int((1-thr_q)*100)}% by score (seq.)',D); return D
cell('ALL candidates (seq., 2R)',run(C[tr|te],2.0))
cell('rule: stage 3 absorption (seq.)',run(C[(tr|te)&C.ab],2.0))
# (a) OOS evaluation on both eras with train-era model scored on train too (train shown for reference)
for q in (.8,.9):
    thr=np.quantile(ptr,q); S_tr=C[tr].copy(); S_tr['sc']=ptr; S_te=C[te].copy(); S_te['sc']=pte; S=pd.concat([S_tr[S_tr.sc>=thr],S_te[S_te.sc>=thr]]); D=run(S,2.0); cell(f'MODEL top {int((1-q)*100)}% (train in-sample | test OOS)',D); 
    if q==.8: DA=D
byyear('MODEL top 20%',DA)
# (b) expanding walk-forward
S_all=[]
for yv in (2018,2022,2023,2024,2025,2026):
    trm=(C.y>=2019)&(C.y<=(2021 if yv==2018 else yv-1)); tem=(C.y==yv)
    if tem.sum()==0: continue
    mdl,ptr_,pte_=fitpred(trm,tem); thr=np.quantile(ptr_,.8); S=C[tem].copy(); S['sc']=pte_; S_all.append(S[S.sc>=thr])
S=pd.concat(S_all); print('\n(b) EXPANDING WALK-FORWARD (2018 by the 2019-21 model; 2022..2026 each by models trained on 2019..y-1): selected',len(S),'of',int(te.sum()))
D=run(S,2.0); y=D.day.dt.year; x=D; print(f'   walk-forward top 20%: n{len(x)} ({len(x)/6.55:.0f}/yr) win{np.mean(x.pts>0)*100:.0f}% net {x.pts.mean():+.1f} t{tstat(x):+.1f}'); byyear('walk-forward top 20%',D)
Tst=C[te]; print('   same years, ALL candidates (seq.):',end=' '); x=run(Tst,2.0); print(f'n{len(x)} net {x.pts.mean():+.1f} t{tstat(x):+.1f}')
