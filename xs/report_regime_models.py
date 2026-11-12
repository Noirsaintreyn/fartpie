"""REGIME MODELS for auction-market states on NQ (declared before results). Question: can a data-science classifier tell, at 10:00/10:30/11:30, whether the day ACCEPTS NEW VALUE (closes outside prior-day value area, value migrating)
or stays in BALANCE (mean-reversion day)?  Y = RTH close outside prior-day VA(70%) [accept] vs inside [balance].
Ten models, each refit in price only: 1 SVM-RBF, 2 random forest, 3 HMM(3-state, 30m bars), 4 Gaussian mixture, 5 Kalman local-trend t-stat, 6 CUSUM change-point, 7 OU reversion speed vs VWAP, 8 variance ratio, 9 permutation entropy, 10 one-class SVM value region (fit on prior 3 days).
Scored: AUC vs Y by era; and the money test: mean-reversion payoff from the call time to the close (fade price vs prior-day POC), lowest vs highest third of the model's accept-score. Supervised/learned models refit each year (test 2019+)."""
import numpy as np, pandas as pd, warnings, math, itertools; warnings.filterwarnings('ignore')
from sklearn.svm import SVC, OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from hmmlearn.hmm import GaussianHMM
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').sort_values('dt').drop_duplicates('dt').reset_index(drop=True)
mn=b.dt.dt.hour*60+b.dt.dt.minute; b['d']=b.dt.dt.normalize(); rth=b[(mn>=570)&(mn<960)].copy()
D=rth.groupby('d').agg(o=('open','first'),h=('high','max'),l=('low','min'),c=('close','last'),v=('volume','sum')); D['rng']=D.h-D.l
D['adr']=D.rng.rolling(20).mean().shift(); D['pdh']=D.h.shift(); D['pdl']=D.l.shift(); D['pc']=D.c.shift(); D['pv']=D.v.shift()
G={d:g.reset_index(drop=True) for d,g in rth.groupby('d')}; days=list(D.index)
def vp(g,bin_=2.0):
    tp=((g.high+g.low+g.close)/3).values; v=g.volume.values; lo=np.floor(tp.min()/bin_)*bin_; k=((tp-lo)/bin_).astype(int); h=np.bincount(k,weights=v); poc=int(h.argmax()); tot=h.sum(); a=b_=poc; acc=h[poc]
    while acc<.7*tot and (a>0 or b_<len(h)-1):
        up=h[b_+1] if b_<len(h)-1 else -1; dn=h[a-1] if a>0 else -1
        if up>=dn: b_+=1; acc+=up
        else: a-=1; acc+=dn
    return lo+poc*bin_+bin_/2, lo+(b_+1)*bin_, lo+a*bin_
VP={days[i]:vp(G[days[i-1]]) for i in range(1,len(days)) if days[i-1] in G}
TS=(30,60,120)
def pe3(x):
    if len(x)<5: return np.nan
    pats={}; 
    for i in range(len(x)-2):
        k=tuple(np.argsort(x[i:i+3])); pats[k]=pats.get(k,0)+1
    p=np.array(list(pats.values()))/sum(pats.values()); return -(p*np.log(p)).sum()/math.log(6)
def kalman_t(c,adr):
    # local linear trend, level+slope; q ratios relative to adr
    x=np.array([c[0],0.0]); P=np.diag([ (.02*adr)**2, (.002*adr)**2 ]); F=np.array([[1,1],[0,1]]); H=np.array([1,0]); R=(.01*adr)**2; Q=np.diag([(.004*adr)**2,(.0006*adr)**2])
    for z in c[1:]:
        x=F@x; P=F@P@F.T+Q; S=H@P@H+R; K=P@H/S; x=x+K*(z-H@x); P=P-np.outer(K,H@P)
    return x[1]/math.sqrt(max(P[1,1],1e-12))
def cusum(c,sd):
    r=np.diff(c)/sd; sp=sn=0.0; m=0.0
    for v in r: sp=max(0,sp+v-.1); sn=max(0,sn-v-.1); m=max(m,sp,sn)
    return m
rows=[]; HM={}                                   # HM: per day 30m obs sequence
for i,d in enumerate(days):
    if i<25 or d not in G or d not in VP or not D.adr[d]==D.adr[d]: continue
    g=G[d]; adr=D.adr[d]; pdh,pdl=D.pdh[d],D.pdl[d]; poc,vah,val=VP[d]; op=g.open.iloc[0]; pc=D.pc[d]; m_=(g.dt.dt.hour*60+g.dt.dt.minute-570).values
    cl=g.close.values; hi=g.high.values; lo=g.low.values; C=cl[-1]; Y=float(C>vah or C<val)
    # prior 3 days closes for OCSVM
    prev=np.concatenate([G[days[j]].close.values for j in range(i-3,i) if days[j] in G])/adr
    oc=OneClassSVM(nu=.15,gamma='scale').fit(prev.reshape(-1,1)) if len(prev)>50 else None
    bars30=g.groupby(m_//30).agg(o=('open','first'),h=('high','max'),l=('low','min'),c=('close','last')); HM[d]=np.c_[(bars30.c-bars30.o)/adr,(bars30.h-bars30.l)/adr,((bars30.c-op)/adr)]
    base=dict(day=d,Y=Y,gap_rel=abs(op-pc)/adr,ins=float(pdl<op<pdh),opos=(op-pdl)/(pdh-pdl),dpoc=abs(op-poc)/adr,inva=float(val<=op<=vah),varng=(vah-val)/adr,rr=D.rng.shift()[d]/adr)
    vw=((g.high+g.low+g.close)/3*g.volume).cumsum()/g.volume.cumsum()
    for T in TS:
        k=T//5; 
        if k+2>=len(g): continue
        x=cl[:k]; f=dict(base); f['T']=T; PT=g.open.iloc[k]
        rg=max(hi[:k].max()-lo[:k].min(),1e-6); f.update(rng=rg/adr,eff=abs(x[-1]-op)/rg,disp=abs(x[-1]-op)/adr,pos=(x[-1]-pdl)/(pdh-pdl),tch=float(hi[:k].max()>=pdh)+float(lo[:k].min()<=pdl),
            cross=float(((x[1:]>op)!=(x[:-1]>op)).sum()),vpace=g.volume.values[:k].sum()/D.pv[d],incl=float(val<=x[-1]<=vah),dpoc_t=abs(x[-1]-poc)/adr)
        r=np.diff(x); sd=max(r.std(),1e-6); dp=np.diff(x)
        # 5 kalman, 6 cusum, 7 OU, 8 VR, 9 PE, 10 OCSVM
        f['m5_kalman']=abs(kalman_t(x,adr)); f['m6_cusum']=cusum(x,sd)
        dev=(x[:-1]-vw.values[:k-1]); b1=np.polyfit(dev,dp,1)[0] if dev.std()>0 else 0.0; f['m7_ou']=b1
        v3=np.array([x[j+2]-x[j] for j in range(len(x)-2)]); f['m8_vr']=v3.var()/(2*max(r.var(),1e-9)) if len(v3)>3 else np.nan
        f['m9_pe']=-pe3(r); f['m10_oc']=float((oc.decision_function((x/adr).reshape(-1,1))<0).mean()) if oc is not None else np.nan
        f['pay_poc']=-np.sign(PT-poc)*(C-PT)-.5; f['pay_vwap']=-np.sign(PT-vw.iloc[k-1])*(C-PT)-.5
        f['k30']=T//30
        rows.append(f)
F=pd.DataFrame(rows).dropna(subset=['gap_rel','rr','m8_vr','m10_oc']); F['yr']=pd.to_datetime(F.day).dt.year
cols=['gap_rel','ins','opos','dpoc','inva','varng','rr','rng','eff','disp','pos','tch','cross','vpace','incl','dpoc_t']
SCORE={}
for T in TS:
    z=F[F['T']==T].copy(); z['m1_svm']=np.nan; z['m2_rf']=np.nan; z['m3_hmm']=np.nan; z['m4_gmm']=np.nan
    for y in range(2019,2027):
        tr=z[z.yr<y]; te=z[z.yr==y]
        if len(te)==0 or len(tr)<300: continue
        sc=StandardScaler().fit(tr[cols]); Xtr=sc.transform(tr[cols]); Xte=sc.transform(te[cols])
        z.loc[te.index,'m1_svm']=SVC(C=1.0,gamma='scale',probability=True,random_state=0).fit(Xtr,tr.Y).predict_proba(Xte)[:,1]
        z.loc[te.index,'m2_rf']=RandomForestClassifier(300,min_samples_leaf=15,random_state=0,n_jobs=-1).fit(Xtr,tr.Y).predict_proba(Xte)[:,1]
        gm=GaussianMixture(4,covariance_type='diag',random_state=0,n_init=3).fit(Xtr); rt=gm.predict_proba(Xtr); ym=(rt*tr.Y.values[:,None]).sum(0)/rt.sum(0); z.loc[te.index,'m4_gmm']=gm.predict_proba(Xte)@ym
        seqs=[HM[d] for d in tr.day if d in HM and len(HM[d])>=8]; L=[len(s) for s in seqs]
        hm=GaussianHMM(3,covariance_type='diag',n_iter=40,random_state=0).fit(np.vstack(seqs),L); hi_state=int(np.argmax(hm.means_[:,1]))      # state with widest 30m ranges
        # state most associated with directional progress
        sa=np.argmax(np.abs(hm.means_[:,2]))
        out=[]
        for d in te.day:
            s=HM[d][:max(T//30,1)]; out.append(hm.predict_proba(s)[-1,sa])
        z.loc[te.index,'m3_hmm']=out
    SCORE[T]=z
# ---- scoring
MODELS=['m1_svm','m2_rf','m3_hmm','m4_gmm','m5_kalman','m6_cusum','m7_ou','m8_vr','m9_pe','m10_oc','eff','incl']
NAME={'m1_svm':'1 SVM-RBF','m2_rf':'2 random forest','m3_hmm':'3 HMM 3-state','m4_gmm':'4 Gaussian mixture','m5_kalman':'5 Kalman trend t','m6_cusum':'6 CUSUM change-point','m7_ou':'7 OU reversion speed','m8_vr':'8 variance ratio','m9_pe':'9 perm. entropy (neg)','m10_oc':'10 one-class SVM outside','eff':'(base) path efficiency','incl':'(base) inside prior VA'}
print('AUC vs Y (accept new value = close outside prior VA; base share Y = %.0f%%)'%(100*F[F['T']==60].Y.mean()))
for T in TS:
    z=SCORE[T]; z=z[z.yr>=2019]
    print(f'--- call at +{T}m (= {9+ (30+T)//60}:{(30+T)%60:02d})')
    for m in MODELS:
        q=z.dropna(subset=[m]); out=[]
        for e,mm in (('19-21',q.yr<=2021),('22-26',q.yr>=2022)):
            a=q[mm]
            if len(a)<60: out.append(f'{e}: n/a (n={len(a)})'); continue
            auc=roc_auc_score(a.Y,a[m]); t3=a[m].quantile([1/3,2/3]).values; lo_=a[a[m]<=t3[0]]; hi_=a[a[m]>=t3[1]]
            d=(lo_.pay_poc.mean()-hi_.pay_poc.mean()); se=math.sqrt(lo_.pay_poc.var()/len(lo_)+hi_.pay_poc.var()/len(hi_))
            out.append(f'{e}: AUC {auc:.3f}  MR payoff lowest3rd {lo_.pay_poc.mean():+6.1f} vs highest3rd {hi_.pay_poc.mean():+6.1f} pts (diff t {d/se:+.1f})')
        print(f'  {NAME[m]:26s} '+' | '.join(out))
pickle=__import__('pickle'); pickle.dump(SCORE,open('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/regime_scores.pkl','wb'))
