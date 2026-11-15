"""report_121 (Nov 15 2026): STEP C. Rule-based regime state machine on NQ daily features vs GMM vs HMM. All identifiers FIT ON TRAIN (<=2021) ONLY, features lagged 1 day (known at the open).
Declared hypotheses (before results): expansion/trend-state legs RA, V15a, SHAPE, ICT-strong need EXPANSION (vol up); mean-revert legs VA-B, IB-B, M1 need QUIET/BALANCED. Test: states sorted on TRAIN pts/trade, same best-minus-worst spread measured on TEST.
Null: circular shift of the state series (keeps persistence), 1000 draws. Pass bar: more legs with positive test spread than null 95th pct AND sum of test spread t above null."""
import pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.mixture import GaussianMixture
from hmmlearn.hmm import GaussianHMM
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
D=pd.read_pickle(SC+'atlas_D.pkl'); D.index=pd.to_datetime(D.index)
q=pd.read_parquet('/tmp/claude-0/ml/nq5_real.parquet'); q['day']=pd.to_datetime(q.day)
rv=q[q.rth].groupby('day').volume.sum().reindex(D.index)
rng=(D.hi-D.lo)
F=pd.DataFrame(index=D.index)
F['volx']=np.log(rng.rolling(5).mean()/rng.rolling(60).mean())              # vol expansion
F['vlvl']=np.log(rng.rolling(14).mean()/rng.rolling(250,min_periods=120).median())  # vol level
F['eff']=D.eff.rolling(10).mean()                                            # trend efficiency
F['rvol']=np.log(rv.ewm(span=5).mean()/rv.ewm(span=60).mean())              # volume EMA ratio
F['inv']=D.strong.astype(float).rolling(10).mean()                           # overnight-inventory share
F['cont']=D.cont.rolling(10).mean()
F=F.shift(1).dropna()
TR=F.index<'2022-01-01'
def hyst(x,hi,mid,lo):
    s=[];cur=1
    for v in x:
        if cur!=2 and v>hi: cur=2
        elif cur==2 and v<mid: cur=1
        if cur!=0 and v<lo: cur=0
        elif cur==0 and v>mid: cur=1
        s.append(cur)
    return np.array(s)
# --- own state machine: vol axis w/ hysteresis (train terciles), trend axis by train median eff
v=F.volx; hi,lo,mid=v[TR].quantile(.67),v[TR].quantile(.33),v[TR].median()
vs=hyst(v.values,hi,mid,lo); te=(F.eff>F.eff[TR].median()).values
SM=pd.Series([f"{['QUIET','MID','EXP'][a]}-{'TRND' if b else 'BAL'}" for a,b in zip(vs,te)],index=F.index)
SMv=pd.Series([['QUIET','MID','EXP'][a] for a in vs],index=F.index)   # vol-only 3-state
# --- GMM / HMM on standardized features, train fit
Z=(F-F[TR].mean())/F[TR].std()
g=GaussianMixture(4,covariance_type='full',n_init=5,random_state=0).fit(Z[TR]); GM=pd.Series(g.predict(Z),index=F.index).astype(str)
h=GaussianHMM(3,covariance_type='diag',n_iter=200,random_state=0).fit(Z[TR].values); HM=pd.Series(h.predict(Z.values),index=F.index).astype(str)
# --- trades
S=pickle.load(open(SC+'s6.pkl','rb')); legs=S[0]['V2']; nm=S[1]; X={}
for (T,u,_),n in zip(legs,nm): x=T.copy(); x['d']=pd.to_datetime(x.day).dt.normalize(); X[n]=pd.DataFrame({'d':x.d,'p':x.pts*x.sz*u})
A=pd.read_pickle(SC+'amt_models_T.pkl'); A['d']=pd.to_datetime(A.d).dt.normalize()
for m in ('VA-B','VA-F','VA-80','IB-B','INV-F','SHAPE'): a=A[A.model==m]; X[m]=pd.DataFrame({'d':a.d,'p':a.pts})
C=pd.read_pickle(SC+'icc_T.pkl'); C['d']=pd.to_datetime(C.d).dt.normalize(); C=C.join(D[['on','strong']],on='d'); X['ICT']=pd.DataFrame({'d':C.d,'p':C.pts}); X['ICT-strong']=pd.DataFrame({'d':C.d[C.strong==True],'p':C.pts[C.strong==True]})
X={k:v_[v_.d.isin(F.index)].reset_index(drop=True) for k,v_ in X.items()}
def spread(st,df,minn=25):
    s=st.reindex(df.d).values; tr=(df.d<'2022-01-01').values; ok=~pd.isna(s)
    g=pd.DataFrame({'s':s,'p':df.p.values,'tr':tr})[ok]
    a=g[g.tr].groupby('s').p.agg(['mean','count']); a=a[a['count']>=minn]
    if len(a)<2: return None
    b,w=a['mean'].idxmax(),a['mean'].idxmin()
    t=g[~g.tr]; pb,pw=t[t.s==b].p,t[t.s==w].p
    if len(pb)<15 or len(pw)<15: return None
    d=pb.mean()-pw.mean(); se=np.sqrt(pb.var()/len(pb)+pw.var()/len(pw)); return a.loc[b,'mean']-a.loc[w,'mean'],d,d/se,b,w
rs=np.random.default_rng(1)
def evaluate(name,st,verbose=True):
    rows={k:spread(st,df) for k,df in X.items()}; rows={k:r for k,r in rows.items() if r}
    pos=sum(r[1]>0 for r in rows.values()); tsum=sum(r[2] for r in rows.values())
    nullp=[];nullt=[];idx=st.index;arr=st.values
    for _ in range(1000):
        k=rs.integers(60,len(arr)-60); sh=pd.Series(np.roll(arr,k),index=idx); rr=[spread(sh,df) for df in X.values()]; rr=[r for r in rr if r]
        nullp.append(sum(r[1]>0 for r in rr)); nullt.append(sum(r[2] for r in rr))
    print(f'\n== {name}: legs scored {len(rows)} | test spread>0 in {pos} | sum t {tsum:+.1f} | null: legs>0 95th {np.percentile(nullp,95):.0f}, sum t 95th {np.percentile(nullt,95):+.1f}; p(sum t) {np.mean(np.array(nullt)>=tsum):.3f}')
    if verbose:
        for k,r in rows.items(): print(f'   {k:11s} train best-worst {r[0]:+6.2f} ({r[3]} vs {r[4]}) -> TEST {r[1]:+6.2f} t {r[2]:+.1f}')
    return rows
print('state counts own SM (train/test):'); print(pd.crosstab(SM,TR).to_string())
for nme,st in (('OWN SM vol x trend (6)',SM),('OWN SM vol only (3)',SMv),('GMM K=4',GM),('HMM 3-state',HM)): evaluate(nme,st)
pickle.dump({'F':F,'SM':SM,'SMv':SMv,'GM':GM,'HM':HM},open(SC+'regime_states.pkl','wb'))
