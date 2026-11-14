"""Nov 14 2026 (Rey's idea): if the EXPECTED RANGE is inside the previous day's VALUE, play mean reversion; if it exceeds value, play continuation. DECLARED BEFORE RESULTS.
rho_D = ADR20 (mean RTH range of the prior 20 sessions, PIT) / prior-day value-area width (VAH-VAL, 70% volume profile). Low rho = expected range small vs value ('fits inside value', balance); high rho = expected range large vs value (imbalance).
PIT rank: rho percentile vs ALL PRIOR days (expanding, >=120 days). LOW = rank<=1/3, HIGH = rank>=2/3.
Mechanism test first: does LOW rho predict a rotational day (close inside prior VA, lower RTH efficiency, price stays within prior VA band)? Then models: mean-reversion models (VA-F, INV-F) should win in LOW, continuation models (VA-B, IB-B) in HIGH. Report by strict eras and by the five declared regimes. PASS = predicted sign in BOTH eras."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/fartpie/xs/report_19_amt_papers.py').read().split("def sim(")[0].split('"""',2)[2])
dl=[d for d in days if d in RTH]; rows=[]; rhos=[]
for i,D in enumerate(dl):
    if i<25 or (D-dl[i-1]).days>5 or dl[i-1] not in PRO: continue
    P=dl[i-1]; pO,vah,val=PRO[P]; w=vah-val
    if not w>0: continue
    adr=np.mean([RTH[d].high.max()-RTH[d].low.min() for d in dl[i-20:i]]); g=RTH[D]; rho=adr/w
    hi,lo,cl,op=g.high.max(),g.low.min(),g.close.values[-1],g.open.values[0]
    mv=g.close.diff().abs().sum()
    rows.append(dict(day=D,rho=rho,adr=adr,w=w,in_close=val<=cl<=vah,in_open=val<=op<=vah,stay=(hi<=vah+.0*w)&(lo>=val-.0*w),rng=hi-lo,eff=abs(cl-op)/mv if mv>0 else np.nan,over_va=max(0,hi-vah)+max(0,val-lo)))
R=pd.DataFrame(rows).set_index('day'); R['rk']=[(R.rho.iloc[:j]<R.rho.iloc[j]).mean() if j>=120 else np.nan for j in range(len(R))]
R['b']=np.where(R.rk<=1/3,'LOW',np.where(R.rk>=2/3,'HIGH','MID')); R.loc[R.rk.isna(),'b']=None
RGS=[('R1','2018-01-01','2020-03-01'),('R2','2020-03-01','2022-01-01'),('R3','2022-01-01','2023-04-01'),('R4','2023-04-01','2025-04-01'),('R5','2025-04-01','2027-01-01')]
rg=lambda t: next((n for n,a,b in RGS if pd.Timestamp(a)<=t<pd.Timestamp(b)),None)
ERA={'18-21':('2018-01-01','2022-01-01'),'22+':('2022-01-01','2027-01-01')}
print('days',len(R),'| median rho',round(R.rho.median(),2),'| median VA width',round(R.w.median(),1),'ADR',round(R.adr.median(),1))
print('\n#### MECHANISM: does LOW rho (expected range fits inside value) give a rotational day?')
for e,(a,b) in ERA.items():
    s=R[(R.index>=a)&(R.index<b)&R.b.notna()]
    for bk in ('LOW','MID','HIGH'):
        x=s[s.b==bk]; print(f'  {e:6s} {bk:4s} n={len(x):4d} | close inside prior VA {x.in_close.mean():.0%} | open inside {x.in_open.mean():.0%} | day range {x.rng.mean():6.1f} ({x.rng.mean()/x.adr.mean():.2f} x ADR) | pts beyond VA edges {x.over_va.mean():6.1f} | efficiency {x.eff.mean():.3f}')
A=pd.read_pickle(SC+'amt_models_T.pkl'); A['day']=pd.to_datetime(A.d).dt.normalize(); A=A.join(R[['b']],on='day')
C=pd.read_pickle(SC+'icc_T.pkl'); C['day']=pd.to_datetime(C.d).dt.normalize(); C=C.join(R[['b']],on='day')
def tt(a,b): return (a.mean()-b.mean())/np.sqrt(a.var()/len(a)+b.var()/len(b)) if len(a)>5 and len(b)>5 else np.nan
def table(df,name,pred):
    print(f'\n{name}  [prediction: {pred}]')
    for e,(a,b) in ERA.items():
        s=df[(df.day>=a)&(df.day<b)&df.b.notna()]; lo=s[s.b=='LOW'].pts; mi=s[s.b=='MID'].pts; hi=s[s.b=='HIGH'].pts
        print(f'   {e:6s} LOW {lo.mean():+7.2f} (n{len(lo)}) | MID {mi.mean():+7.2f} (n{len(mi)}) | HIGH {hi.mean():+7.2f} (n{len(hi)}) | LOW-HIGH {lo.mean()-hi.mean():+6.2f} t {tt(lo,hi):+.1f}')
    line='   by regime LOW/HIGH: '
    for n,_,_ in RGS:
        s=df[(df.day.map(rg)==n)&df.b.notna()]; lo=s[s.b=='LOW'].pts; hi=s[s.b=='HIGH'].pts; line+=f'{n} {lo.mean():+.1f}/{hi.mean():+.1f} (n{len(lo)}/{len(hi)})  '
    print(line)
for m,pr in (('VA-F','LOW > HIGH (fade works when range fits in value)'),('INV-F','LOW > HIGH'),('VA-B','HIGH > LOW (continuation when range exceeds value)'),('IB-B','HIGH > LOW'),('VA-80','?'),('SHAPE','?')): table(A[A.model==m],f'NQ {m} (pts/trade)',pr)
table(C,'ICT chain (pts/trade)','HIGH > LOW if chain is directional')
import pickle
S=pickle.load(open(SC+'s6.pkl','rb')); legs=S[0]['V2']; nm=S[1]; allp=[]
for (T,u,_),n in zip(legs,nm):
    x=T.copy(); x['day']=pd.to_datetime(x.day).dt.normalize(); x['pts']=x.pts*x.sz*u; x=x.join(R[['b']],on='day'); table(x,f'main-algo leg {n} (pts/trade x unit)','?'); allp.append(x[['day','pts','b']])
d=pd.concat(allp).groupby('day').agg(pts=('pts','sum'),b=('b','first')).reset_index(); table(d,'main algo all legs (pts/DAY)','?')
R.to_pickle(SC+'rho_R.pkl')
