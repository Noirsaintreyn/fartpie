"""Report 262: HAWKES branching ratio (endogeneity) as a day-type predictor. DECLARED before results.
Events: 5m bars from 18:00 prior evening to 11:00 ET whose |return| / (trailing-20-session median |return| for that same clock slot) >= 2.0 (so seasonality is removed). Exponential-kernel Hawkes (mu, alpha, beta) fitted by MLE per day, beta on a grid {0.2,0.5,1,2} per bar; branching ratio n = alpha/beta (max over beta of likelihood), capped .95. Hypothesis: high n (self-excited/endogenous cascade) -> the rest of the day TRENDS (low efficiency); low n -> mean-reverting. Label = efficiency of the REST of the day after 11:00 (as report 254), MR_after = bottom third, TREND_after = top third. Check: AUC of -n for MR_after vs TREND_after by split train 2019-21 | test 2018+2022-26 (consistent side of .5 in both). Also compared: simple event COUNT in the window (control)."""
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b['dt']=pd.to_datetime(b.dt); b=b.set_index('dt').sort_index()
b['r']=np.log(b.close).diff().abs(); b['slot']=b.index.hour*60+b.index.minute
b['sc']=b.groupby('slot').r.transform(lambda s:s.shift(1).rolling(20,min_periods=10).median())  # per-slot trailing median (by bars, ~20 sessions)
b['ev']=(b.r/b.sc>=2.0)
rth=b.between_time('09:30','15:55'); D=rth.groupby(rth.index.normalize())
def nll(par,t,T,beta):
    mu=np.exp(par[0]); n=.95/(1+np.exp(-par[1])); al=n*beta
    R=0.0; ll=0.0
    for k in range(len(t)):
        if k>0: R=np.exp(-beta*(t[k]-t[k-1]))*(R+1)
        ll+=np.log(mu+al*R)
    comp=mu*T+(al/beta)*np.sum(1-np.exp(-beta*(T-t)))
    return -(ll-comp)
rows=[]
for d,g in D:
    if len(g)<70: continue
    w=b.loc[d-pd.Timedelta(hours=6):d+pd.Timedelta(hours=11)-pd.Timedelta(minutes=5)]
    ev=np.where(w.ev.values)[0].astype(float)
    if len(ev)<6: continue
    T=float(len(w)); best=None
    for beta in (.2,.5,1.,2.):
        r=minimize(nll,[np.log(len(ev)/T),0.],args=(ev,T,beta),method='Nelder-Mead',options=dict(maxiter=120,xatol=1e-3,fatol=1e-4))
        if best is None or r.fun<best[0]: best=(r.fun,.95/(1+np.exp(-r.x[1])),beta)
    ca=g.close.values; mm=g.index.hour*60+g.index.minute; ia=np.where(mm>=660)[0]
    if len(ia)<30: continue
    c0=g.close.values[np.where(mm<660)[0][-1]]; er=abs(ca[-1]-c0)/max(g.high.values[ia].max()-g.low.values[ia].min(),1e-9)
    rows.append((d,len(ev),best[1],best[2],er))
R=pd.DataFrame(rows,columns=['d','cnt','n','beta','erA']); R['yr']=R.d.dt.year; R['split']=np.where((R.yr>=2019)&(R.yr<=2021),'train','test'); R=R[R.yr>=2018]
lo,hi=R.erA.quantile([1/3,2/3]); X=R.assign(lab=np.where(R.erA<=lo,1,np.where(R.erA>=hi,0,np.nan))).dropna(subset=['lab'])
def auc(x,y):
    r=rankdata(x); n1=y.sum(); n0=len(y)-n1; return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
print('days',len(R),'median n',R.n.median().round(2),'median events',R.cnt.median())
for nm,sg in (('-n (low branching -> MR)',-R.n),('-count',-R.cnt)):
    pass
for f,sg in (('n',-1),('cnt',-1)):
    print(f'AUC of {"-" if sg<0 else ""}{f} for MR_after vs TREND_after:',' | '.join(f'{sp} {auc(sg*g[f].values,g.lab.values.astype(int)):.3f} (n {len(g)})' for sp,g in X.groupby('split')))
print('corr(n, count)',round(R.n.corr(R.cnt),2),'| mean erA by n tercile (low/mid/high):',R.groupby(pd.qcut(R.n,3,labels=['lo','mid','hi'])).erA.mean().round(3).to_dict())
for sp,g in R.groupby('split'): print(' ',sp,R[R.split==sp].groupby(pd.qcut(R[R.split==sp].n,3,labels=['lo','mid','hi'])).erA.mean().round(3).to_dict())
R.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/hawkes262.pkl')
