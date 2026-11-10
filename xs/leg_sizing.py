"""Nov 10 2026: PORTFOLIO-LEVEL LEG SIZING. DECLARED BEFORE RESULTS. Daily sz-weighted pts per leg (S6 V2, ON- skipped on event days = current candidate). Weights re-set each month from trailing 252 days only:
  EQ equal (current) | IV inverse-vol | ERC equal-risk-contribution (long-only, covariance) | SH Sharpe-tilt (w ~ max(trailing mean/std,0)+0.25) ; each normalised so the average weight is 1 (same total size).
Compare 2018-21 vs 2022-26: Sharpe, pts/day, worst month, max drawdown in pts. PASS = Sharpe up in both periods without a worse max drawdown."""
import numpy as np, pandas as pd, pickle, warnings; warnings.filterwarnings('ignore')
from scipy.optimize import minimize
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad'
exec(open('event_days.py').read().split("print('event days per year")[0])
O,N=pickle.load(open(SP+'/s6.pkl','rb')); legs=O['V2']
rows=[]
for nm,(df,_,_) in zip(N,legs):
    df=df.copy(); df['day']=pd.to_datetime(df.day).dt.normalize(); df['w']=df.pts*df.sz; df['leg']=nm; rows.append(df[['day','leg','w']])
T=pd.concat(rows); T=T[T.day>='2018-02-01']; T['ev']=E1.reindex(T.day).fillna(False).values.astype(bool); T=T[~((T.leg=='ON-')&T.ev)]
days=pd.DatetimeIndex(sorted(D.index[D.index>='2017-01-01'])); L=T.pivot_table(index='day',columns='leg',values='w',aggfunc='sum').reindex(days[days>='2018-02-01']).fillna(0)
def erc(C):
    n=len(C); f=lambda w: ((w*(C@w)-(w@C@w)/n)**2).sum(); r=minimize(f,np.ones(n)/n,bounds=[(.02,1)]*n,constraints={'type':'eq','fun':lambda w:w.sum()-1}); return r.x
W={k:pd.DataFrame(1.0,index=L.index,columns=L.columns) for k in ('EQ','IV','ERC','SH')}
months=sorted(set(L.index.to_period('M')))
for p in months:
    idx=L.index[L.index.to_period('M')==p]; hist=L[L.index<idx[0]].iloc[-252:]
    if len(hist)<120: continue
    sd=hist.std().replace(0,np.nan); iv=(1/sd); W['IV'].loc[idx]=(iv/iv.mean()).fillna(1).values
    C=hist.cov().values+1e-9*np.eye(L.shape[1]); w=erc(C); W['ERC'].loc[idx]=w*len(w)
    sh=(hist.mean()/sd).clip(lower=0)+.25; W['SH'].loc[idx]=(sh/sh.mean()).fillna(1).values
S=lambda x: x.mean()/x.std()*np.sqrt(252)
def dd(x): e=x.cumsum(); return (e-e.cummax()).min()
print('weights last month (IV):',W['IV'].iloc[-1].round(2).to_dict()); print('weights last month (ERC):',W['ERC'].iloc[-1].round(2).to_dict())
for lab,m in (('18-21',L.index.year<2022),('22-26',L.index.year>=2022)):
    for k in W:
        x=(L*W[k]).sum(axis=1)[m & (L.index>='2019-02-01')]
        print(f' {lab} {k:3}: pts/day {x.mean():+6.2f} Sharpe {S(x):.2f} worst month {x.resample("ME").sum().min():6.0f} maxDD {dd(x):7.0f}')
