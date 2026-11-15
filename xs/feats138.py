"""Shared daily 'structure' features (all lagged one session): critical slowing down, price impact, OU/QHO. d: DataFrame with h,l,c,v indexed by day."""
import numpy as np, pandas as pd
def dfeat(d):
    c=d.c; r=np.log(c).diff(); dv=(c*d.v).replace(0,np.nan)
    ac=r.rolling(20).corr(r.shift(1)); vr=r.rolling(10).var()/r.rolling(60).var()
    x=np.log(c); dx=x.diff(); xl=x.shift(1)
    b=dx.rolling(60).cov(xl)/xl.rolling(60).var(); th=(-b).clip(1e-3,1)                 # OU mean-reversion rate per day
    sd=x.rolling(60).std(); z=(x-x.rolling(60).mean())/sd                                # displacement in stationary-std units (QHO ground-state width)
    f=pd.DataFrame({'AC1':ac,'dAC':ac-ac.shift(10),'LVR':np.log(vr),'dLVR':np.log(vr)-np.log(vr).shift(10),
        'LAMIH':np.log((r.abs()/dv).rolling(20).mean()),
        'LABS':np.log(dv.rolling(5).sum()/((d.h.rolling(5).max()-d.l.rolling(5).min())/c)),
        'LHL':np.log(np.log(2)/th),'OUZ':z})
    return f.shift(1)
def contagion(R,flipdates,k=5,win=5,fit=250):
    """R: daily log-return panel (days x names). flipdates: dict tk-> sorted np.datetime64 array. Returns f(tk,date)->#top-k correlated peers that flipped in prior `win` days."""
    nb={}
    def build(y,m):
        t0=pd.Timestamp(y,m,1); w=R[R.index<t0].iloc[-fit:]; ok=w.columns[w.notna().sum()>=fit*.9]
        if len(ok)<20: return {}
        Z=w[ok].fillna(0); Z=(Z-Z.mean())/(Z.std()+1e-9); C=np.corrcoef(Z.values.T); np.fill_diagonal(C,-9)
        idx=np.argsort(-C,axis=1)[:,:k]; return {ok[i]:[ok[j] for j in idx[i]] for i in range(len(ok))}
    def f(tk,d):
        key=(d.year,d.month)
        if key not in nb: nb[key]=build(*key)
        n=0
        for p in nb[key].get(tk,[]):
            a=flipdates.get(p)
            if a is None: continue
            lo=np.searchsorted(a,np.datetime64(d-pd.Timedelta(days=win))); hi=np.searchsorted(a,np.datetime64(d))
            n+= hi>lo
        return n
    return f
