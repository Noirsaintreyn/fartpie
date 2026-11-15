"""CTA/bank-literature features, all lagged one session: time-series momentum (Moskowitz-Ooi-Pedersen), multi-horizon agreement, trend t-stat,
Haar multi-scale energy (fractal/wavelet), return skew/kurtosis, Daniel-Moskowitz momentum-crash state (market 6m return<0 x market vol)."""
import numpy as np, pandas as pd
MKT=None    # set by caller: daily log-return Series of a market proxy
def dfeat(d):
    c=d.c; x=np.log(c); r=x.diff()
    f=pd.DataFrame(index=c.index)
    f['TS12']=x.shift(21)-x.shift(252); f['TS1']=x-x.shift(21)
    f['MAC']=sum(np.sign(x-x.shift(h)) for h in (21,63,126,252))
    t=np.arange(60); tm=t-t.mean()
    f['TTR']=x.rolling(60).apply(lambda w:(lambda b:b/ (np.std(w-w.mean()-b*tm)/np.sqrt((tm**2).sum())+1e-12))((w*tm).sum()/(tm**2).sum()),raw=True)
    v2=r.rolling(2).sum().rolling(120).var()/2; v16=r.rolling(16).sum().rolling(120).var()/16
    f['ENR']=np.log(v16/v2)                         # >0 = energy at slow scales (persistent), <0 = fast/noisy
    f['SKEW']=r.rolling(60).skew(); f['KURT']=r.rolling(60).kurt()
    if MKT is not None:
        m=MKT.reindex(c.index).fillna(0); m126=m.rolling(126).sum(); mv=m.rolling(20).std(); mvr=mv/mv.rolling(250,min_periods=100).median()
        f['PANIC']=(m126<0).astype(float)*mvr; f['MKT126']=m126; f['MVR']=mvr
    return f.shift(1)
FE=['TS12','TS1','MAC','TTR','ENR','SKEW','KURT','PANIC','MKT126','MVR']
