"""IDEA 5 sizing: x1.25 on high-balance days, x0.75 on low (trailing-252 terciles of ov5, no look-ahead), applied to S6 all-legs/day (ON legs excluded from the multiplier: their P&L is the gap, set x1), and to the noise engine. Sharpe + pts, both periods."""
import numpy as np, pandas as pd
src=open('idea5_balance.py').read(); exec(src[:src.index("def rep(")])
allr2=[]
for nm,(df,_,_) in zip(N,O['V2']):
    df=df.copy(); df['day']=pd.to_datetime(df.day).dt.normalize(); df['w']=df.pts*df.sz; df['on']=nm.startswith('ON'); allr2.append(df[['day','w','on']])
Z2=pd.concat(allr2); day=Z2[~Z2.on].groupby('day').w.sum(); onl=Z2[Z2.on].groupby('day').w.sum()
tern=pd.DataFrame({'ov':ov5}); lo=ov5.rolling(252,min_periods=120).quantile(1/3).shift(1); hi=ov5.rolling(252,min_periods=120).quantile(2/3).shift(1)
def mult(idx,up,dn): 
    m=pd.Series(1.0,index=idx); o=ov5.reindex(idx); m[o>=hi.reindex(idx)]=up; m[o<=lo.reindex(idx)]=dn; return m
S=lambda x: x.mean()/x.std()*np.sqrt(252)
for name,ser,extra in (('S6 all legs',day,onl),('NZ engine',NZd.set_index('day').w,None)):
    idx=ser.index[ser.index>='2018-02-01']
    for up,dn in ((1.25,0.75),(1.5,0.5),(1.0,0.5)):
        m=mult(idx,up,dn); o=[]
        for lab,mm in (('18-21',idx.year<2022),('22-26',idx.year>=2022)):
            b=ser.reindex(idx)[mm]; s=b*m[mm]
            if extra is not None: e=extra.reindex(idx).fillna(0)[mm]; b=b+e; s=s+e
            o.append(f'{lab}: Sharpe {S(b):.2f} -> {S(s):.2f}, pts {b.sum():+.0f} -> {s.sum():+.0f}')
        print(f' {name:11} x{up}/x{dn}: ',' | '.join(o))
