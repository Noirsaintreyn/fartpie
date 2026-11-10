import numpy as np, pandas as pd
src=open('xs/event_days.py').read(); exec(src[:src.index("def tt(a,b)")])
# all S6 legs (incl ON) per day
allp=pd.concat([df[['day','w']] for df in [U[n] for n in N]]).groupby('day').w.sum()
allp=allp[allp.index>='2018-02-01']
f=E1.reindex(allp.index).fillna(False).values.astype(bool)
for lab,m in (('2018-21',allp.index.year<2022),('2022-26',allp.index.year>=2022)):
    x=allp[m]; ff=f[m]; y=np.where(ff,0,x); yrs=len(x)/252
    mo=lambda z: pd.Series(z,index=x.index).groupby(x.index.to_period('M')).sum()
    print(f'{lab}: trade-all {x.sum():+8.0f} pts ({x.sum()/yrs:+6.0f}/yr, Sharpe {x.mean()/x.std()*252**.5:.2f}, worst mo {mo(x.values).min():+.0f}) | skip event days {y.sum():+8.0f} ({y.sum()/yrs:+6.0f}/yr, Sharpe {y.mean()/y.std()*252**.5:.2f}, worst mo {mo(y).min():+.0f}) | event days {ff.sum()} = {x[ff].sum():+.0f} pts ({x[ff].sum()/x.sum():.0%} of total)')
