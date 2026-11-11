"""CVD-EXTREME FADE (follow-up to report_of_screen: session CVD ratio has IC -0.23 (15m) .. -0.36 (60m) vs forward return in BOTH halves). Is it more than price-from-open reversion? One trade/day: first minute 10:30-14:30 where |cvd ratio| >= prior-days' expanding 80th pct, trade AGAINST it, hold H min (or to 15:55), stop S on closes, cost .5. Compare with fading price-from-open (same trigger count). Nov 10 2026."""
import pandas as pd, numpy as np
X=pd.read_parquet('/mnt/user-data/uploads/fartpie/mnq_orderflow_1min.parquet')
X['et']=X.datetime.dt.tz_convert('America/New_York').dt.tz_localize(None); X['day']=X.et.dt.normalize(); X['m']=X.et.dt.hour*60+X.et.dt.minute
X['delta']=X.sell_vol-X.buy_vol; X['vol']=X.buy_vol+X.sell_vol
X=X[(X.m>=570)&(X.m<960)]; B={d:g.set_index('m') for d,g in X.groupby('day') if len(g)>300}
days=sorted(B); feat={}
for d in days:
    g=B[d]; cv=g.delta.cumsum()/g.vol.cumsum().clip(lower=1); rt=g.last_price-g.last_price.iloc[0]; feat[d]=(cv,rt)
def tstat(x): x=np.asarray(x); return x.mean()/(x.std()/np.sqrt(len(x))) if len(x)>8 and x.std()>0 else np.nan
def run(kind,q,H,S):
    out=[]; hist=[]
    for i,d in enumerate(days):
        cv,rt=feat[d]; s=cv if kind=='cvd' else rt/ (rt.abs().max()+1e-9) if False else (cv if kind=='cvd' else rt)
        if len(hist)>=20:
            th=np.quantile(np.abs(np.concatenate(hist)),q)
            w=s[(s.index>=630)&(s.index<=870)]; hit=w[w.abs()>=th]
            if len(hit):
                m0=hit.index[0]; dr=-int(np.sign(hit.iloc[0])); g=B[d]; e=g.last_price[m0]; path=g.last_price[(g.index>m0)&(g.index<=min(m0+H,955))]
                res=None
                for v in path.values:
                    if (v-e)*dr<=-S: res=-S; break
                out.append((d,(res if res is not None else (path.iloc[-1]-e)*dr)-.5))
        hist.append(s[(s.index>=630)&(s.index<=870)].values)
    return pd.DataFrame(out,columns=['day','pts'])
print('trigger | q   hold stop |  n  mean pts  win   t   | half1 (n,mean,t) | half2 (n,mean,t) | total pts')
for kind in ('cvd','price'):
    for q in (.8,.9):
        for H in (30,60,120):
            for S in (40,):
                T=run(kind,q,H,S)
                if len(T)<20: continue
                mid=T.day.iloc[len(T)//2]; a=T[T.day<mid].pts; b=T[T.day>=mid].pts
                print(f'{kind:6} | {q:.1f} {H:4d} {S:4d} | {len(T):3d} {T.pts.mean():+7.1f} {np.mean(T.pts>0):4.0%} {tstat(T.pts):+5.1f} | {len(a):3d} {a.mean():+6.1f} {tstat(a):+5.1f} | {len(b):3d} {b.mean():+6.1f} {tstat(b):+5.1f} | {T.pts.sum():+.0f}')
