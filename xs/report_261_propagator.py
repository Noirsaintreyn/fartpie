"""Report 261: TRANSIENT-IMPACT (propagator) test of absorption. DECLARED before results.
Checkpoints 10:30,11:30,12:30,13:30,14:30 ET (one obs per day per checkpoint, no overlap). P = price change over the last 6 5m bars / ATR14(daily RTH range). Q = sum of est. delta over the same 6 bars / (6 x mean bar volume of the prior 60 bars) (flow in volume units, est delta = vol x (2c-h-l)/(h-l)). beta fitted causally by OLS P~Q on the trailing 120 days of observations at the same checkpoint (only past data). Impact-driven part I = beta*Q; unexplained part U = P - I. Response y = next-60-min price change / ATR (signed, 12 bars).
Transient-impact prediction: coefficient on I is NEGATIVE (impact decays: reverts) and more negative than the coefficient on U (information-like, permanent). Test y ~ I + U pooled, by split (train 2019-21 | test 2018+2022-26), t-stats clustered by day. Tradable check: fade (sign opposite to I) when |I| is in the top decile of its trailing 120-day distribution, hold 12 bars, net of 1pt, pts/trade by split.
Alternative absorption reading (flow aligned with a push but price stalls): bottom-quartile |U| with large |Q|; reported too."""
import numpy as np, pandas as pd
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b['dt']=pd.to_datetime(b.dt); b=b.set_index('dt').sort_index().between_time('09:30','15:55'); b['d']=b.index.normalize()
D=b.groupby('d'); dd=pd.DataFrame({'h':D.high.max(),'l':D.low.min(),'n':D.size()}); dd=dd[dd.n>=70]; dd['atr']=(dd.h-dd.l).rolling(14).mean().shift(1)
rows=[]
for d,g in D:
    if d not in dd.index or not np.isfinite(dd.atr[d]): continue
    a=dd.atr[d]; o=g.open.values;h=g.high.values;l=g.low.values;c=g.close.values;v=g.volume.values.astype(float); mm=g.index.hour*60+g.index.minute
    dl=v*(2*c-h-l)/np.maximum(h-l,1e-9)
    for ck in (630,690,750,810,870):
        i=np.where(mm==ck)[0]
        if len(i)==0: continue
        i=i[0]
        if i<12 or i+12>=len(g): continue
        P=(c[i-1]-o[i-6])/a; pv=v[max(0,i-60):i].mean(); Q=dl[i-6:i].sum()/(6*pv); y=(c[i+11]-o[i])/a   # decision at open of bar i (checkpoint), hold 12 bars
        rows.append((d,ck,P,Q,y,o[i],c[i+11]))
R=pd.DataFrame(rows,columns=['d','ck','P','Q','y','en','ex']); R['yr']=R.d.dt.year; R['split']=np.where((R.yr>=2019)&(R.yr<=2021),'train','test'); R=R[R.yr>=2018].sort_values(['ck','d']).reset_index(drop=True)
# causal beta per checkpoint
Ii=np.full(len(R),np.nan)
for ck,g in R.groupby('ck'):
    idx=g.index.values
    for k in range(120,len(idx)):
        w=R.loc[idx[k-120:k]]; q=w.Q.values; p=w.P.values; beta=(q*p).sum()/(q*q).sum() if (q*q).sum()>0 else 0
        Ii[idx[k]]=beta*R.Q[idx[k]]
R['I']=Ii; R['U']=R.P-R.I; R=R.dropna(subset=['I'])
def ols_cluster(X,y,cl):
    X=np.c_[np.ones(len(X)),X]; XtXi=np.linalg.inv(X.T@X); b=XtXi@X.T@y; e=y-X@b
    meat=np.zeros((X.shape[1],)*2)
    for k in np.unique(cl):
        m=cl==k; s=X[m].T@e[m]; meat+=np.outer(s,s)
    V=XtXi@meat@XtXi; return b,b/np.sqrt(np.diag(V))
print('y ~ I + U  (coef, cluster t) ; prediction: coef(I) negative and < coef(U)')
for sp in ('train','test'):
    g=R[R.split==sp]; b_,t_=ols_cluster(g[['I','U']].values,g.y.values,g.d.values.astype('int64'))
    print(f'  {sp}: n {len(g)}  I {b_[1]:+.3f} (t {t_[1]:+.1f})   U {b_[2]:+.3f} (t {t_[2]:+.1f})')
    for ck in (630,690,750,810,870):
        h_=g[g.ck==ck]; bb,tt_=ols_cluster(h_[['I','U']].values,h_.y.values,h_.d.values.astype('int64')); print(f'      {ck//60}:{ck%60:02d} I {bb[1]:+.3f}(t{tt_[1]:+.1f}) U {bb[2]:+.3f}(t{tt_[2]:+.1f}) n{len(h_)}')
# tradable: fade sign(-I) when |I| top decile of trailing distribution
R['aI']=R.I.abs(); R['thr']=R.groupby('ck').aI.transform(lambda s:s.shift(1).rolling(120,min_periods=60).quantile(.9))
T=R[R.aI>=R.thr].copy(); T['pts']=-np.sign(T.I)*(T.ex-T.en)-1.0
print('fade impact-driven displacement (|I| top decile), hold 60 min, net pts/trade:')
for sp in ('train','test'):
    x=T[T.split==sp]; print(f'  {sp}: n {len(x)} ({len(x)/(3 if sp=="train" else 5.1):.0f}/yr) {x.pts.mean():+.2f} (t {x.pts.mean()/(x.pts.std()/np.sqrt(len(x))):+.1f}) win {(x.pts>0).mean():.0%}')
# alt: large |Q|, small |U| (flow with no unexplained move = absorbed)
R['aQ']=R.Q.abs(); qh=R.groupby('ck').aQ.transform(lambda s:s.shift(1).rolling(120,min_periods=60).quantile(.75)); ul=R.groupby('ck').U.transform(lambda s:s.abs().shift(1).rolling(120,min_periods=60).quantile(.25))
A=R[(R.aQ>=qh)&(R.U.abs()<=ul)].copy(); A['pts_with']=np.sign(A.Q)*(A.ex-A.en)-1.0
for sp in ('train','test'):
    x=A[A.split==sp]; print(f'  absorbed flow (big |Q|, tiny |U|) trade WITH flow: {sp} n {len(x)} {x.pts_with.mean():+.2f} (t {x.pts_with.mean()/(x.pts_with.std()/np.sqrt(len(x))):+.1f}); against flow {(-x.pts_with-2).mean():+.2f}')
