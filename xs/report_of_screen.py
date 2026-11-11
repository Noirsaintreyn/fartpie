"""ORDER-FLOW SCREEN (MNQ 1-min, 2024-12..2025-11, SIGN-CORRECTED: delta = sell_vol - buy_vol since the export has buy/sell swapped). Nov 10 2026.
Part 1: per-day rank IC of each feature vs forward return (5/15/30/60 min) at RTH minutes 9:45-15:30 every 5th minute; mean IC, t across days, both halves. PASS = same sign & |t|>=2 in both halves.
Part 2: walk-forward gradient boosting on all features (train first 60% of days, test last 40%), top-decile signal trade sim with 0.5pt cost."""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
X=pd.read_parquet('/mnt/user-data/uploads/fartpie/mnq_orderflow_1min.parquet')
X['et']=X.datetime.dt.tz_convert('America/New_York').dt.tz_localize(None); X['day']=X.et.dt.normalize(); X['m']=X.et.dt.hour*60+X.et.dt.minute
X['delta']=X.sell_vol-X.buy_vol; X['vol']=X.buy_vol+X.sell_vol
X=X[(X.m>=570)&(X.m<960)].sort_values('et').reset_index(drop=True)
X=X[X.groupby('day').day.transform('size')>300].copy()
g=X.groupby('day'); px=X.last_price
def roll(s,w): return s.groupby(X.day).transform(lambda v: v.rolling(w,min_periods=w).sum())
def rollm(s,w): return s.groupby(X.day).transform(lambda v: v.rolling(w,min_periods=w).mean())
F=pd.DataFrame(index=X.index)
for w in (5,15,30): F[f'd{w}']=roll(X.delta,w)/roll(X.vol,w).clip(lower=1)
F['cvd']=X.delta.groupby(X.day).cumsum()/X.vol.groupby(X.day).cumsum().clip(lower=1)
bi=(X.mean_bid_sz-X.mean_ask_sz)/(X.mean_bid_sz+X.mean_ask_sz)
for w in (1,5,15): F[f'bookimb{w}']=rollm(bi,w)
for w in (15,30): F[f'ret{w}']=px-px.groupby(X.day).shift(w)
F['div30']=F.ret30.groupby(X.day).rank(pct=True)-F.d30.groupby(X.day).rank(pct=True)    # price up more than delta supports -> expect fade
F['absorb15']=np.sign(F.d15)*(F.d15.abs().groupby(X.day).rank(pct=True)-F.ret15.abs().groupby(X.day).rank(pct=True))
sz=X.vol/X.n_trades.clip(lower=1); F['blocks5']=np.sign(F.d5)*(rollm(sz,5)/rollm(sz,60).replace(0,np.nan))
F['spread']=rollm(X.mean_spread_ticks,5); F['volspike']=rollm(X.vol,5)/rollm(X.vol,60).replace(0,np.nan)
F['dpx15']=F.d15*np.sign(F.ret15)                # delta agrees with 15-min price move
F['dminute']=X.m
fw={h: px.groupby(X.day).shift(-h)-px for h in (5,15,30,60)}
S=X.m.between(585,930)&((X.m-570)%5==0)
mid=X.day.sort_values().iloc[len(X)//2]
def ic_table(feats,target,absr=False):
    out=[]
    for f in feats:
        rows=[]
        for d,idx in X[S].groupby('day').groups.items():
            a=F.loc[idx,f]; b=target.loc[idx]
            if absr: b=b.abs()
            ok=a.notna()&b.notna()
            if ok.sum()>20 and a[ok].nunique()>3: rows.append((d,spearmanr(a[ok],b[ok])[0]))
        T=pd.DataFrame(rows,columns=['day','ic']).dropna(); 
        t=lambda s: s.mean()/(s.std()/np.sqrt(len(s))) if len(s)>10 else np.nan
        A=T[T.day<mid].ic; B=T[T.day>=mid].ic
        out.append((f,len(T),T.ic.mean(),t(T.ic),A.mean(),t(A),B.mean(),t(B),'PASS' if (np.sign(A.mean())==np.sign(B.mean()) and abs(t(A))>=2 and abs(t(B))>=2) else ''))
    return out
SIG=[c for c in F.columns if c not in ('spread','volspike','dminute')]
print(f'RTH bars {len(X)}, days {X.day.nunique()}')
for h in (5,15,30,60):
    print(f'\n== forward {h} min return: mean daily rank IC (t) | half1 | half2')
    for r in ic_table(SIG,fw[h]): print(f' {r[0]:10} n {r[1]:3d} IC {r[2]:+.3f} (t {r[3]:+5.1f}) | {r[4]:+.3f} (t {r[5]:+5.1f}) | {r[6]:+.3f} (t {r[7]:+5.1f}) {r[8]}')
print('\n== volatility predictors: IC vs |forward 15 min move|')
for r in ic_table(['spread','volspike','d15','absorb15','bookimb5'],fw[15],absr=True): print(f' {r[0]:10} IC {r[2]:+.3f} (t {r[3]:+5.1f}) | {r[4]:+.3f} (t {r[5]:+5.1f}) | {r[6]:+.3f} (t {r[7]:+5.1f}) {r[8]}')
# ---- Part 2 ML
from sklearn.ensemble import HistGradientBoostingRegressor
cols=[c for c in F.columns]; D=F[cols].copy(); D['day']=X.day.values; D['y']=fw[15].values; D=D[S.values].dropna()
days=sorted(D.day.unique()); cut=days[int(len(days)*.6)]; tr=D[D.day<cut-pd.Timedelta(days=1)]; te=D[D.day>=cut]
mdl=HistGradientBoostingRegressor(max_depth=3,learning_rate=.05,max_iter=150,min_samples_leaf=200,l2_regularization=5).fit(tr[cols],tr.y)
te=te.assign(p=mdl.predict(te[cols])); ic=spearmanr(te.p,te.y)[0]
print(f'\n== ML walk-forward: train {len(tr)} rows to {cut.date()}, test {len(te)} rows ({te.day.nunique()} days). Test rank IC {ic:+.3f}')
for q in (.9,.95,.98):
    hi=te[te.p.abs()>=te.p.abs().quantile(q)]; pts=np.sign(hi.p)*hi.y-.5
    print(f'  top {1-q:.0%} |pred| signals: n {len(hi):5d} mean pts/trade (15-min hold, 0.5 cost) {pts.mean():+.2f} win {np.mean(pts>0):.0%} t {pts.mean()/(pts.std()/np.sqrt(len(hi))):+.2f}')
imp=pd.Series(0.0,index=cols)
from sklearn.inspection import permutation_importance
pi=permutation_importance(mdl,te[cols].iloc[::7],te.y.iloc[::7],n_repeats=3,random_state=0); print('  permutation importance:',', '.join(f'{c} {v:+.4f}' for c,v in sorted(zip(cols,pi.importances_mean),key=lambda x:-x[1])[:6]))
