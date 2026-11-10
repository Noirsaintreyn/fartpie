"""Nov 10 2026: EVENT-DAY BEHAVIOR (8:30 ET releases: CPI, NFP, PPI, retail sales, claims...). DECLARED BEFORE RESULTS.
Event flag known at 09:30: ratio = range of the 08:30 NQ 5m bar / median of that same bar over the prior 60 sessions. E1 = ratio >= 2.5 (the release moved the market); E2 = ratio >= 4.
Also pre-open reaction R = |close 09:25 - close 08:25| / 20d mean range (continuous; terciles).
Units: S6 V2 legs (sz-weighted pts/trade, per leg), S6 day (non-overnight legs), noise engine pts/day. 2018-21 vs 2022-26. PASS: same sign and |t|>=2 in both for event vs other.
Then: sizing test on S6 day: x0.5 or x1.5 on event days; Sharpe both periods."""
import numpy as np, pandas as pd, pickle, warnings; warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad'
X=pd.read_parquet('/tmp/claude-0/ml/nq5_real.parquet'); X['day']=pd.to_datetime(X.day).dt.normalize()
b=X[X.m==510].set_index('day'); rg=(b.high-b.low); med=rg.rolling(60,min_periods=30).median().shift(1); ratio=(rg/med)
pre=X[X.m.isin([505,565])].pivot_table(index='day',columns='m',values='close'); 
D=pd.read_pickle(SP+'/nq_days.pkl'); D.index=pd.to_datetime(D.index).normalize(); r20=(D.rng).rolling(20).mean().shift(1)
R=((pre[565]-pre[505]).abs()/r20).reindex(D.index)
E1=(ratio>=2.5).reindex(D.index).fillna(False); E2=(ratio>=4).reindex(D.index).fillna(False)
print('event days per year (E1 ratio>=2.5):', E1.groupby(E1.index.year).sum().to_dict())
O,N=pickle.load(open(SP+'/s6.pkl','rb')); legs=O['V2']; U={}; allr=[]
for nm,(df,_,_) in zip(N,legs):
    df=df.copy(); df['day']=pd.to_datetime(df.day).dt.normalize(); df['w']=df.pts*df.sz; U[nm]=df[['day','w']]
    if not nm.startswith('ON'): allr.append(df[['day','w']])
U['S6day']=pd.concat(allr).groupby('day').w.sum().reset_index()
G=pd.read_pickle(SP+'/G_noise.pkl'); G['day']=pd.to_datetime(G.day).dt.normalize(); U['NZ']=G.groupby('day').pts.sum().reset_index().rename(columns={'pts':'w'})
def tt(a,b): return (a.mean()-b.mean())/np.sqrt(a.var()/len(a)+b.var()/len(b)) if len(a)>=10 and len(b)>=10 else np.nan
print('\nEVENT (E1) vs OTHER: mean pts per trade (legs) / per day (S6day, NZ); 2018-21 | 22-26')
for un,d in U.items():
    d=d[d.day>='2018-02-01']; o=[]
    for lab,m in (('a',d.day.dt.year<2022),('b',d.day.dt.year>=2022)):
        dd=d[m]; f=E1.reindex(dd.day).fillna(False).values.astype(bool); o.append(f'{lab}: ev {dd.w[f].mean():+6.1f} (n{f.sum()}) oth {dd.w[~f].mean():+6.1f} t {tt(dd.w[f],dd.w[~f]):+.2f}')
    print(f' {un:6}',' | '.join(o))
print('\nPRE-OPEN REACTION terciles (known at 09:30): S6day / NZ pts per day, low-mid-high')
for un in ('S6day','NZ'):
    d=U[un]; d=d[d.day>='2018-02-01']; o=[]
    for lab,m in (('a',d.day.dt.year<2022),('b',d.day.dt.year>=2022)):
        dd=d[m]; x=R.reindex(dd.day).values; q=np.nanquantile(x,[1/3,2/3]); lo,hi=x<=q[0],x>=q[1]; o.append(f'{lab}: lo {dd.w[lo].mean():+5.1f} mid {dd.w[~lo&~hi].mean():+5.1f} hi {dd.w[hi].mean():+5.1f} (hi-lo t {tt(dd.w[hi],dd.w[lo]):+.2f})')
    print(f' {un:6}',' | '.join(o))
print('\nSIZING S6day: Sharpe (daily x sqrt252) base | x0.5 on E1 days | x1.5 on E1 days | skip E1')
d=U['S6day']; d=d[d.day>='2018-02-01'].set_index('day').w; f=E1.reindex(d.index).fillna(False).values.astype(bool)
S=lambda x: x.mean()/x.std()*np.sqrt(252)
for lab,m in (('18-21',d.index.year<2022),('22-26',d.index.year>=2022)):
    x=d[m]; ff=f[m]; print(f' {lab}: {S(x):.2f} | {S(np.where(ff,x*.5,x)):.2f} | {S(np.where(ff,x*1.5,x)):.2f} | {S(np.where(ff,0,x)):.2f}   (event days {ff.sum()}, event share of pts {x[ff].sum()/x.sum():.0%})')

print('\nSKIP ON- ON EVENT DAYS (S6 incl. all legs): total pts, Sharpe, worst month; 18-21 | 22-26')
allt=pd.concat([df.assign(day=pd.to_datetime(df.day).dt.normalize(),w=lambda z:z.pts*z.sz,leg=nm) for nm,(df,_,_) in zip(N,legs)])[['day','leg','w']]
allt=allt[allt.day>='2018-02-01']; allt['ev']=E1.reindex(allt.day).fillna(False).values.astype(bool)
days=pd.DatetimeIndex(sorted(D.index[D.index>='2018-02-01']))
for lab,m in (('18-21',lambda y:y<2022),('22-26',lambda y:y>=2022)):
    t=allt[m(allt.day.dt.year)]; dd=days[m(days.year)]
    base=t.groupby('day').w.sum().reindex(dd).fillna(0); new=t[~((t.leg=='ON-')&t.ev)].groupby('day').w.sum().reindex(dd).fillna(0)
    mm=lambda x: x.resample('ME').sum().min()
    print(f' {lab}: base {base.sum():7.0f} pts Sharpe {S(base):.2f} worst month {mm(base):.0f} | skip ON- on events {new.sum():7.0f} pts Sharpe {S(new):.2f} worst month {mm(new):.0f}  (ON- trades skipped: {((t.leg=="ON-")&t.ev).sum()})')
