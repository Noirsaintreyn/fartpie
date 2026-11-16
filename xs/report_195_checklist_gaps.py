"""Report 195: checklist gaps on all 3 systems. No entry-delay test (vetoed: late entries sit at the peak).
A) stop/loser slippage sweep per leg (extra pts on every losing trade = upper bound on stop slippage)  B) outlier removal best 1/5/10% vs random 10%
C) long/short split (RA, V15a only have dir)  D) per-year edge and edge as % of NQ price  E) combo worst day/month/DD, outlier days, loser slippage
F) large-cap + mid/small: per-year, outlier days/months, worst day/month  G) multiple-testing bar from number of searches run."""
import sys, json, glob; sys.path.insert(0,'/home/claude/fartpie/xs')
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy import stats
import report_179_fractal as f
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; rng=np.random.default_rng(5)
TR=f.prep(); U={'RA':1.0,'M1':.3,'R1':.5,'ON+':.4,'ON-':.1,'RULES':1.0,'V15a':.3}
def tt(x): x=np.asarray(x); return x.mean()/(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>2 and x.std()>0 else np.nan
def era(t): d=pd.to_datetime(t.day); return d.dt.year<=2021
def line(x): return f'n{len(x):4d} mean {x.mean():6.2f} t{tt(x):5.1f}'
print('=== A) extra slippage on every LOSING trade (pts per contract), train 2018-21 | test 2022+')
for n,t in TR.items():
    e=era(t); s=[]
    for ex in (0,.25,.5,1,2):
        p=t.pts.values-np.where(t.pts.values<0,ex,0); s.append(f'+{ex}: {p[e].mean():5.2f} t{tt(p[e]):4.1f} | {p[~e].mean():5.2f} t{tt(p[~e]):4.1f}')
    print(f'{n:6s}', ' ; '.join(s))
print('=== B) remove best 1/5/10% trades (mean pts) vs remove random 10%, train | test')
for n,t in TR.items():
    e=era(t); r=[]
    for m in (e,~e):
        x=np.sort(t.pts.values[m])[::-1]; k=len(x); row=[f'all {x.mean():5.2f}']
        for q in (.01,.05,.10): row.append(f'-best{int(q*100)}% {x[max(1,int(round(q*k))):].mean():5.2f}')
        rnd=np.mean([np.delete(x,rng.choice(k,max(1,int(.1*k)),replace=False)).mean() for _ in range(300)]); row.append(f'rand10% {rnd:5.2f}'); r.append(' '.join(row))
    print(f'{n:6s}',' || '.join(r))
print('=== C) long/short (dir>0 long), train | test')
for n in ('RA','V15a'):
    t=TR[n]; e=era(t)
    for nm,m in (('long',t.dir.values>0),('short',t.dir.values<0)):
        print(f'{n:5s}{nm:6s} train {line(t.pts[e&m])} | test {line(t.pts[~e&m])}')
print('=== D) per-year mean pts and as % of NQ price (bp of price per trade)')
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b['y']=b.dt.dt.year; px=b.groupby('y').close.mean()
for n,t in TR.items():
    y=pd.to_datetime(t.day).dt.year; g=t.groupby(y).pts.agg(['mean','count'])
    print(f'{n:6s}',' '.join(f'{yy}:{r["mean"]:5.1f}pt/{r["mean"]/px.get(yy,np.nan)*1e4:4.1f}bp(n{int(r["count"])})' for yy,r in g.iterrows()))
print('=== E) combo daily $ (core x U x sz + S1 0.2): worst day/month, maxDD, outlier days, loser slippage')
S=pd.read_pickle(SP+'s1_trades162.pkl'); S=S[S.reg>0]; S['day']=pd.to_datetime(S.day).dt.normalize()
idx=pd.bdate_range('2018-02-01','2026-01-30')
def combo(ex=0):
    parts=[]
    for n,t in TR.items():
        p=t.pts.values-np.where(t.pts.values<0,ex,0); parts.append(pd.DataFrame({'day':pd.to_datetime(t.day),'usd':p*t.sz.values*U[n]*20}))
    p=S.pts.values-np.where(S.pts.values<0,ex*.5,0); parts.append(pd.DataFrame({'day':S.day.values,'usd':p*.2*20}))
    return pd.concat(parts).groupby('day').usd.sum().reindex(idx).fillna(0)
for ex in (0,.5,1,2):
    P=combo(ex); eq=P.cumsum(); dd=(eq-eq.cummax()).min(); m=P.groupby(P.index.to_period('M')).sum()
    print(f'loser slip +{ex}: total ${P.sum():,.0f}, worst day ${P.min():,.0f}, worst month ${m.min():,.0f}, maxDD ${dd:,.0f} (limit-half = $1,000), +months {np.mean(m>0):.0%}, yr test(2022+) ${P[P.index.year>=2022].sum():,.0f}')
P=combo(0); x=np.sort(P[P!=0].values)[::-1]; k=len(x)
for q in (.01,.05,.10): print(f'  drop best {int(q*100)}% of active days: total ${x[int(round(q*k)):].sum():,.0f} of ${x.sum():,.0f}')
print('  per year $:',' '.join(f'{y}:{v:,.0f}' for y,v in P.groupby(P.index.year).sum().items()))
print('=== F) large-cap + mid/small')
L=pd.read_parquet('/mnt/user-data/uploads/fartpie/xs/ext/site_wide_daily.parquet'); L.index=pd.to_datetime(L.index)
r=L.m; ex=r-L.SPY
print(f'large-cap daily: n{len(r)} mean {r.mean()*1e4:.1f}bp t{tt(r):.1f}; excess vs SPY mean {ex.mean()*1e4:.1f}bp t{tt(ex):.1f}; worst day {r.min():.1%}, best {r.max():.1%}')
x=np.sort(r.values)[::-1]; k=len(x)
for q in (.01,.05,.10): print(f'  drop best {int(q*100)}% days: mean {x[int(round(q*k)):].mean()*1e4:.1f}bp (all {x.mean()*1e4:.1f}) ; random10% {np.mean([np.delete(x,rng.choice(k,int(.1*k),replace=False)).mean() for _ in range(300)])*1e4:.1f}bp')
for y,g in r.groupby(r.index.year): print(f'  {y}: ret {(1+g).prod()-1:6.1%} spy {(1+L.SPY[g.index]).prod()-1:6.1%} sharpe {g.mean()/g.std()*np.sqrt(252):4.1f}')
m=(1+r).groupby(r.index.to_period('M')).prod()-1; print(f'  worst month {m.min():.1%}, +months {np.mean(m>0):.0%}')
j=json.load(open('/home/claude/fartpie/live_data/site_series.json'))['mid']; mi=pd.PeriodIndex(j['d'],freq='M'); gb=pd.Series(j['b'],index=mi); ge=pd.Series(j['ew'],index=mi)
rm=gb.pct_change().dropna(); re=ge.pct_change().dropna(); em=rm-re
print(f'mid/small monthly: n{len(rm)} mean {rm.mean():.1%} t{tt(rm):.1f}; excess vs equal-weight mean {em.mean():.1%} t{tt(em):.1f}; worst month {rm.min():.1%}, +months {np.mean(rm>0):.0%}')
x=np.sort(rm.values)[::-1]; k=len(x)
for q in (.05,.10,.20): print(f'  drop best {int(q*100)}% months: mean {x[max(1,int(round(q*k))):].mean():.2%} (all {x.mean():.2%})')
for y,g in rm.groupby(rm.index.year): print(f'  {y}: n{len(g)} ret {(1+g).prod()-1:6.1%}  ew {(1+re[g.index]).prod()-1:6.1%}')
print('=== G) multiple testing')
nrep=len(glob.glob('/home/claude/fartpie/xs/report_*.py')); print(f'report scripts in repo: {nrep}; Bonferroni two-sided 5% |t| bar for N tests:')
for N in (20,100,300,1000): print(f'  N={N}: |z|>={stats.norm.isf(.025/N):.2f}')
print('leg test-era t-stats vs those bars:'); print('  ',' '.join(f'{n}:{tt(t.pts[~era(t)]):.1f}' for n,t in TR.items()))
