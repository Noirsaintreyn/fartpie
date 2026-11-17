"""Report 259: FOMC decision days from the LSE 'US.FOMC Minutes' release dates (minutes come exactly 3 weeks after the meeting: decision day = minutes date - 21 days, 14:00 ET release; emergency meetings not included). Test vs day type, main algo and the report-255 extreme fade, train 2019-21 | test 2018+2022-26. Declared: FOMC days TREND after 14:00 / unpredictable before; the fade should lose."""
import pandas as pd, numpy as np, pickle
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
m=pd.read_parquet('/mnt/user-data/attachments/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/31250181-US.FOMC_Minutes_macro_release_2015-01-07_2026-12-30.parquet')
dec=pd.DatetimeIndex(m.released_at.dt.tz_convert('America/New_York').dt.tz_localize(None).dt.normalize())-pd.Timedelta(days=21)
dec=dec[(dec>='2018-01-01')&(dec<='2026-01-30')]; print('decision days',len(dec),'weekdays',pd.Series(dec.dayofweek).value_counts().to_dict())
dd=pickle.load(open(SP+'ev256.pkl','rb')); T=pickle.load(open(SP+'ext255.pkl','rb')); A,_=pickle.load(open(SP+'amd257_lag0.pkl','rb'))
dd['fomc']=dd.index.isin(dec); print('matched to sessions',int(dd.fomc.sum()))
dd['pre']=dd.index.isin(dec-pd.Timedelta(days=1))|dd.index.isin(dec-pd.Timedelta(days=3))
def tt(x): return x.mean()/(x.std()/np.sqrt(len(x))) if len(x)>2 else np.nan
for sp in ('train','test'):
    g=dd[dd.split==sp]; on=g[g.fomc]; off=g[~g.fomc]
    print(f'{sp}: FOMC n{len(on)} MR {(on.type=="MR").mean():.0%} TREND {(on.type=="TREND").mean():.0%} ER {on.er.mean():.2f} algo {on.alg.mean():+.1f} | other MR {(off.type=="MR").mean():.0%} TREND {(off.type=="TREND").mean():.0%} ER {off.er.mean():.2f} algo {off.alg.mean():+.1f} | rng/atr? ')
T['split']=np.where((T.d.dt.year>=2019)&(T.d.dt.year<=2021),'train','test'); T['f']=T.d.isin(dec)
A['f']=A.d.isin(dec)
for sp in ('train','test'):
    x=T[(T.split==sp)]; print(f'{sp} fade V0 1R FOMC {x[x.f].r1.mean():+.1f} (n{int(x.f.sum())}) vs other {x[~x.f].r1.mean():+.1f}; 2R {x[x.f].r2.mean():+.1f} vs {x[~x.f].r2.mean():+.1f}')
    y=A[(A.lab=='asia')&(A.split==sp)]; print(f'     asia sweep 2R FOMC {y[y.f].r2.mean():+.1f} (n{int(y.f.sum())}) vs {y[~y.f].r2.mean():+.1f}')
print('algo pts on FOMC days by year:',dd[dd.fomc].groupby('yr').alg.agg(['count','mean']).round(1).T.to_dict())
