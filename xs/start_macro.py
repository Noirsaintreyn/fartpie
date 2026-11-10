"""Nov 10 2026: EVAL START vs FOMC / CPI / NFP CALENDAR and the INFLATION / POLICY regime at start. DECLARED BEFORE RESULTS.
Outcomes = per-start eval results from report_start.py (Flex 50K, variants I and P). Calendar: FOMC decision days (2018-24 from memory, 2025-26 verified on fedratecalc), CPI = release-day proxy (highest 08:30 NQ bar range ratio among days 9-16 each month, ratio >= 2),
NFP = first Friday of the month. Features at start: trading days to next / since last FOMC, CPI, NFP; events in next 20 / 40 trading days; inflation regime (CPI yoy level and 3m-annualized minus yoy, lagged 2 months), policy regime (fed funds 6m change), macro quadrant+HMM (macro_regime.csv, lagged 1 extra month).
Inference caution: starts overlap, so besides halves we report month-block means (one obs per calendar month)."""
import numpy as np, pandas as pd, pickle, requests, io, warnings; warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad'
OUT=pickle.load(open(SP+'/start_out.pkl','rb'))
FOMC=pd.to_datetime(sum([[f'{y}-{m}' for m in ms] for y,ms in {2018:['01-31','03-21','05-02','06-13','08-01','09-26','11-08','12-19'],2019:['01-30','03-20','05-01','06-19','07-31','09-18','10-30','12-11'],2020:['01-29','03-03','03-15','04-29','06-10','07-29','09-16','11-05','12-16'],2021:['01-27','03-17','04-28','06-16','07-28','09-22','11-03','12-15'],2022:['01-26','03-16','05-04','06-15','07-27','09-21','11-02','12-14'],2023:['02-01','03-22','05-03','06-14','07-26','09-20','11-01','12-13'],2024:['01-31','03-20','05-01','06-12','07-31','09-18','11-07','12-18'],2025:['01-29','03-19','05-07','06-18','07-30','09-17','10-29','12-10'],2026:['01-28','03-18','04-29','06-17','07-29','09-16','10-28','12-09']}.items()],[]))
X=pd.read_parquet('/tmp/claude-0/ml/nq5_real.parquet'); X['day']=pd.to_datetime(X.day).dt.normalize(); b=X[X.m==510].set_index('day'); rg=b.high-b.low; ratio=rg/rg.rolling(60,min_periods=30).median().shift(1)
days=pd.DatetimeIndex(sorted(X.day.unique())); days=days[days>='2017-06-01']
cpi=[]
for (y,m),g in ratio[(ratio.index.day>=9)&(ratio.index.day<=16)].groupby([ratio[(ratio.index.day>=9)&(ratio.index.day<=16)].index.year,ratio[(ratio.index.day>=9)&(ratio.index.day<=16)].index.month]):
    if g.max()>=2: cpi.append(g.idxmax())
CPI=pd.DatetimeIndex(cpi); NFP=pd.DatetimeIndex([d for d in days if d.dayofweek==4 and d.day<=7])
print('events found: FOMC',len(FOMC[(FOMC>='2018-01-01')&(FOMC<='2026-01-30')]),'CPI proxy',len(CPI[(CPI>='2018-01-01')]),'NFP',len(NFP[NFP>='2018-01-01']))
pos={d:i for i,d in enumerate(days)}
def tdpos(d): return int(days.searchsorted(d))
def feats(day):
    i=tdpos(day); r={}
    for nm,E in (('fomc',FOMC),('cpi',CPI),('nfp',NFP)):
        ei=np.array([tdpos(e) for e in E]); nx=ei[ei>=i]; pv=ei[ei<i]
        r[f'{nm}_next']=(nx.min()-i) if len(nx) else np.nan; r[f'{nm}_last']=(i-pv.max()) if len(pv) else np.nan
        r[f'{nm}_n20']=int(((ei>=i)&(ei<i+20)).sum()); r[f'{nm}_n40']=int(((ei>=i)&(ei<i+40)).sum())
    r['ev_n20']=r['fomc_n20']+r['cpi_n20']+r['nfp_n20']; r['ev_n40']=r['fomc_n40']+r['cpi_n40']+r['nfp_n40']; return r
def fred(s):
    t=requests.get(f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}',timeout=30).text; d=pd.read_csv(io.StringIO(t)); d.columns=['date','v']; d['date']=pd.to_datetime(d.date); d['v']=pd.to_numeric(d.v,errors='coerce'); return d.set_index('date').v.dropna()
try:
    cp=fred('CPIAUCSL'); ff=fred('FEDFUNDS')
    yoy=cp.pct_change(12)*100; m3=((cp/cp.shift(3))**4-1)*100; infl=pd.DataFrame({'cpi_yoy':yoy,'cpi_acc':m3-yoy}); fchg=(ff-ff.shift(6)).rename('ff_chg6'); fl=ff.rename('ff_lvl')
    REG=pd.concat([infl,fchg,fl],axis=1)
except Exception as e: print('FRED failed',str(e)[:80]); REG=None
MR=pd.read_csv(SP+'/macro_regime.csv',index_col=0,parse_dates=True)
def reg(day,lag_m):
    d=(pd.Timestamp(day)-pd.offsets.MonthBegin(lag_m)).to_period('M').to_timestamp(); 
    return d
def tt(a,b): return (a.mean()-b.mean())/np.sqrt(a.var()/len(a)+b.var()/len(b)) if len(a)>=15 and len(b)>=15 else np.nan
for nm,T in OUT.items():
    T=T[T.index<T.index.max()-120+1].copy(); F=pd.DataFrame([feats(d) for d in T.day],index=T.index); T=T.join(F)
    if REG is not None:
        for c in ('cpi_yoy','cpi_acc'): T[c]=[REG[c].reindex([reg(d,2)]).iloc[0] for d in T.day]
        for c in ('ff_chg6','ff_lvl'): T[c]=[REG[c].reindex([reg(d,1)]).iloc[0] for d in T.day]
    T['quad']=[MR.quad.reindex([reg(d,2)]).iloc[0] for d in T.day]; T['hmm']=[MR.hmm.reindex([reg(d,2)]).iloc[0] for d in T.day]; T['ym']=pd.to_datetime(T.day).dt.to_period('M')
    h1=lambda x: x[x.index<1000].net.mean(); h2=lambda x: x[x.index>=1000].net.mean()
    def line(lab,g): return f'{lab:30} n {len(g):4} | net ${g.net.mean():+6,.0f} | pass {g.passed.mean():4.0%} | paid {g.paid.mean():4.0%} | halves {h1(g):+5,.0f}/{h2(g):+5,.0f} | month-blocks n{g.ym.nunique():3}'
    print(f'\n########## {nm}: {len(T)} starts'); 
    for col,cuts,labs in (('fomc_next',[0,5,10,15,20,100],None),('fomc_last',[-1,4,9,19,29,100],None),('cpi_next',[0,3,7,12,16,100],None),('cpi_last',[-1,3,7,12,16,100],None),('nfp_next',[0,3,7,12,16,100],None)):
        g=T.dropna(subset=[col]); print(f'-- {col} (trading days) buckets')
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            x=g[(g[col]>lo)&(g[col]<=hi)] if lo>=0 and col.endswith('next') and lo==0 else g[(g[col]>lo)&(g[col]<=hi)]
            if len(x)>=25: print('   '+line(f'{col} {lo+1 if lo>=0 else 0}-{hi}',x))
    for col in ('ev_n20','ev_n40'):
        print(f'-- {col}: events (FOMC+CPI+NFP) in next window'); [print('   '+line(f'{col} = {v}',T[T[col]==v])) for v in sorted(T[col].unique()) if (T[col]==v).sum()>=25]
    for col in ('cpi_yoy','cpi_acc','ff_chg6','ff_lvl'):
        if col not in T or T[col].isna().all(): continue
        g=T.dropna(subset=[col]); q=np.nanquantile(g[col],[1/3,2/3]); lo,mid,hi=g[g[col]<=q[0]],g[(g[col]>q[0])&(g[col]<q[1])],g[g[col]>=q[1]]
        print(f'-- {col} terciles (cuts {q[0]:+.2f}/{q[1]:+.2f})'); print('   '+line('LOW',lo)); print('   '+line('MID',mid)); print('   '+line('HIGH',hi)+f' | hi-lo t {tt(hi.net,lo.net):+.2f}')
    for col in ('quad','hmm'):
        print(f'-- {col}'); [print('   '+line(f'{col} {v}',T[T[col]==v])) for v in T[col].dropna().unique() if (T[col]==v).sum()>=25]
    # month-block view: one obs per calendar month for the strongest calendar feature
    M=T.groupby('ym').agg(net=('net','mean'),fn=('fomc_next','mean'),cn=('cpi_next','mean')); print(f'-- month-block corr(net, fomc_next) {np.corrcoef(M.net,M.fn)[0,1]:+.2f} | corr(net, cpi_next) {np.corrcoef(M.net,M.cn)[0,1]:+.2f} (n={len(M)} months)')
