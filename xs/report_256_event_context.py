"""Report 256: EVENT / NEWS CONTEXT vs day type, the main algo and the extreme fade. DECLARED before results (Rey Nov 17: news, calendar, big-tech earnings).
No external news feed exists here, so flags are derived from data already on disk: (1) RELEASE = 08:30 ET NQ 5m bar range >= 2.5x its 60-session median (CPI/NFP/PPI/claims-type releases that moved the market, known at 09:30); (2) NFP day = first Friday of month; OPEX = third Friday; MONTHEND = last session of month; (3) BIGTECH_REACT = at least one of AAPL MSFT GOOGL AMZN META NVDA TSLA AVGO reported after the close the previous session (or before the open today); BIGTECH_NIGHT = at least one reports after the close today; NREP = number of those reporting in the day-1..day window.
Hypotheses fixed: events/reaction days TREND (lower MR share); opex and month-end MR (pinning/rebalance); earnings night (positioning into the print) MR.
Outputs by flag, train 2019-21 | test 2018+2022-26: share of days that are MR (bottom third of RTH efficiency), main algo NQ pts/day (deploy size, shareable250 daily), and the report-255 extreme fade (V0 and V1 volume-building) pts/trade at 1R. A flag counts if the MR share / P&L differs in the same direction in both splits."""
import numpy as np, pandas as pd, pickle, glob, os
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b['dt']=pd.to_datetime(b.dt); b=b.set_index('dt').sort_index()
rth=b.between_time('09:30','15:55'); D=rth.groupby(rth.index.normalize())
dd=pd.DataFrame({'o':D.open.first(),'h':D.high.max(),'l':D.low.min(),'c':D.close.last(),'n':D.size()}); dd=dd[dd.n>=70]
dd['er']=(dd.c-dd.o).abs()/(dd.h-dd.l); dd['type']=pd.qcut(dd.er,3,labels=['MR','MID','TREND'])
r830=b.between_time('08:30','08:30'); r830.index=r830.index.normalize(); rg=(r830.high-r830.low); rg=rg[~rg.index.duplicated()]
dd['rel']=(rg/rg.rolling(60,min_periods=30).median().shift(1)).reindex(dd.index)>=2.5
idx=dd.index; dow=idx.dayofweek; dom=idx.day
dd['nfp']=(dow==4)&(dom<=7); dd['opex']=(dow==4)&(dom>=15)&(dom<=21); dd['mend']=pd.Series(idx.to_period('M'),index=idx).ne(pd.Series(idx.to_period('M'),index=idx).shift(-1)).values
names=['AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','AVGO']; ev=[]
for n in names:
    e=pd.read_parquet(f'/home/claude/fartpie/data/fund/earn/{n}.parquet'); w=pd.to_datetime(e.when)
    for x in w: ev.append((n,x))
E=pd.DataFrame(ev,columns=['n','when']); E['hr']=E.when.dt.hour; E['dte']=E.when.dt.normalize()
print('earnings rows',len(E),'range',E.dte.min().date(),E.dte.max().date(),'hours',E.hr.value_counts().head(4).to_dict())
tds=list(idx); pos={d:i for i,d in enumerate(tds)}
amc=E[E.hr>=15]; bmo=E[E.hr<13]
def nxt(d): 
    i=np.searchsorted(idx.values,d.to_datetime64()); return idx[i] if i<len(idx) else None
react=pd.Series(0,index=idx); night=pd.Series(0,index=idx)
for _,r in amc.iterrows():
    d=r.dte
    if d in pos: night[d]+=1; 
    k=nxt(d+pd.Timedelta(days=1))
    if k is not None and (k-d).days<=5: react[k]+=1
for _,r in bmo.iterrows():
    d=r.dte
    if d in pos: react[d]+=1
dd['react']=react>0; dd['night']=night>0; dd['nreact']=react
dd['yr']=dd.index.year; dd['split']=np.where((dd.yr>=2019)&(dd.yr<=2021),'train','test'); dd=dd[dd.yr>=2018]
Z=pickle.load(open(SP+'shareable250.pkl','rb')); dd['alg']=(Z['daily']/20).reindex(dd.index)
T=pickle.load(open(SP+'ext255.pkl','rb')); T['split']=np.where((T.d.dt.year>=2019)&(T.d.dt.year<=2021),'train','test')
fl=['rel','nfp','opex','mend','react','night']
print('flag days: '+' '.join(f'{f} {int(dd[f].sum())}' for f in fl))
print('flag | MR share ON (train|test) vs OFF | algo pts/day ON vs OFF | fade V0 1R ON vs OFF | V1 1R ON')
for f in fl:
    s=[]
    for sp in ('train','test'):
        g=dd[dd.split==sp]; on=g[g[f]]; off=g[~g[f]]
        s.append((f'{(on.type=="MR").mean():.0%}/{(off.type=="MR").mean():.0%}', f'{on.alg.mean():+.1f}/{off.alg.mean():+.1f}', len(on)))
    fo=[]
    for sp in ('train','test'):
        t=T[T.split==sp]; m=t.d.map(dd[f]).fillna(False).astype(bool); v1=t.vb
        fo.append(f'{t[m].r1.mean():+.1f}/{t[~m].r1.mean():+.1f} V1on {t[m&v1].r1.mean():+.1f}(n{(m&v1).sum()})')
    print(f' {f:5s} MR {s[0][0]} | {s[1][0]}   algo {s[0][1]} | {s[1][1]}   n {s[0][2]}|{s[1][2]}   fade {fo[0]} | {fo[1]}')
print('react by count: MR share / algo pts/day'); 
for k in (0,1,2,3):
    g=dd[dd.nreact==k] if k<3 else dd[dd.nreact>=3]
    print(f'  nreact={k}{"+" if k==3 else ""}: n {len(g)} MR {(g.type=="MR").mean():.0%} TREND {(g.type=="TREND").mean():.0%} algo {g.alg.mean():+.1f}')
pickle.dump(dd,open(SP+'ev256.pkl','wb'))
