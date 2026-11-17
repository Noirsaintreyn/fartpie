"""Report 230: fill models for payout droughts, REVISED SPLIT (Rey Nov 17): train 2019-2021, test = 2018 + 2022-2026 (so 2018 is out-of-sample and the test is not just the recent trend). Declared before results.
Adds the mechanism filter Rey asked for: dealer GAMMA REGIME at the prior close (QQQ net GEX, xs/ext/gex/QQQ.csv, daily, 2018+). Hypothesis: negative gamma = dealers sell dips / buy rips = trends extend, so vector flips (M_A, 1:1) should do better there and a VWAP fade worse; positive gamma = dampening, so the fade should do better there.
Cells: 'neg' = prior-close net GEX < 0 (64% of days), 'low3' = bottom third of its trailing-252-day rank, 'pos' = the complement of neg. VWAP model uses sigma bands (std of close - session VWAP, >= 12 bars), target = VWAP, stop = same distance (1:1 to VWAP).
Judge: same sign, |t| >= 2, n >= 30 in both eras; cells are many, so read t ~3 as the bar.
"""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/claude/fartpie/xs'); from vector_pine import run_state_machine; import report_179_fractal as f
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b=b[b.dt>='2017-06-01'].reset_index(drop=True); o,h,l,c,v=[b[k].values for k in ('open','high','low','close','volume')]; n=len(b)
m=(b.dt.dt.hour*60+b.dt.dt.minute).values; day=b.dt.dt.normalize().values; yr=b.dt.dt.year.values; t5=b.dt.values
TR=f.prep(); U={'RA':1.0,'M1':.3,'R1':.5,'ON+':.4,'ON-':.1,'RULES':1.0,'V15a':.3}; core=pd.concat([t_.assign(day=pd.to_datetime(t_.day)).assign(usd=lambda x:x.pts*x.sz*U[k]*20)[['day','usd']] for k,t_ in TR.items()])
S=pd.read_pickle(SP+'s1_trades162.pkl'); S=S[S.reg>0]; s1=pd.DataFrame({'day':pd.to_datetime(S.day).dt.normalize(),'usd':S.pts.values*.2*20}); idx=pd.bdate_range('2018-02-01','2026-01-30'); P=(core.groupby('day').usd.sum().reindex(idx).fillna(0)+s1.groupby('day').usd.sum().reindex(idx).fillna(0)); DRY=(P.rolling(40).sum().shift(1)<=0)
print(f'main algo: drought days (trailing 40-day P&L <= 0) {DRY.mean():.0%} of days; train {DRY[(DRY.index.year>=2019)&(DRY.index.year<=2021)].mean():.0%}, test {DRY[(DRY.index.year==2018)|(DRY.index.year>=2022)].mean():.0%}')
G=pd.read_csv('/home/claude/fartpie/xs/ext/gex/QQQ.csv',parse_dates=['date']).set_index('date').net_gex.sort_index(); G.index=G.index.normalize()
GR=G.rolling(252,min_periods=120).rank(pct=True)
def gx(d):
    k=G.index.searchsorted(pd.Timestamp(d))-1
    return (np.nan,np.nan) if k<0 else (G.iloc[k],GR.iloc[k])
GXD={d:gx(d) for d in pd.unique(day)}
def filt(sigs,mode):
    out=[]
    for x in sigs:
        a,r=GXD[day[x[0]]]
        if not np.isfinite(a): continue
        if mode=='all' or (mode=='neg' and a<0) or (mode=='pos' and a>=0) or (mode=='low3' and r<=1/3): out.append(x)
    return out
def sim(sigs):
    """sigs: list of (5m index of entry bar, side, target pts, stop pts)"""
    out=[]; last=-1
    for e,s,tg,st in sigs:
        if e<=last or e>=n-2 or not (575<=m[e]<=930) or yr[e]<2018: continue
        pe=o[e]; res=None; j=e
        while j<n:
            adv=(pe-l[j]) if s==1 else (h[j]-pe); fav=(h[j]-pe) if s==1 else (pe-l[j])
            if adv>=st: res=-st; break
            if fav>=tg: res=tg; break
            if m[j]>=955 or day[j]!=day[e]: res=s*(c[j]-pe); break
            j+=1
        if res is None: res=s*(c[min(j,n-1)]-pe)
        out.append((day[e],res-1.0,st)); last=j
    return pd.DataFrame(out,columns=['day','pts','R'])
def report(nm,R):
    cells=[]; dd=R.groupby('day').pts.sum(); dd.index=pd.to_datetime(dd.index); D=dd.reindex(idx).fillna(0)
    for era,mk in (('train',(idx.year>=2019)&(idx.year<=2021)),('test',(idx.year==2018)|(idx.year>=2022))):
        _y=pd.to_datetime(R.day).dt.year; x=R[(_y>=2019)&(_y<=2021)] if era=='train' else R[(_y==2018)|(_y>=2022)]; g=x.groupby('day').pts.sum(); t=g.mean()/(g.std(ddof=1)/np.sqrt(len(g))) if len(g)>5 else np.nan; d_=D[mk]; dry=DRY[mk]
        cells.append(f'{era}: n{len(x):5d} win {np.mean(x.pts>0)*100:3.0f}% net {x.pts.mean():+6.2f} pts t{t:+5.1f} R~{x.R.median():3.0f} | per day {d_.mean():+5.2f}, drought days {d_[dry].mean():+5.2f}, other {d_[~dry].mean():+5.2f} | corr with main {np.corrcoef(d_,P[mk])[0,1]:+.2f}')
    print(f'{nm:26s}',' || '.join(cells),flush=True)
for rule,tag in (('5min','5m'),('15min','15m'),('60min','1H')):
    g=b.set_index('dt').resample(rule,label='right',closed='left').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(); res=run_state_machine(g.assign(datetime=g.index)); reg=np.asarray(res[0]); lev=np.asarray(res[1],float); ends=g.index.values; base=[]
    for i in range(1,len(g)):
        s=1 if (reg[i]==1 and reg[i-1]!=1) else -1 if (reg[i]==-1 and reg[i-1]!=-1) else 0
        if s==0 or not np.isfinite(lev[i]): continue
        e=np.searchsorted(t5,ends[i])
        if e>=n or t5[e]!=ends[i]: continue
        R_=abs(o[e]-lev[i])
        if R_<3: continue
        base.append((e,s,R_))
    for k in ((1.0,2.0) if tag!='1H' else (1.0,2.0)):
        for mode in ('all','neg','low3','pos'): report(f'M_A {tag} {k}R gex={mode}',sim(filt([(e,s,k*R_,R_) for e,s,R_ in base],mode)))
# M_B VWAP reversion
pv=np.zeros(n); vv=np.zeros(n); vw=np.full(n,np.nan); sg=np.full(n,np.nan); k0=0
tp=(h+l+c)/3
for i in range(n):
    if i==0 or day[i]!=day[i-1] or m[i]==570: 
        if m[i]>=570: k0=i
    if m[i]<570 or m[i]>955: continue
    if i==k0: pv[i]=tp[i]*v[i]; vv[i]=v[i]
    else: pv[i]=pv[i-1]+tp[i]*v[i]; vv[i]=vv[i-1]+v[i]
    vw[i]=pv[i]/max(vv[i],1)
    if i-k0>=12: sg[i]=np.std(c[k0:i+1]-vw[k0:i+1])
for K in (2.0,2.5):
    sigs=[]
    for i in range(n-1):
        if not np.isfinite(sg[i]) or sg[i]<=0: continue
        z=(c[i]-vw[i])/sg[i]
        if abs(z)>=K:
            dist=abs(o[i+1]-vw[i])
            if dist>=3: sigs.append((i+1,-1 if z>0 else 1,dist,dist))
    for mode in ('all','pos','neg'): report(f'M_B VWAP {K}s gex={mode}',sim(filt(sigs,mode)))
