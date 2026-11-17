"""Report 229: a separate small model to fill the payout droughts. DECLARED before results.
M_A (Rey): 1:1 vector. At a vector flip the stop is the vector's new level; the target is the same distance R the other way (also 0.5R, 1.5R, 2R for shape). Flips on 5m, 15m and 1H NQ bars; entry at the open of the 5m bar after the flip bar closes; fills checked on 5m bars, stop wins a bar that touches both; RTH entries 09:35-15:25, flat 15:55; one position at a time per model; cost 1.0 pt.
M_B (complement, mechanism: the trend legs starve in chop, a fade should eat there): VWAP reversion. When a 5m close is >= K sigma from the session VWAP (sigma = running std of close - VWAP since 09:30, at least 12 bars), fade toward VWAP: target = VWAP at entry, stop = the same distance beyond, flat 15:55. K = 2.0 and 2.5.
Judged on (1) the strict split: net pts per trade, day-clustered t, train 2018-21 | test 2022-26 (PASS = same sign, |t| >= 2, n >= 30; 14 cells -> Bonferroni 3.0); (2) FILL value: daily correlation with the main algo's daily P&L (core legs + premium fade, report 222 series) and its mean daily points on DROUGHT days (main algo's trailing 40-day P&L <= 0) vs other days. A fill model must be positive on drought days in both eras."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/claude/fartpie/xs'); from vector_pine import run_state_machine; import report_179_fractal as f
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b=b[b.dt>='2017-06-01'].reset_index(drop=True); o,h,l,c,v=[b[k].values for k in ('open','high','low','close','volume')]; n=len(b)
m=(b.dt.dt.hour*60+b.dt.dt.minute).values; day=b.dt.dt.normalize().values; yr=b.dt.dt.year.values; t5=b.dt.values
TR=f.prep(); U={'RA':1.0,'M1':.3,'R1':.5,'ON+':.4,'ON-':.1,'RULES':1.0,'V15a':.3}; core=pd.concat([t_.assign(day=pd.to_datetime(t_.day)).assign(usd=lambda x:x.pts*x.sz*U[k]*20)[['day','usd']] for k,t_ in TR.items()])
S=pd.read_pickle(SP+'s1_trades162.pkl'); S=S[S.reg>0]; s1=pd.DataFrame({'day':pd.to_datetime(S.day).dt.normalize(),'usd':S.pts.values*.2*20}); idx=pd.bdate_range('2018-02-01','2026-01-30'); P=(core.groupby('day').usd.sum().reindex(idx).fillna(0)+s1.groupby('day').usd.sum().reindex(idx).fillna(0)); DRY=(P.rolling(40).sum().shift(1)<=0)
print(f'main algo: drought days (trailing 40-day P&L <= 0) {DRY.mean():.0%} of days; train {DRY[DRY.index.year<2022].mean():.0%}, test {DRY[DRY.index.year>=2022].mean():.0%}')
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
    for era,mk in (('train',idx.year<2022),('test',idx.year>=2022)):
        x=R[(pd.to_datetime(R.day).dt.year<2022) if era=='train' else (pd.to_datetime(R.day).dt.year>=2022)]; g=x.groupby('day').pts.sum(); t=g.mean()/(g.std(ddof=1)/np.sqrt(len(g))) if len(g)>5 else np.nan; d_=D[mk]; dry=DRY[mk]
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
    for k in (.5,1.0,1.5,2.0): report(f'M_A {tag} flip, target {k}R',sim([(e,s,k*R_,R_) for e,s,R_ in base]))
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
    report(f'M_B VWAP fade {K} sigma',sim(sigs))
