"""Rey: on the charts the winners were already BLUE (bull vector) BEFORE the move. Test: do volume-explosion alerts that align with the vector beat the ones that don't? Declared before results.
Same alert sim as report_14b (VM>=20 x median daily vol cumulative incl. premarket, gain>=15% vs prior close, entry next 5m open +1%, exit 15:55 close, 0.5% cost, optional -10% stop).
Vector computed on 5m bars (all hours, incl. prior sessions) and on 15m resample. Regime read at the alert bar (5m) / last COMPLETED 15m bar (causal).
Groups: reg5 bull/bear/none at alert, reg15 bull/bear, both bull, 'bull BEFORE the move' = 5m regime bull at the last bar before price first reached +5% over prior close that day, bull-since >=1 session.
~60 days, no train/test split (sample too small) -> sanity check only."""
import sys, pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'xs'); sys.path.insert(0,'.')
from vector_pine import run_state_machine
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; D5=pickle.load(open(SC+'lowpx_5m.pkl','rb')); DD=pickle.load(open(SC+'lowpx_daily.pkl','rb'))
def prep(X):
    X=X.copy(); X.index=X.index.tz_convert('America/New_York').tz_localize(None); X=X.rename(columns=str.lower)[['open','high','low','close','volume']].astype(float); return X[~X.index.duplicated()]
R=[]
for tk,X0 in D5.items():
    X=prep(X0); d=DD[tk]; d=d[~d.index.duplicated()]; med=d.Volume.shift(1).rolling(20).median(); dc=d.Close
    if len(X)<400: continue
    try:
        r5=pd.Series(run_state_machine(X.assign(datetime=X.index))[0],index=X.index)
        X15=X.resample('15min').agg(dict(open='first',high='max',low='min',close='last',volume='sum')).dropna(subset=['close']); r15=pd.Series(run_state_machine(X15.assign(datetime=X15.index))[0],index=X15.index)
    except Exception as e: continue
    days=X.index.normalize()
    for day in pd.unique(days):
        dk=pd.Timestamp(day)
        if dk not in med.index or not np.isfinite(med[dk]) or med[dk]<=0: continue
        j=dc.index.get_loc(dk)
        if j<1: continue
        pcl=dc.iloc[j-1]
        if not(.3<=pcl<=20): continue
        m=(days==day); g=X[m]; c=g.close.values; o=g.open.values; lo=g.low.values; hi=g.high.values; tm=g.index.hour*60+g.index.minute; cv=g.volume.cumsum().values
        reg=np.where((tm>=570)&(tm<=955))[0]
        if len(reg)<20: continue
        last=reg[-1]; ok=np.where((cv>=20*med[dk])&(c>=pcl*1.15))[0]
        if len(ok)==0: continue
        a=ok[0]
        if tm[a]>=930 or a+1>last: continue
        t=g.index[a]; en=o[a+1]*1.01; rc=c[last]/en-1-.005; stop=en*.9; hit=np.where(lo[a+1:last+1]<=stop)[0]; rs=(stop*.99/en-1-.005) if len(hit) else rc
        s5=int(r5.loc[t]); q=r15[r15.index+pd.Timedelta(minutes=15)<=t]; s15=int(q.iloc[-1]) if len(q) else 0
        pre=np.where(c>=pcl*1.05)[0]; pi=pre[0] if len(pre) else a; bt=g.index[max(pi-1,0)]; sb=int(r5.loc[bt]); qb=r15[r15.index+pd.Timedelta(minutes=15)<=bt]; sb15=int(qb.iloc[-1]) if len(qb) else 0
        # bull-since: bars bull before pre-move bar (5m, last 200 bars)
        w=r5[:bt].iloc[-200:]; since=int((w[::-1].cumprod()==1).sum()) if False else int((w[::-1]==1).cumsum().where(w[::-1]==1).max() if False else 0)
        R.append(dict(tk=tk,day=dk,t=t,gain_at=c[a]/pcl-1,rc=rc,rs=rs,mfe=hi[a+1:last+1].max()/en-1,mae=lo[a+1:last+1].min()/en-1,runner=float(hi[reg].max()/pcl>=1.5),s5=s5,s15=s15,sb=sb,sb15=sb15))
E=pd.DataFrame(R); E.to_pickle(SC+'alert_align.pkl'); print('alerts',len(E))
def st(g):
    if len(g)<8: return 'n%d'%len(g)
    a=g.rc.sort_values(); return 'n%-4d P(+50%% day) %2.0f%% | to-close mean %+6.2f%% med %+6.2f%% win %2.0f%% ex-top1 %+6.2f | w/-10%%stop %+6.2f%% | MFE med %+5.1f%% MAE med %+5.1f%%'%(len(g),g.runner.mean()*100,g.rc.mean()*100,g.rc.median()*100,(g.rc>0).mean()*100,a.iloc[:max(int(len(a)*.99),1)].mean()*100,g.rs.mean()*100,g.mfe.median()*100,g.mae.median()*100)
print('ALL alerts                           ',st(E))
for nm,f in (('5m bull AT alert',E.s5==1),('5m bear AT alert',E.s5==-1),('5m none AT alert',E.s5==0),('15m bull AT alert',E.s15==1),('15m bear/none AT alert',E.s15!=1),('5m+15m both bull AT alert',(E.s5==1)&(E.s15==1)),('5m bull BEFORE the move',E.sb==1),('5m NOT bull before the move',E.sb!=1),('15m bull BEFORE the move',E.sb15==1),('both bull BEFORE the move',(E.sb==1)&(E.sb15==1)),('both bull BEFORE & at alert',(E.sb==1)&(E.sb15==1)&(E.s5==1)&(E.s15==1)),('bull before, flipped bear by alert',(E.sb==1)&(E.s5==-1))):
    print('%-37s'%nm,st(E[f]))
print('\n-- same, only alerts with gain-at-alert 15-60% (early enough to still trade)'); F=E[E.gain_at<.6]
for nm,f in (('ALL',F.s5>-9),('5m bull BEFORE',F.sb==1),('5m not bull BEFORE',F.sb!=1),('both bull BEFORE',(F.sb==1)&(F.sb15==1)),('both bull BEFORE & at alert',(F.sb==1)&(F.sb15==1)&(F.s5==1)&(F.s15==1))): print('%-37s'%nm,st(F[f]))
print('\n-- runner days (day high >=+50%%) : share with 5m bull BEFORE move: %.0f%% | non-runners: %.0f%%'%(100*(E[E.runner==1].sb==1).mean(),100*(E[E.runner==0].sb==1).mean()))
