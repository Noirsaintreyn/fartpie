"""Maximin pivot LB = highest(lowest(low,L),L) as a screener filter. TFs: 15m, 1h (from 15m), daily. Feature = close_prev/LB-1 at entry (completed bars only). Also reclaim cross (close back above LB within last 4 bars)."""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from collections import deque
N=6; BRK=.08
T=pd.read_parquet('xs/ext/wide_T.parquet'); T=T[T.sp&(T.ratio>=1.5)].copy()
def lb(low,L): return low.rolling(L).min().rolling(L).max()
F=[]
for tk in sorted(T.tk.unique()):
    f=glob.glob(f'lse_clean/{tk}_15m_2018-01-01_*.parquet')
    if not f: continue
    H=pd.read_parquet(f[0],columns=['datetime','open','high','low','close']); H['dt']=pd.to_datetime(H.datetime,utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m=H.dt.dt.hour*60+H.dt.dt.minute; H=H[(m>=570)&(m<960)].drop_duplicates('dt').sort_values('dt').reset_index(drop=True)
    t=T[T.tk==tk].copy().sort_values('entry'); res={}
    for L in (10,20):
        # 15m: known at bar close
        H15=H.assign(lb=lb(H.low,L),cl=H.close,known=H.dt+pd.Timedelta(minutes=15))[['known','lb','cl']].dropna()
        # 1h from 15m bars (09:30 anchored)
        h=H.set_index('dt').resample('60min',origin='start_day',offset='30min').agg({'low':'min','close':'last','high':'max'}).dropna()
        H1=pd.DataFrame({'known':h.index+pd.Timedelta(minutes=60),'lb':lb(h.low,L).values,'cl':h.close.values}).dropna()
        d=H.groupby(H.dt.dt.normalize()).agg(low=('low','min'),close=('close','last'))
        HD=pd.DataFrame({'known':d.index+pd.Timedelta(hours=16),'lb':lb(d.low,L).values,'cl':d.close.values}).dropna()
        for nm,S in (('15m',H15),('1h',H1),('1d',HD)):
            S=S.sort_values('known'); x=pd.merge_asof(t[['entry']].sort_values('entry'),S,left_on='entry',right_on='known',direction='backward')
            x.index=t.sort_values('entry').index
            res[f'd_{nm}_{L}']=x.cl/x.lb-1
            # reclaim: price crossed from below LB to above within last 4 completed bars (15m only)
        s=H15.set_index('known'); below=(s.cl<s.lb).astype(int); recl=((below.shift(1)==1)&(below==0)).astype(int).rolling(4).max()
        x=pd.merge_asof(t[['entry']].sort_values('entry'),recl.rename('rc').reset_index(),left_on='entry',right_on='known',direction='backward'); x.index=t.sort_values('entry').index
        res[f'rc_{L}']=x.rc
    F.append(t.join(pd.DataFrame(res)))
T=pd.concat(F).dropna(); T['yr']=T.entry.dt.year; print(len(T),'trades')
def row(x): return f'n {len(x):4d} {x.ret.mean()*1e4:+6.1f}bp win {(x.ret>0).mean():.0%}' if len(x)>=30 else f'n {len(x)}'
for L in (10,20):
  for nm in ('15m','1h','1d'):
    c=f'd_{nm}_{L}'; print(f'== LB {nm} L={L}: close vs maximin pivot (below / 0-1% / 1-3% / >3% above)')
    for e,mm in (('19-21',T.yr<=2021),('22-26',T.yr>=2022)):
        z=T[mm]; b=[z[z[c]<0],z[(z[c]>=0)&(z[c]<.01)],z[(z[c]>=.01)&(z[c]<.03)],z[z[c]>=.03]]
        print(f'   {e}: '+' | '.join(row(q) for q in b))
for L in (10,20):
    print(f'== reclaim cross (15m L={L}) ON/OFF',[ (e,row(T[m&(T[f"rc_{L}"]>0)]),row(T[m&(T[f"rc_{L}"]==0)])) for e,m in (('19-21',T.yr<=2021),('22-26',T.yr>=2022))])
def sim(Tx,w=None):
    Tx=Tx.sort_values(['entry','ratio'],ascending=[True,False],kind='stable')
    op=[];pend=[];real=0.;hist=[];win=deque();lvl0=0.;out=[]
    for e,x,r in zip(Tx.entry.values,Tx.exit.values,Tx.ret.values):
        done=sorted([p for p in pend if p[0]<=e],key=lambda z:z[0]);pend=[p for p in pend if p[0]>e]
        for xt,rr in done: real+=np.log1p(rr/N);hist.append((xt,real));win.append((xt,real))
        op=[o for o in op if o>e]
        if len(op)>=N: continue
        if hist:
            t0=e-np.timedelta64(28,'D')
            while win and win[0][0]<t0: lvl0=win.popleft()[1]
            if real-max([lvl0]+[h for _,h in win])<np.log1p(-BRK): continue
        op.append(x);pend.append((x,r));out.append((pd.Timestamp(e),r))
    s=pd.DataFrame(out,columns=['e','r']);s['yr']=s.e.dt.year;return s
def rep(n,Tx):
    s=sim(Tx);o=[]
    for lab,m in (('19-21',s.yr<=2021),('22-26',s.yr>=2022),('all',s.yr>0)):
        z=s[m];y=max((z.e.max()-z.e.min()).days/365.25,.1);o.append(f'{lab}: n {len(z)} ann~{np.expm1(np.log1p(z.r/N).sum()/y):+.1%}')
    print(f'{n:34s} '+' | '.join(o))
print('=== slot sim'); rep('base',T)
for nm in ('15m','1h','1d'):
    rep(f'skip below LB {nm} L20',T[T[f'd_{nm}_20']>=0]); rep(f'only 0-3% above LB {nm} L20',T[(T[f'd_{nm}_20']>=0)&(T[f'd_{nm}_20']<.03)])
rep('skip below LB on 1d AND 1h',T[(T.d_1d_20>=0)&(T.d_1h_20>=0)])
rep('reclaim only (15m L20)',T[T.rc_20>0])
