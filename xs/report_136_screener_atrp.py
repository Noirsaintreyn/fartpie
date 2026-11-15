"""report 136: ATRP / EXT on the live large-cap screener book (S&P PIT, options ratio>=1.5, 6 slots, 8% breaker, ties by ratio).
Train 2019-21, test 2022-26. Realized-return equity (no daily MTM) - for relative comparison only."""
import glob, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from collections import deque
R='/mnt/user-data/uploads/fartpie/'
T=pd.read_parquet(R+'xs/ext/wide_T.parquet'); T=T[T.sp&(T.ratio>=1.5)].copy()
D={}
for tk in T.tk.unique():
    f=glob.glob(R+f'lse_clean/{tk}_15m_*.parquet')+glob.glob(R+f'lse_ohlcv_cache/{tk}_15m_*.parquet')
    H=pd.read_parquet(f[0],columns=['datetime','open','high','low','close']); H['dt']=pd.to_datetime(H.datetime,utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m=H.dt.dt.hour*60+H.dt.dt.minute; H=H[(m>=570)&(m<960)].drop_duplicates('dt').set_index('dt').sort_index()
    g=H.groupby(H.index.normalize()); d=pd.DataFrame({'h':g.high.max(),'l':g.low.min(),'c':g.close.last()})
    pc=d.c.shift(); tr=pd.concat([d.h-d.l,(d.h-pc).abs(),(d.l-pc).abs()],axis=1).max(axis=1)
    d['ATRP']=(tr.rolling(14).mean()/d.c).shift(1)          # known before entry day
    d['EXT']=(d.c/d.l.rolling(60).min()-1).shift(1)
    d['R20']=(d.c/d.c.shift(20)-1).shift(1)
    D[tk]=d
def feat(r,c): 
    d=D[r.tk]; i=d.index.searchsorted(r.entry.normalize()); return d[c].iloc[i] if i<len(d) else np.nan
for c in ('ATRP','EXT','R20'): T[c]=[feat(r,c) for r in T.itertuples()]
T=T.dropna(subset=['ATRP','EXT']); T['yr']=T.entry.dt.year; T['tr']=T.yr<=2021
print(len(T),'trades with features')
print('\ntrade-level (all signals, ratio>=1.5): mean ret% and rank-corr(feature,ret) by half')
for c in ('ATRP','EXT','R20','ratio'):
    for nm,s in (('train',T[T.tr]),('test',T[~T.tr])):
        print(f'  {c:6s}{nm}: spearman {s[c].corr(s.ret,method="spearman"):+.3f}  top-tercile mean {100*s[s[c]>=s[c].quantile(2/3)].ret.mean():+.3f}%  bottom {100*s[s[c]<=s[c].quantile(1/3)].ret.mean():+.3f}%')
thr={c:T[T.tr][c].quantile([1/3,2/3]).values for c in ('ATRP','EXT')}   # frozen from train
N=6;BRK=.08
def book(S,key,skip=None,asc=None):
    S=S.copy()
    if skip is not None: S=S[~skip(S)]
    S=S.assign(_k=key(S)).sort_values(['entry','_k','tk'],ascending=[True,False,True],kind='stable')
    op=[];pend=[];real=0.;hist=[];win=deque();lvl0=0.;out=[]
    for e,x,r,tk in zip(S.entry.values,S.exit.values,S.ret.values,S.tk.values):
        done=sorted([p for p in pend if p[0]<=e],key=lambda z:z[0]); pend=[p for p in pend if p[0]>e]
        for xt,rr in done: real+=np.log1p(rr/N); hist.append((xt,real)); win.append((xt,real))
        op=[o for o in op if o>e]
        if len(op)>=N: continue
        if hist:
            t0=e-np.timedelta64(28,'D')
            while win and win[0][0]<t0: lvl0=win.popleft()[1]
            if real-max([lvl0]+[h for _,h in win])<np.log1p(-BRK): continue
        op.append(x); pend.append((x,r)); out.append((pd.Timestamp(e),pd.Timestamp(x),r))
    return pd.DataFrame(out,columns=['e','x','r'])
def st(B,a,b):
    z=B[(B.x>=a)&(B.x<=b+'-12-31')].sort_values('x'); 
    if len(z)==0: return 'n/a'
    eq=np.cumsum(np.log1p(z.r.values/N)); y=(z.x.iloc[-1]-z.x.iloc[0]).days/365.25
    dd=(np.exp(eq)/np.maximum.accumulate(np.exp(eq))-1).min()
    return f'{np.expm1(eq[-1]/y):+6.1%}/{dd:5.0%} n{len(z):4d} avg{100*z.r.mean():+.2f}%'
V={'live (ratio)':(lambda S:S.ratio,None),
 'ATRP high first':(lambda S:S.ATRP+1e-9*S.ratio,None),
 'ATRP low first':(lambda S:-S.ATRP,None),
 'ratio+ATRP rank':(lambda S:S.ratio.rank(pct=True)+S.ATRP.rank(pct=True),None),
 'EXT high first':(lambda S:S.EXT,None),
 'ratio, skip ATRP<tercile1':(lambda S:S.ratio,lambda S:S.ATRP<thr['ATRP'][0]),
 'ratio, skip ATRP>tercile2':(lambda S:S.ratio,lambda S:S.ATRP>thr['ATRP'][1]),
 'ratio, skip EXT<tercile1':(lambda S:S.ratio,lambda S:S.EXT<thr['EXT'][0]),
 'ratio, skip EXT>tercile2':(lambda S:S.ratio,lambda S:S.EXT>thr['EXT'][1])}
print('\nbook (CAGR/maxDD)           2019-21                         2022-26')
for k,(key,sk) in V.items():
    B=book(T,key,sk); print(f'{k:28s}{st(B,"2019","2021")}   {st(B,"2022","2026")}')
T.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/screener_T_feat.pkl')
