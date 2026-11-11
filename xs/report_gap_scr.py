import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from collections import deque
N=6; BRK=.08
T=pd.read_parquet('xs/ext/wide_T.parquet'); print(T['var'].value_counts().to_dict())
T=T[T.sp&(T.ratio>=1.5)].copy()
G={}
for tk in sorted(T.tk.unique()):
    f=glob.glob(f'lse_clean/{tk}_15m_2018-01-01_*.parquet')
    if not f: continue
    H=pd.read_parquet(f[0],columns=['datetime','open','close']); H['dt']=pd.to_datetime(H.datetime,utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
    m=H.dt.dt.hour*60+H.dt.dt.minute; H=H[(m>=570)&(m<960)].drop_duplicates('dt').sort_values('dt')
    d=H.dt.dt.normalize(); g=H.groupby(d).agg(o=('open','first'),c=('close','last'))
    g['gap']=g.o/g.c.shift()-1; g['gabs']=g.gap.abs().rolling(20).mean().shift()   # prior-20d typical gap, no lookahead
    g['gz']=g.gap/g.gabs
    G[tk]=g
T['d']=T.entry.dt.normalize()
T['gap']=[G[t].gap.get(d,np.nan) if t in G else np.nan for t,d in zip(T.tk,T.d)]
T['gz']=[G[t].gz.get(d,np.nan) if t in G else np.nan for t,d in zip(T.tk,T.d)]
T=T.dropna(subset=['gz']); T['yr']=T.entry.dt.year; T['era']=np.where(T.yr<=2021,'A 19-21','B 22-26')
print(len(T),'trades with gap')
T['open_bar']=(T.entry.dt.hour*60+T.entry.dt.minute)<=585
def row(n,x): print(f'{n:28s} n {len(x):5d} mean {x.ret.mean()*1e4:+7.1f}bp win {(x.ret>0).mean():.0%} t {x.ret.mean()/(x.ret.std()/np.sqrt(len(x))+1e-12):+.1f}')
bins=[-99,-1.5,-.5,.5,1.5,99]; T['gb']=pd.cut(T.gz,bins,labels=['<-1.5','-1.5..-.5','flat','.5..1.5','>1.5'])
for e,x in [('ALL',T)]+list(T.groupby('era')):
    print('==',e)
    for b,y in x.groupby('gb'): row(f'gap z {b}',y)
    row('first-bar entries',x[x.open_bar]); row('later entries',x[~x.open_bar])
# slot sim with rule variants. mode: base, skip(gz<thr), pref(gap-aligned tie-break within same entry time)
def sim(Tx,key):
    Tx=Tx.sort_values(['entry']+key,ascending=[True]+[False]*len(key),kind='stable')
    op=[];pend=[];real=0.;hist=[];win=deque();lvl0=0.;out=[]
    for e,x,r,i in zip(Tx.entry.values,Tx.exit.values,Tx.ret.values,Tx.index):
        done=sorted([p for p in pend if p[0]<=e],key=lambda z:z[0]);pend=[p for p in pend if p[0]>e]
        for xt,rr in done: real+=np.log1p(rr/N);hist.append((xt,real));win.append((xt,real))
        op=[o for o in op if o>e]
        if len(op)>=N: continue
        if hist:
            t0=e-np.timedelta64(28,'D')
            while win and win[0][0]<t0: lvl0=win.popleft()[1]
            if real-max([lvl0]+[h for _,h in win])<np.log1p(-BRK): continue
        op.append(x);pend.append((x,r));out.append((pd.Timestamp(e),r))
    s=pd.DataFrame(out,columns=['e','r']);s['yr']=s.e.dt.year
    return s
def rep(name,Tx,key):
    s=sim(Tx,key); out=[]
    for lab,m in (('19-21',s.yr<=2021),('22-26',s.yr>=2022),('all',s.yr>0)):
        z=s[m]; yrs=max((z.e.max()-z.e.min()).days/365.25,.1)
        ec=np.log1p(z.r/N).sum(); out.append(f'{lab}: n {len(z)} avg {z.r.mean()*1e4:+.1f}bp ann~{np.expm1(ec/yrs):+.1%}')
    print(f'{name:34s} '+' | '.join(out))
T['ratio_']=T.ratio
print('\n=== live-rule slot sim (n/ann are trade-sum approx, no MTM) ===')
rep('base: highest ratio',T,['ratio'])
T['al']=T.gz            # long-only, gap up = aligned
rep('tie-break: bigger gap z first',T,['al','ratio'])
T['al2']=-T.gz
rep('tie-break: smaller gap first',T,['al2','ratio'])
for thr in (-.5,0,.5):
    rep(f'skip gz<{thr}',T[T.gz>=thr],['ratio'])
for thr in (.5,1.0):
    rep(f'skip gz>{thr} (fade-gap out)',T[T.gz<=thr],['ratio'])
rep('first-bar gap-up only (gz>.5)',T[T.open_bar&(T.gz>.5)],['ratio'])
