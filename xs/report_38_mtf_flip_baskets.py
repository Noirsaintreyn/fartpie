"""Nov 14 2026: 15m vector flip alone vs 15m flip WITH 1h vector already bull (MTF), +/- 200d EMA, on 9 LSE names; per-trade bp and daily equal-weight baskets. Era split 2018-21 / 2022+. 1h state used only after that 1h bar has closed (PIT)."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'/home/claude/fartpie/xs')
from vector_pine import run_state_machine
ERA={'18-21':('2018-01-01','2022-01-01'),'22+':('2022-01-01','2027-01-01')}
def load(f):
    d=pd.read_parquet(f); d['datetime']=pd.to_datetime(d.datetime); d=d.drop_duplicates('datetime').set_index('datetime').sort_index(); d.index=d.index.tz_localize('UTC').tz_convert('America/New_York').tz_localize(None); return d[['open','high','low','close','volume']].astype(float).between_time('09:30','15:59')
def tt(a,b=None):
    if b is None: return a.mean()/(a.std()/np.sqrt(len(a))) if len(a)>5 else np.nan
    return (a.mean()-b.mean())/np.sqrt(a.var()/len(a)+b.var()/len(b)) if len(a)>5 and len(b)>5 else np.nan
rows=[]
for f in sorted(glob.glob('lse_clean/*.parquet')):
    tk=f.split('/')[-1].split('.')[0]; x=load(f)
    z=x.resample('60min',offset='30min').agg(dict(open='first',high='max',low='min',close='last',volume='sum')).dropna(subset=['close'])
    r1,_,_,_=run_state_machine(z.assign(datetime=z.index)); h=pd.DataFrame({'end':z.index+pd.Timedelta(minutes=60),'r1':np.asarray(r1)}).sort_values('end')
    reg,lev,_,_=run_state_machine(x.assign(datetime=x.index)); reg=np.asarray(reg); lev=np.asarray(lev)
    dc=x.close.groupby(x.index.normalize()).last(); dup=(dc>dc.ewm(span=200,adjust=False).mean()).shift(1)
    m=pd.merge_asof(pd.DataFrame({'t':x.index+pd.Timedelta(minutes=15)}),h,left_on='t',right_on='end'); r1a=m.r1.values
    lb=x.low.rolling(20).min().rolling(20).max().values; c=x.close.values; o=x.open.values; n=len(x); idx=x.index
    for i in range(300,n-17):
        if reg[i]==1 and reg[i-1]!=1 and c[i]>lb[i] and idx[i+1].date()==idx[i].date():
            en=o[i+1]; xx=i+1
            while xx<n-1 and reg[xx]==1 and c[xx]>=lev[xx] and idx[xx].date()==idx[i].date(): xx+=1
            rows.append(dict(tk=tk,t=idx[i],d=idx[i].normalize(),h1=r1a[i]==1,up=dup.get(idx[i].normalize(),np.nan),r16=(c[i+16]/en-1)*1e4 if idx[i+16].date()==idx[i].date() else np.nan,hold=(c[xx]/en-1)*1e4,nat=(c[min(xx,n-1)]/en-1)*1e4))
P=pd.DataFrame(rows).dropna(subset=['up']); P['up']=P.up.astype(bool)
S={'A 15m flip alone':P.index==P.index,'B 15m flip + 1h bull':P.h1,'C 15m flip, 1h NOT bull':~P.h1,'D 15m + 1h bull + >200dEMA':P.h1&P.up,'E 15m + >200dEMA':P.up,'F 15m + 1h bull + <200dEMA':P.h1&~P.up}
print('n flips',len(P),'| share with 1h bull',round(P.h1.mean(),2))
print('\nPER TRADE (bp), exit: same-day hold-to-bear (hold)')
for k,msk in S.items():
    line=f'{k:28s}'
    for e,(a,b) in ERA.items():
        s=P[msk&(P.t>=a)&(P.t<b)]; line+=f' | {e}: n{len(s):4d} {s.hold.mean():+6.1f} t{tt(s.hold):+4.1f}'
    print(line)
print('\nPER TRADE (bp), exit: 16 bars (4h) same day')
for k,msk in S.items():
    line=f'{k:28s}'
    for e,(a,b) in ERA.items():
        s=P[msk&(P.t>=a)&(P.t<b)].dropna(subset=['r16']); line+=f' | {e}: n{len(s):4d} {s.r16.mean():+6.1f} t{tt(s.r16):+4.1f}'
    print(line)
print('\nB vs C (1h bull minus not), hold: '+' | '.join(f"{e} {P[P.h1&(P.t>=a)&(P.t<b)].hold.mean()-P[~P.h1&(P.t>=a)&(P.t<b)].hold.mean():+.1f} t{tt(P[P.h1&(P.t>=a)&(P.t<b)].hold,P[~P.h1&(P.t>=a)&(P.t<b)].hold):+.1f}" for e,(a,b) in ERA.items()))
print('\nBASKETS: equal-weight mean of that day\'s signals (hold exit), bp per basket-day')
for k,msk in S.items():
    line=f'{k:28s}'
    for e,(a,b) in ERA.items():
        s=P[msk&(P.t>=a)&(P.t<b)]; g=s.groupby('d').hold.mean(); line+=f' | {e}: {len(g):4d}d {g.mean():+6.1f} t{tt(g):+4.1f} avg names/d {s.groupby("d").size().mean():.1f}'
    print(line)
