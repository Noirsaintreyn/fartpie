"""Report 273 (declared before results). Rey: 'wish the vector worked on GC, CL, other futures so I could screen it like stocks'. No GC/CL futures on disk, so ETF proxies on 15m RTH bars (GLD SLV USO TLT FXE FXB FXY, plus SPY IWM DIA QQQ as equity controls), vector parameters UNCHANGED from NQ (ATR-based, scale-free). Trade = follow the vector regime: enter at the close of the flip bar, hold to the next flip (long when regime +1, short when -1), cost 2 bps round trip. Metrics: trades/yr, mean % per trade, t, share of years positive, train 2019-21 | test 2018+2022-26. Pass bar: positive and |t|>=2 in test, not negative in train, on at least several assets (else the vector is an NQ/equity-trend tool, not a general tool)."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'/home/claude/fartpie/xs')
from vector_pine import run_state_machine
P='/mnt/user-data/uploads/fartpie/lse_ohlcv_cache/'; import glob, os
rows=[]
for t in ('GLD','SLV','USO','TLT','FXE','FXB','FXY','SPY','IWM','DIA','QQQ'):
    f=glob.glob(P+t+'_15m_*.parquet') or glob.glob('/mnt/user-data/uploads/fartpie/lse_clean/'+t+'_15m_*.parquet')
    if not f: print(t,'missing'); continue
    d=pd.read_parquet(f[0]); d['datetime']=pd.to_datetime(d.datetime); d['datetime']=d.datetime.dt.tz_localize('UTC').dt.tz_convert('America/New_York').dt.tz_localize(None) if d.datetime.dt.tz is None else d.datetime.dt.tz_convert('America/New_York').dt.tz_localize(None)
    d=d.set_index('datetime').sort_index().between_time('09:30','15:45')
    reg=np.asarray(run_state_machine(d.assign(datetime=d.index))[0]); c=d.close.values
    ch=np.where(np.diff(reg)!=0)[0]+1; ch=ch[reg[ch]!=0]
    for a,b in zip(ch[:-1],ch[1:]):
        r=reg[a]*(c[b]/c[a]-1)-0.0002; rows.append((t,d.index[a],r))
T=pd.DataFrame(rows,columns=['t','dt','r']); T['yr']=T.dt.dt.year; T['sp']=np.where((T.yr>=2019)&(T.yr<=2021),'train','test')
def st(x): return f'n{len(x):4d} {x.mean()*100:+.3f}% (t{x.mean()/(x.std()/np.sqrt(len(x))):+.1f})' if len(x)>3 else 'n<4'
print('asset   trades/yr | train | test | years positive')
for t,g in T.groupby('t'):
    yp=g.groupby('yr').r.sum(); print(f'{t:5s} {len(g)/ (g.yr.nunique()):5.0f} | {st(g[g.sp=="train"].r)} | {st(g[g.sp=="test"].r)} | {(yp>0).sum()}/{len(yp)}')
