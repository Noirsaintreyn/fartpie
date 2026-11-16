"""report 174: options flow gates INSIDE the 0DTE era only (Rey: compare 2018 to now is wrong; 0DTE hedging flow only matters since 2022). DECLARED BEFORE RESULTS. Walk-forward inside the regime: cuts frozen on W1 = 2022-05-01..2023-12-31, judged on W2 = 2024-01-01..2025-12-15 (greeks/walls data end 2025-12-15). 2022-05 = daily SPX expiries complete.
Trade sets: S1 premium fade (s1_trades162.pkl) and IFVG (uni_ifvg.pkl). Gates (all use prior-day values): net vanna low/mid/high tercile (W1 cuts, raw: the drift problem is mostly gone inside one regime), vanna sign, charm sign, 250d z of vanna; GEX wall room: ROOM = dir*(target wall - price)/ATRd > 0.5 where target wall = call wall for longs / put wall for shorts (room to run before the wall), WALL_NEAR = price within 0.5 ATRd of either wall.
Report W1 | W2 for IN and OUT; PASS (relaxed for small n) = same sign, |t|>=2, n>=20 in both windows."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
g=pd.read_csv('/mnt/user-data/uploads/fartpie/options_greeks_daily_v2_QQQ.csv',parse_dates=['date']).set_index('date'); g['z']=(g.net_vanna-g.net_vanna.rolling(250,min_periods=120).mean())/g.net_vanna.rolling(250,min_periods=120).std(); g=g.shift(1)
w=pd.read_csv('/mnt/user-data/uploads/fartpie/gex_walls_nq.csv',parse_dates=['date']).set_index('date').shift(1)
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b['d']=b.dt.dt.normalize(); b['m']=b.dt.dt.hour*60+b.dt.dt.minute
r=b[(b.m>=570)&(b.m<=955)].groupby('d').agg(o=('open','first'),h=('high','max'),l=('low','min')); atr=(r.h-r.l).rolling(14).mean().shift(1)
S1=pd.read_pickle(S+'s1_trades162.pkl').rename(columns={'day':'d'}); S1['d']=pd.to_datetime(S1.d); S1['ent']=S1.d.map(r.o); S1['dr']=-1
I=pd.DataFrame(pd.read_pickle(S+'ext/uni_ifvg.pkl')); I=I[I.model=='IFVG'].copy(); I['d']=pd.to_datetime(I.d)
T1,T2=('2022-05-01','2023-12-31'),('2024-01-01','2025-12-15')
def prep(T):
    T=T[(T.d>=T1[0])&(T.d<=T2[1])].copy()
    for c in ('net_vanna','net_charm','z'): T[c]=T.d.map(g[c])
    T['cw']=T.d.map(w.call_wall_nq); T['pw']=T.d.map(w.put_wall_nq); T['atr']=T.d.map(atr); T['w']=np.where(T.d<=T1[1],1,2)
    tw=np.where(T.dr>0,T.cw,T.pw); T['room']=T.dr*(tw-T.ent)/T.atr; T['near']=((T.ent-T.cw).abs()/T.atr<=.5)|((T.ent-T.pw).abs()/T.atr<=.5)
    return T.dropna(subset=['net_vanna'])
def st(s):
    n=len(s); return f'n{n:4d} {s.mean():+7.2f} t{(s.mean()/(s.std()/np.sqrt(n)) if n>2 and s.std()>0 else 0):+5.1f} w{(s>0).mean()*100:3.0f}%' if n else 'n   0'
for nm,T0 in (('S1 premium fade',S1),('IFVG',I)):
    T=prep(T0); a=T[T.w==1]; print(f'\n{nm}: n {len(T)} (W1 {len(a)}, W2 {(T.w==2).sum()})  base W1 {st(a.pts)} | W2 {st(T[T.w==2].pts)}')
    q=a.net_vanna.quantile([1/3,2/3]).values
    G={'vanna low3rd':T.net_vanna<=q[0],'vanna mid3rd':(T.net_vanna>q[0])&(T.net_vanna<=q[1]),'vanna high3rd':T.net_vanna>q[1],'vanna>0':T.net_vanna>0,'charm>0':T.net_charm>0,'z<-0.5':T.z<-.5,'z>0.5':T.z>.5,'ROOM>0.5ATR':T.room>.5,'ROOM<=0.5ATR':T.room<=.5,'NEAR a wall':T.near}
    for lab,m in G.items():
        x,y=T[(T.w==1)&m].pts,T[(T.w==2)&m].pts; ok=len(x)>=20 and len(y)>=20 and abs(x.mean()/(x.std()/np.sqrt(len(x))))>=2 and abs(y.mean()/(y.std()/np.sqrt(len(y))))>=2 and np.sign(x.mean())==np.sign(y.mean())
        print(f'  {lab:14s} W1 {st(x)} | W2 {st(y)} {"PASS" if ok else ""}')
