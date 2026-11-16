"""report 157: backtest of Rey's Pine strategy 'MES Dual EMA + Filters' on ES 5m (2014-2026-01; the screenshot window Mar-Apr 2026 is outside our data). Rules copied from the Pine file; nothing tuned.
Entry = market at NEXT bar open after the crossover bar; stop/limit = signal-bar CLOSE -+ stop/tp. Both hit in one bar -> stop first. Position gating: longs only if prior GLOBEX session close>open (else shorts only), long only above daily-reset VWAP(hlc3), (optional) >=10 pts from rolling 24h high AND low (288 5m bars), no entries 200 min before 16:00 (12:40-16:00 ET), flat at the 15:55 bar close, pyramiding 2 same direction, no flip. Roll days skipped for entries.
Chart in the screenshot is 3m; data here is 5m, so EMA lengths run RAW (80/90 bars) and TIME-SCALED (x0.6 -> 48/54 bars). Commission 1.25/side/MES = 0.5 pt round trip; SLIP adds 0.5 pt.
CONFIGS: SHOT = A 80/90 stop 8 tp 15, B off, far-from-24h on (10), block 200 (screenshot). DEF = script defaults A 9/21 5/10 + B 12/34 8/16, far off, block 30. A 2018-21 | B 2022+."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
e=pd.read_parquet('/tmp/claude-0/ml/es5_real.parquet').sort_values('dt').reset_index(drop=True)
e=e[(e.dt>='2017-06-01')].reset_index(drop=True)
day=e.day.values; sess=pd.DataFrame({'d':day,'o':e.open,'c':e.close}).groupby('d').agg(o=('o','first'),c=('c','last')); sess['bull']=(sess.c>sess.o); prev=sess.bull.shift(1)
e['bull']=pd.Series(day).map(prev).fillna(False).values.astype(bool)
tp=(e.high+e.low+e.close)/3; e['vwap']=(tp*e.volume).groupby(e.day).cumsum()/e.volume.groupby(e.day).cumsum()
e['hh24']=e.high.rolling(288,min_periods=1).max(); e['ll24']=e.low.rolling(288,min_periods=1).min()
O,H,L,C=[e[k].values.astype(float) for k in ('open','high','low','close')]; M=e.m.values; DT=e.dt.values; ROLL=e.roll.values.astype(bool); BULL=e.bull.values; VW=e.vwap.values; HH=e.hh24.values; LL=e.ll24.values
def ema(n): return e.close.ewm(span=n,adjust=False).mean().values
def cross(f,s):
    up=(f[1:]>s[1:])&(f[:-1]<=s[:-1]); dn=(f[1:]<s[1:])&(f[:-1]>=s[:-1]); return np.r_[False,up],np.r_[False,dn]
def backtest(setups,far,block,sc):
    sig=[]
    for (fa,sl,st,tpp) in setups:
        f,s=ema(max(2,int(round(fa*sc)))),ema(max(3,int(round(sl*sc)))); u,d=cross(f,s); sig.append((u,d,st,tpp))
    pos=[]; trades=[]; pend=[]; n=len(e); wu=1  # pos items: dict dir, stop, lim, ent
    for i in range(n-1):
        for p in pend:   # fill at this bar open
            if len(pos)<2 and (not pos or pos[0]['dr']==p['dr']): p['ent']=O[i]; pos.append(p)
        pend=[]
        new=[]
        for p in pos:
            dr=p['dr']; ex=None
            if (O[i]<=p['stop']) if dr>0 else (O[i]>=p['stop']): ex=O[i]
            elif (L[i]<=p['stop']) if dr>0 else (H[i]>=p['stop']): ex=p['stop']
            elif (H[i]>=p['lim']) if dr>0 else (L[i]<=p['lim']): ex=p['lim']
            elif 955<=M[i]<960: ex=C[i]
            if ex is None: new.append(p)
            else: trades.append((DT[i],(ex-p['ent'])*dr))
        pos=new
        if ROLL[i]: continue
        blk=(960-block)<=M[i]<960; 
        if blk: continue
        longok=BULL[i] and C[i]>VW[i]; shortok=(not BULL[i]) and C[i]<VW[i]
        if far: fr=(HH[i]-C[i])>=10 and (C[i]-LL[i])>=10; longok&=fr; shortok&=fr
        for (u,d,st,tpp) in sig:
            if u[i] and longok and (not pos or pos[0]['dr']>0) : pend.append(dict(dr=1,stop=C[i]-st,lim=C[i]+tpp))
            if d[i] and shortok and (not pos or pos[0]['dr']<0): pend.append(dict(dr=-1,stop=C[i]+st,lim=C[i]-tpp))
    return pd.DataFrame(trades,columns=['dt','gross'])
CFG={'SHOT':([(80,90,8,15)],True,200),'DEF':([(9,21,5,10),(12,34,8,16)],False,30)}
def rep(T,nm,slip):
    T=T.copy(); T['dt']=pd.to_datetime(T.dt); T['pts']=T.gross-.5-slip; T=T[(T.dt>='2018-02-01')&(T.dt<'2026-02-01')]; T['era']=np.where(T.dt<'2022-01-01','A','B')
    def s(g):
        n=len(g); 
        if n==0: return 'n   0'
        pf=g.pts[g.pts>0].sum()/max(-g.pts[g.pts<0].sum(),1e-9); cu=g.pts.cumsum(); dd=(cu.cummax()-cu).max()
        return f'n{n:5d} {g.pts.mean():+5.2f} t{(g.pts.mean()/(g.pts.std()/np.sqrt(n)) if n>2 else 0):+5.1f} w{(g.pts>0).mean()*100:3.0f}% PF{pf:4.2f} tot{g.pts.sum():+7.0f} dd{dd:5.0f}'
    print(f'{nm:20s} A {s(T[T.era=="A"])} | B {s(T[T.era=="B"])}')
    return T
for cn,(su,far,blk) in CFG.items():
    for scn,sc in (('raw',1.0),('x0.6',0.6)):
        T=backtest(su,far,blk,sc)
        for slip in (0,.5): X=rep(T,f'{cn}_{scn}_slip{slip}',slip)
        if cn=='SHOT' and scn=='raw': X0=X
print('\nSHOT_raw by year (pts, n):'); print(X0.assign(y=X0.dt.dt.year).groupby('y').pts.agg(['size','sum']).round(0).T.to_string())
