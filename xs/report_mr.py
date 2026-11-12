"""MEAN-REVERSION models for NQ (declared before results). 5m bars (ETH), RTH 09:30-16:00 ET, cost 0.5 pt/trade, entries fill at the NEXT bar's open, stop-first when both touched, time stop 24 bars (2h) / 15:55.
A SWEEP-RECLAIM: a 5m bar trades through PDH/PDL/ONH/ONL (prior-day / overnight extremes) and a close back inside within 6 bars -> fade it. Stop = sweep extreme +-2; skip risk <4 or >40 pts. Exits: 1R, 2R, MID (prior-day midpoint).
C VWAP-STRETCH: price >= 0.30 ADR20 from session VWAP, then a reversal bar (close back toward VWAP) -> fade. Stop 0.15 ADR beyond entry, exits VWAP-at-entry, 1R.
D RANGE-EDGE ROTATION (days that open inside the prior range): first touch within 0.05 ADR of PDH/PDL from inside -> fade to MID. Stop 0.15 ADR beyond edge.
Scored on all days and on 'idle' days (core legs flat), eras 2018-21 / 2022-26."""
import numpy as np, pandas as pd, pickle
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').sort_values('dt').drop_duplicates('dt').reset_index(drop=True)
mn=b.dt.dt.hour*60+b.dt.dt.minute; b['d']=b.dt.dt.normalize(); rth=b[(mn>=570)&(mn<960)].copy()
idle=pd.read_pickle(SC+'idle_state.pkl')['trade']; idle=~idle.astype(bool)     # True = core legs idle
D=rth.groupby('d').agg(o=('open','first'),h=('high','max'),l=('low','min'),c=('close','last')); D['rng']=D.h-D.l
D['adr']=D.rng.rolling(20).mean().shift(); D['pdh']=D.h.shift(); D['pdl']=D.l.shift(); D['pdc']=D.c.shift()
days=list(D.index); eth=b.set_index('dt')
ONH={};ONL={}
for i,d in enumerate(days):
    if i==0: continue
    x=eth.loc[days[i-1]+pd.Timedelta(hours=18):d+pd.Timedelta(hours=9,minutes=29),['high','low']]
    if len(x): ONH[d]=x.high.max(); ONL[d]=x.low.min()
G={d:g.reset_index(drop=True) for d,g in rth.groupby('d')}
COST=.5
def resolve(g,i,dr,stop,tgt,maxb=24):
    e=g.open.iloc[i]; hi=g.high.values; lo=g.low.values; n=min(len(g),i+maxb)
    for j in range(i,n):
        if dr>0:
            if lo[j]<=stop: return stop-e-COST
            if tgt is not None and hi[j]>=tgt: return tgt-e-COST
        else:
            if hi[j]>=stop: return e-stop-COST
            if tgt is not None and lo[j]<=tgt: return e-tgt-COST
    return (g.close.iloc[n-1]-e)*dr-COST
rows=[]
for d in days:
    if d not in G or d not in ONH or not D.adr.get(d)==D.adr.get(d): continue
    g=G[d]; adr=D.adr[d]; pdh,pdl=D.pdh[d],D.pdl[d]; mid=(pdh+pdl)/2; op=g.open.iloc[0]; inside=pdl<op<pdh
    hi=g.high.values; lo=g.low.values; cl=g.close.values; m_=(g.dt.dt.hour*60+g.dt.dt.minute).values; N=len(g)
    # ---- A
    got=set()
    for nm,L,up in (('PDH',pdh,1),('ONH',ONH[d],1),('PDL',pdl,0),('ONL',ONL[d],0)):
        for i in range(1,N-2):
            if m_[i]>870: break
            if (up and hi[i]>L) or ((not up) and lo[i]<L):
                ext=hi[i] if up else lo[i]
                for j in range(i,min(i+6,N-2)):
                    ext=max(ext,hi[j]) if up else min(ext,lo[j])
                    if (up and cl[j]<L) or ((not up) and cl[j]>L):
                        dr=-1 if up else 1; e=g.open.iloc[j+1]; st=ext+2*(1 if up else -1); R=abs(e-st)
                        if 4<=R<=40 and (nm,dr) not in got:
                            got.add((nm,dr))
                            for ex in ('1R','2R','MID'):
                                if ex=='MID': t=mid; ok=(t<e) if dr<0 else (t>e)
                                else: t=e+dr*R*(1 if ex=='1R' else 2); ok=True
                                if ok: rows.append(dict(day=d,m='A',sub=nm,ex=ex,t=g.dt.iloc[j+1],dir=dr,pts=resolve(g,j+1,dr,st,t),R=R,inside=inside))
                        break
                break
    # ---- C
    vw=((g.high+g.low+g.close)/3*g.volume).cumsum()/g.volume.cumsum(); done=False
    for i in range(6,N-3):
        if m_[i]<600 or m_[i]>900: continue
        s=cl[i-1]-vw.iloc[i-1]
        if abs(s)>=.30*adr and ((s>0 and cl[i]<cl[i-1]) or (s<0 and cl[i]>cl[i-1])):
            dr=-1 if s>0 else 1; e=g.open.iloc[i+1]; st=e-dr*(-1)*.15*adr if False else e+(-dr)*(-1)*0   # placeholder
            st=e+(.15*adr if dr<0 else -.15*adr); t_v=vw.iloc[i]
            R=abs(e-st)
            for ex in ('VWAP','1R'):
                t=t_v if ex=='VWAP' else e+dr*R
                if (dr<0 and t<e) or (dr>0 and t>e): rows.append(dict(day=d,m='C',sub='stretch',ex=ex,t=g.dt.iloc[i+1],dir=dr,pts=resolve(g,i+1,dr,st,t),R=R,inside=inside))
            break
    # ---- D
    if inside:
        for i in range(1,N-3):
            if m_[i]<580 or m_[i]>870: continue
            if hi[i]>=pdh-.05*adr or lo[i]<=pdl+.05*adr:
                dr=-1 if hi[i]>=pdh-.05*adr else 1; e=g.open.iloc[i+1]; edge=pdh if dr<0 else pdl; st=edge+(.15*adr if dr<0 else -.15*adr)
                if ((dr<0 and mid<e<st) or (dr>0 and st<e<mid)): rows.append(dict(day=d,m='D',sub='edge',ex='MID',t=g.dt.iloc[i+1],dir=dr,pts=resolve(g,i+1,dr,st,mid),R=abs(e-st),inside=inside))
                break
T=pd.DataFrame(rows); T['era']=np.where(T.day<'2022-01-01','A 18-21','B 22-26'); T['idle']=T.day.map(idle).fillna(False)
T.to_pickle(SC+'mr_trades.pkl'); print(len(T),'signals')
def line(x):
    if len(x)<20: return f'n {len(x):4d}'
    return f'n {len(x):4d} avg {x.pts.mean():+6.2f} win {(x.pts>0).mean():.0%} t {x.pts.mean()/(x.pts.std()/len(x)**.5):+.1f}'
for (m,sub,ex),x in T.groupby(['m','sub','ex']):
    print(f'{m} {sub:7s} {ex:5s}','| ALL:', ' || '.join(f'{e[:1]} {line(x[x.era==e])}' for e in sorted(x.era.unique())),'\n'+' '*18+'| IDLE:',' || '.join(f'{e[:1]} {line(x[(x.era==e)&x.idle])}' for e in sorted(x.era.unique())))
print('--- model-level (first signal per day, best-known exit not chosen): per idle day pts')
for (m,ex),x in T.groupby(['m','ex']):
    x1=x.sort_values('t').drop_duplicates('day')
    out=[]
    for e in ('A 18-21','B 22-26'):
        z=x1[(x1.era==e)&x1.idle]; nd=idle[(idle.index>=('2018-01-01' if e[0]=='A' else '2022-01-01'))&(idle.index<('2022-01-01' if e[0]=='A' else '2027-01-01'))].sum()
        out.append(f'{e[:1]}: {len(z)} trades on {nd} idle days, {z.pts.sum()/max(nd,1):+.2f} pts/idle-day, avg {z.pts.mean():+.2f}')
    print(f'{m} {ex:5s}',' | '.join(out))
