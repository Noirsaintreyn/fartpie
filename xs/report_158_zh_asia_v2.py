"""report 158: backtest of Rey's Pine 'ZONE HUNTER - ASIA V2 dbg5' on NQ 1m (ET = file ts - 6h; 2017-04..2026-07). Rules copied from the Pine file, nothing tuned. Re-implemented in Python (TradingView not available), approximations flagged:
- HTF values (4H anchored 18:00 ET, 1H, 15m) become visible at the 1m bar that completes the HTF bar (non-repainting security). pivothigh/low: left strict >, right >=.
- Session = 18:00 ET start; OR 18:00-18:29 (open of 18:00 bar -> close of 18:29 bar) gives spike; entry window 18:30-22:00, one trade per session; direction = FADE prior Globex session (prior up -> short). 3 ct if spike is against trade dir (HIGH) else 2 ct.
- Zones oldest-first: 4H swing supply/demand (pivot 3,1; body edge) and 1H FVGs; zone must match dir, 1m close inside it, 15m confirmation on last 3 completed 15m bars; stop = far edge +-30, risk <= 80; target nearest 1H swing level (pivot 5,2; age<=20000 1m bars) >= 20 pts away else 2R. Entry at bar close.
- Exit: stop/limit (stop first if both in a bar); MFE >= 1R -> stop to entry+-1 from next bar; after 180 bars if MFE < 0.5R close at bar close; 960 bars max. Cost 0.5 pt per contract round trip. A 2018-02..2021-12 | B 2022-01.."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
d=pd.read_csv('/home/claude/nqdrive/NQ_1m_ETH.csv',parse_dates=['timestamp']); d['dt']=d.timestamp-pd.Timedelta(hours=6); d=d.sort_values('dt').drop_duplicates('dt').reset_index(drop=True)
DT=d.dt.values; O,H,L,C=[d[k].values.astype(float) for k in ('open','high','low','close')]; n=len(d); M=(d.dt.dt.hour*100+d.dt.dt.minute).values
sess=(d.dt+pd.to_timedelta(np.where(d.dt.dt.hour>=18,1,0),unit='D')).dt.normalize().values
def htf(key):
    g=d.assign(k=key).groupby('k',sort=True); a=g.agg(o=('open','first'),h=('high','max'),l=('low','min'),c=('close','last')); a['pos']=g.apply(lambda x:x.index[-1]); return a.reset_index(drop=True)
h4=htf(((d.dt-pd.Timedelta(hours=18)).dt.floor('4h')).values); h1=htf(d.dt.dt.floor('60min').values); m15=htf(d.dt.dt.floor('15min').values)
# zone / target event lists
Z=[]; T_=[]  # (pos,order,top,bot,dir)
pST=pDB=pFBT=pFBR=pSH=pSL=np.nan
def chg(v,p): return (not np.isnan(v)) and (np.isnan(p) or v!=p)
a=h4; ev=[]
for k in range(4,len(a)):
    hi=a.h.values; lo=a.l.values; p=a.pos.values[k]
    ph=hi[k-1] if (hi[k-1]>hi[k-4:k-1].max() and hi[k-1]>=hi[k]) else np.nan; pl=lo[k-1] if (lo[k-1]<lo[k-4:k-1].min() and lo[k-1]<=lo[k]) else np.nan
    if chg(ph,pST): pST=ph; bot=min(a.o.values[k-1],a.c.values[k-1]); ev.append((p,0,ph,bot,-1))
    if chg(pl,pDB): pDB=pl; top=max(a.o.values[k-1],a.c.values[k-1]); ev.append((p,1,top,pl,1))
a=h1; hi=a.h.values; lo=a.l.values
for k in range(7,len(a)):
    p=a.pos.values[k]
    bt=lo[k] if lo[k]>hi[k-2] else np.nan; br=hi[k] if hi[k]<lo[k-2] else np.nan
    if chg(bt,pFBT): pFBT=bt; ev.append((p,2,bt,hi[k-2],1))
    if chg(br,pFBR): pFBR=br; ev.append((p,3,lo[k-2],br,-1))
    sh=hi[k-2] if (hi[k-2]>hi[k-7:k-2].max() and hi[k-2]>=hi[k-1:k+1].max()) else np.nan; sl=lo[k-2] if (lo[k-2]<lo[k-7:k-2].min() and lo[k-2]<=lo[k-1:k+1].min()) else np.nan
    if chg(sh,pSH): pSH=sh; T_.append((p,sh))
    if chg(sl,pSL): pSL=sl; T_.append((p,sl))
ev.sort(key=lambda x:(x[0],x[1])); zp=np.array([e[0] for e in ev]); zt=np.array([e[2] for e in ev]); zb=np.array([e[3] for e in ev]); zd=np.array([e[4] for e in ev])
T_.sort(); tp=np.array([t[0] for t in T_]); tl=np.array([t[1] for t in T_])
m15e=m15.pos.values; m15h=m15.h.values; m15l=m15.l.values; m15c=m15.c.values
def ok15(zt_,zb_,dr,pH,pL,pC):
    mid=(zb_+zt_)/2
    if dr==-1: return (pH>=zb_ and pC<zb_) or (zb_<=pC<=zt_ and pC<mid)
    return (pL<=zt_ and pC>zt_) or (zb_<=pC<=zt_ and pC>mid)
# prior session direction
sd=pd.DataFrame({'s':sess,'o':O,'c':C}).groupby('s').agg(o=('o','first'),c=('c','last')); sd['up']=sd.c>sd.o; prior=dict(zip(sd.index[1:],sd.up.values[:-1]))
first=pd.Series(np.arange(n)).groupby(sess).first(); R=[]
for s,i0 in first.items():
    if s not in prior: continue
    idx=np.where((sess==s))[0]; i_a,i_b=idx[0],idx[-1]
    mm=M[i_a:i_b+1]; 
    if mm[0]!=1800: continue
    orr=np.where((mm>=1800)&(mm<=1829))[0]
    if not len(orr): continue
    spike=1 if C[i_a+orr[-1]]>O[i_a] else -1; dr=-1 if prior[s] else 1; high=(-spike==dr); cts=3 if high else 2
    done=False
    for i in range(i_a+orr[-1]+1,i_b+1):
        if M[i]>2200: break
        if M[i]<1830: continue
        z0=np.searchsorted(zp,i,'right'); zz=zp[:z0]; sel=np.where((i-zz<=10080)&(zd[:z0]==dr)&(zb[:z0]<=C[i])&(C[i]<=zt[:z0]))[0]
        if not len(sel): continue
        j=np.searchsorted(m15e,i,'right')-1
        if j<2: continue
        for q in sel:
            if not any(ok15(zt[q],zb[q],dr,m15h[j-r],m15l[j-r],m15c[j-r]) for r in range(3)): continue
            sp=zt[q]+30 if dr==-1 else zb[q]-30; rk=abs(C[i]-sp)
            if rk<=0 or rk>80: continue
            t0=np.searchsorted(tp,i,'right'); ok=(i-tp[:t0]<=20000); lv=tl[:t0][ok]; dd=(lv-C[i])*dr; dd=dd[dd>=20]
            tg=(C[i]+dr*dd.min()) if len(dd) else C[i]+dr*2*rk
            done=True; break
        if not done: continue
        ep=C[i]; stp=sp; mfe=0.0; be=False; ex=None
        for k in range(i+1,min(i+1+960,n)):
            if (L[k]<=stp) if dr>0 else (H[k]>=stp): ex=stp; break
            if (H[k]>=tg) if dr>0 else (L[k]<=tg): ex=tg; break
            mfe=max(mfe,(H[k]-ep) if dr>0 else (ep-L[k]))
            if not be and mfe>=rk: be=True; stp=ep+dr
            bi=k-i
            if bi>=180 and mfe<rk*.5: ex=C[k]; break
            if bi>=960: ex=C[k]; break
        if ex is None: ex=C[min(i+960,n-1)]
        R.append(dict(d=pd.Timestamp(s),dr=dr,hi=high,cts=cts,rk=rk,tgt=abs(tg-ep),pts=(ex-ep)*dr-.5,be=be)); break
T=pd.DataFrame(R); T=T[(T.d>='2018-02-01')]; T['era']=np.where(T.d<'2022-01-01','A','B'); T['usd']=T.pts*T.cts*20; T.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/zh158.pkl')
def s_(g):
    nn=len(g)
    if not nn: return 'n   0'
    pf=g.usd[g.usd>0].sum()/max(-g.usd[g.usd<0].sum(),1e-9); cu=g.usd.cumsum(); dd=(cu.cummax()-cu).max()
    return f'n{nn:4d} {g.pts.mean():+6.2f}pt t{(g.pts.mean()/(g.pts.std()/np.sqrt(nn)) if nn>2 else 0):+5.1f} w{(g.pts>0).mean()*100:3.0f}% PF{pf:4.2f} ${g.usd.sum():+8.0f} dd${dd:6.0f}'
print('ZH ASIA V2 (pts per contract, $ with 2-3 cts)'); print('ALL ',s_(T[T.era=="A"]),'|',s_(T[T.era=="B"]))
for nm,m in (('HIGH 3ct',T.hi),('MED 2ct',~T.hi),('LONG',T.dr>0),('SHORT',T.dr<0)): print(f'{nm:9s}',s_(T[(T.era=="A")&m]),'|',s_(T[(T.era=="B")&m]))
print(T.assign(y=T.d.dt.year).groupby('y').agg(n=('pts','size'),pts=('pts','sum'),usd=('usd','sum')).round(0).T.to_string()); print('median risk',T.rk.median(),'median tgt',T.tgt.median(),'trades/yr',len(T)/8.5)
