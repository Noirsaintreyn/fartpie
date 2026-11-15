"""report 142: GENERIC fair-value-gap entry on NQ 5m (every RTH day, not only the sweep->MSS chain) + confluence tests. DECLARED BEFORE RESULTS.
Rey: pair FVG with THOD/TLOD, OPTICS-clustered levels, Fibonacci; make FVG fill the days the core legs are idle.
EVENT: RTH 5m bar j (09:35-14:00) with body >= 0.6 ATR14 and a 3-bar gap (bull: low[j] > high[j-2]; bear: high[j] < low[j-2]). Entry = limit at gap midpoint within 12 bars (0.25 pt through).
Stop = far edge of bar j-2 minus/plus 2 pts (risk 6..40 pts else skip). Target 2R. Flat 15:55. Cost 0.5 pt. FIRST valid signal per day (so it can fill idle days). NQ points per trade, 1 contract.
Split: A 2018-02..2021-12 (cuts frozen here), B 2022-01..2026-01. PASS = same sign, |t|>=2 both eras, n>=30 each.
H1 THOD/TLOD: sigma_day = GK sigma of last 60 completed 1h bars (bot formula) x sqrt(6.5); THOD/TLOD = RTH open +/- 1 sigma_day. POS = (entry-open)/sigma_day x dir (positive = trading with the day's extension);
   ROOM = (THOD - entry)/risk for longs, (entry - TLOD)/risk for shorts. Prediction: low ROOM (target past the theoretical extreme) loses; entries near the opposite extreme (POS very negative = buying near TLOD) win.
H2 OPTICS: levels known at signal = PDH PDL PDC ONH ONL AsiaH AsiaL LonH LonL RTH open + 100-pt rounds within 400 pts; OPTICS(min_samples=3, xi=0.05, eps cap 0.15 sigma_day) -> zones. CONF = entry inside a zone (+/-2 pts); BLOCK = zones between entry and target.
   Prediction: CONF wins, BLOCK loses.
H3 Fibonacci: leg = swing extreme of the 10 bars before j-2 to the extreme of bar j; FIB = retrace depth of the entry. Prediction: OTE 0.62-0.79 best; shallow (<0.38) fails to fill / runs; deep (>0.79) fails.
IDLE: share of trades on days the core legs are idle (idle_state) and their mean."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.cluster import OPTICS
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; COST=.5
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').sort_values('dt').drop_duplicates('dt').reset_index(drop=True); b['d']=b.dt.dt.normalize(); b['m']=b.dt.dt.hour*60+b.dt.dt.minute
by=dict(tuple(b.groupby('d'))); days=sorted(by)
X1=b.set_index('dt')[['open','high','low','close']].resample('1h').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
RT=b[(b.m>=570)&(b.m<960)].groupby('d').agg(o=('open','first'),h=('high','max'),l=('low','min'),c=('close','last'))
def sig_h(t):
    k=X1.index.searchsorted(t,'left'); r=X1.iloc[max(0,k-60):k]
    if len(r)<40: return np.nan
    o,h,l,c=r.open.values,r.high.values,r.low.values,r.close.values; v=np.maximum(.5*np.log(h/l)**2-(2*np.log(2)-1)*np.log(c/o)**2,1e-9); return np.sqrt(v.mean())*c[-1]
def sim(h,l,c,f,ent,stop,dr,xend,tgt):
    for q in range(f,xend+1):
        if (l[q]<=stop) if dr>0 else (h[q]>=stop): return -abs(ent-stop)-COST
        if (h[q]>=tgt) if dr>0 else (l[q]<=tgt): return abs(tgt-ent)-COST
    return (c[xend]-ent)*dr-COST
R=[]
for i,D in enumerate(days):
    if i<21 or D<pd.Timestamp('2018-02-01') or D>pd.Timestamp('2026-01-30') or days[i-1] not in RT.index or D not in RT.index: continue
    p=by[days[i-1]]; p=p[p.m>=1080]; cD=by[D]; cD=cD[cD.m<960]
    if len(p)<30 or len(cD)<150: continue
    x=pd.concat([p,cD]).reset_index(drop=True); m=x.m.values; msx=np.where(m>=1080,m-1080,m+360)
    h,l,c,o=[x[k].values.astype(float) for k in ('high','low','close','open')]; tr5=np.maximum.reduce([h-l,np.abs(h-np.r_[c[0],c[:-1]]),np.abs(l-np.r_[c[0],c[:-1]])]); atr=pd.Series(tr5).rolling(14).mean().values
    i930=int(np.searchsorted(msx,930)); i03=int(np.searchsorted(msx,540)); i08=int(np.searchsorted(msx,840)); ie=np.where(m==840)[0]; xe=np.where(m==955)[0]
    if not len(ie) or not len(xe): continue
    ie=int(ie[-1]); xend=int(xe[-1]); pd_=RT.loc[days[i-1]]; op=o[i930]; sg=sig_h(D+pd.Timedelta(minutes=570))*np.sqrt(6.5)
    if not sg>0: continue
    lv={'PDH':pd_.h,'PDL':pd_.l,'PDC':pd_.c,'ONH':h[:i930].max(),'ONL':l[:i930].min(),'AH':h[:i03].max(),'AL':l[:i03].min(),'LH':h[i03:i08].max(),'LL':l[i03:i08].min(),'OPEN':op}
    base=np.array(list(lv.values())+[r for r in np.arange(np.floor((op-400)/100)*100,op+401,100)]); base=np.sort(base)
    zones=[]
    try:
        lab=OPTICS(min_samples=3,xi=.05,max_eps=.15*sg).fit(base.reshape(-1,1)).labels_
        for g in set(lab)-{-1}: v=base[lab==g]; zones.append((v.min(),v.max()))
    except Exception: pass
    THOD,TLOD=op+sg,op-sg; done=False
    for j in range(i930+3,ie+1):
        if done: break
        body=abs(c[j]-o[j])
        if body<.6*atr[j]: continue
        for dr in (1,-1):
            if not ((c[j]>o[j] and l[j]>h[j-2]) if dr>0 else (c[j]<o[j] and h[j]<l[j-2])): continue
            ent=((l[j]+h[j-2])/2) if dr>0 else ((h[j]+l[j-2])/2); stop=(l[j-2]-2) if dr>0 else (h[j-2]+2); risk=abs(ent-stop)
            if risk<6 or risk>40: continue
            f=None
            for q in range(j+1,min(j+13,len(x))):
                if (l[q]<=ent-.25) if dr>0 else (h[q]>=ent+.25): f=q; break
            if f is None or f>ie+12: continue
            tgt=ent+2*risk*dr; pts=sim(h,l,c,f,ent,stop,dr,xend,tgt)
            w=slice(max(j-12,0),j-1); ex0=(l[w].min() if dr>0 else h[w].max()); ex1=(h[j] if dr>0 else l[j]); leg=abs(ex1-ex0); fib=abs(ex1-ent)/leg if leg>0 else np.nan
            room=((THOD-ent) if dr>0 else (ent-TLOD))/risk; pos=(ent-op)/sg*dr
            conf=any(z[0]-2<=ent<=z[1]+2 for z in zones); lo,hi=sorted((ent,tgt)); block=sum(1 for z in zones if z[1]>lo+risk*.5 and z[0]<hi and not(z[0]-2<=ent<=z[1]+2))
            R.append(dict(d=D,dr=dr,m=int(m[f]),pts=pts,risk=risk,room=room,pos=pos,conf=conf,block=block,fib=fib,nlev=int(((np.abs(base-ent)<=.25*sg)).sum()),ozones=len(zones))); done=True; break
T=pd.DataFrame(R); T['A']=T.d<'2022-01-01'; idl=~pd.read_pickle(SC+'idle_state.pkl')['trade'].astype(bool); idl.index=pd.to_datetime(idl.index); T['idle']=T.d.map(idl).fillna(False).astype(bool)
T.to_pickle(SC+'ext/fvg_confluence.pkl')
def t_(a): return a.mean()/(a.std()/np.sqrt(len(a))) if len(a)>8 else np.nan
def line(lab,m):
    a,b_=T[m&T.A].pts,T[m&~T.A].pts; return f'  {lab:34s} A n{len(a):4d} {a.mean():+6.2f} win {100*(a>0).mean():3.0f}% t{t_(a):+5.1f} | B n{len(b_):4d} {b_.mean():+6.2f} win {100*(b_>0).mean():3.0f}% t{t_(b_):+5.1f}'
print(len(T),'trades (first FVG per day), days with a trade',T.d.nunique(),'of',len(days)); print(line('ALL',T.pts.notna()))
for nm,m in (('long',T.dr>0),('short',T.dr<0),('core legs idle',T.idle),('core legs active',~T.idle)): print(line(nm,m))
def terc(c):
    q=T[T.A][c].quantile([1/3,2/3]).values; return q
print('\nH1 THOD/TLOD (thirds cut on A)')
for c in ('room','pos'):
    q=terc(c)
    for nm,m in ((f'{c} low',T[c]<=q[0]),(f'{c} mid',(T[c]>q[0])&(T[c]<=q[1])),(f'{c} high',T[c]>q[1])): print(line(nm,m))
print('  room<1R (target past the theoretical extreme)',); print(line('room<2 (2R target beyond THOD/TLOD)',T.room<2)); print(line('room>=2',T.room>=2))
print('\nH2 OPTICS zones'); print(line('entry inside zone (CONF)',T.conf)); print(line('not in zone',~T.conf)); print(line('BLOCK=0',T.block==0)); print(line('BLOCK>=1',T.block>=1)); print(line('CONF & BLOCK=0',T.conf&(T.block==0))); print(line('nlev>=3 within .25 sigma',T.nlev>=3))
print('\nH3 Fibonacci retrace of the displacement leg')
for nm,m in (('<0.38',T.fib<.38),('0.38-0.62',(T.fib>=.38)&(T.fib<.62)),('OTE 0.62-0.79',(T.fib>=.62)&(T.fib<=.79)),('>0.79',T.fib>.79)): print(line('fib '+nm,m))
print('\ncombos (frozen cuts):'); q=terc('pos')
for nm,m in (('room>=2 & BLOCK=0',(T.room>=2)&(T.block==0)),('room>=2 & CONF',(T.room>=2)&T.conf),('OTE & room>=2',(T.fib>=.62)&(T.fib<=.79)&(T.room>=2)),('OTE & CONF',(T.fib>=.62)&(T.fib<=.79)&T.conf),('room>=2 & BLOCK=0 & idle',(T.room>=2)&(T.block==0)&T.idle)): print(line(nm,m))
