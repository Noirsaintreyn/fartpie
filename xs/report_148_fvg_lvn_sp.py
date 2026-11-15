"""report 148 (declared before results, Rey: LVN / TPO single print alignment with ALL fair-value gaps, not only IFVG): plain 3-bar FVG signals (report 142 rules, every non-overlapping signal 09:35-14:00, 2R, 0.5 pt cost) tagged by prior-day LVN / single print / shelf / gamma wall within TOL x ADR of the gap (ict.py LEV levels). Prediction: LVN/SP-aligned FVG > others (fast-travel zones). [original 142 header follows] report 142: GENERIC fair-value-gap entry on NQ 5m (every RTH day, not only the sweep->MSS chain) + confluence tests. DECLARED BEFORE RESULTS.
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
    i930=int(np.searchsorted(msx,930)); ie=np.where(m==840)[0]; xe=np.where(m==955)[0]
    if not len(ie) or not len(xe): continue
    ie=int(ie[-1]); xend=int(xe[-1]); busy=-1
    for j in range(i930+3,ie+1):
        if j<=busy: continue
        if abs(c[j]-o[j])<.6*atr[j]: continue
        for dr in (1,-1):
            if not ((c[j]>o[j] and l[j]>h[j-2]) if dr>0 else (c[j]<o[j] and h[j]<l[j-2])): continue
            zlo,zhi=((h[j-2],l[j]) if dr>0 else (h[j],l[j-2])); ent=(zlo+zhi)/2; stop=(l[j-2]-2) if dr>0 else (h[j-2]+2); risk=abs(ent-stop)
            if risk<6 or risk>40: continue
            f=None
            for q in range(j+1,min(j+13,len(x))):
                if (l[q]<=ent-.25) if dr>0 else (h[q]>=ent+.25): f=q; break
            if f is None or f>ie+12: continue
            R.append(dict(d=D,j=j,dr=dr,ent=ent,zlo=zlo,zhi=zhi,risk=risk,pts=sim(h,l,c,f,ent,stop,dr,xend,ent+2*risk*dr))); busy=f; break
T=pd.DataFrame(R); T['A']=T.d<'2022-01-01'
_s=open('/home/claude/fartpie/xs/ict.py').read(); exec(_s[:_s.index('def zones_and_trades')])
adr={}
for k,dd in enumerate(days):
    if k>=14: adr[dd]=np.mean([RT.loc[days[q]].h-RT.loc[days[q]].l for q in range(k-14,k) if days[q] in RT.index])
import os; TOL=float(os.getenv('TOL','0.1'))
for nm in ('LVN','SP','SHELF','WALL'):
    T[nm]=[(d in LEV and d in adr and any((lv>=z0-TOL*adr[d]) and (lv<=z1+TOL*adr[d]) for lv in np.atleast_1d(LEV[d][nm]).astype(float))) for d,z0,z1 in zip(T.d,T.zlo,T.zhi)]
T['FAST']=T.LVN|T.SP
T.to_pickle(SC+'ext/fvg_all_vp.pkl')
def t_(a): return a.mean()/(a.std()/np.sqrt(len(a))) if len(a)>8 else np.nan
def line(lab,m,firstday=False):
    S_=T[m]; 
    if firstday: S_=S_.sort_values(['d','j']).groupby('d').head(1)
    a,b_=S_[S_.A].pts,S_[~S_.A].pts; ps=len(a)>=30 and len(b_)>=30 and np.sign(a.mean())==np.sign(b_.mean()) and abs(t_(a))>=2 and abs(t_(b_))>=2
    return f'  {lab:26s} A n{len(a):4d} {a.mean():+6.2f} win{100*(a>0).mean():3.0f}% t{t_(a):+5.1f} | B n{len(b_):4d} {b_.mean():+6.2f} win{100*(b_>0).mean():3.0f}% t{t_(b_):+5.1f}{"  <== PASS" if ps else ""}'
print(f'TOL {TOL} ADR; all plain-FVG signals (non-overlapping), {len(T)} trades on {T.d.nunique()} days')
print(line('ALL FVG',T.pts.notna()))
for c in ('LVN','SP','FAST','SHELF','WALL'): print(line(c+' aligned',T[c])); print(line(c+' NOT',~T[c]))
print('-- first aligned FAST signal per day (fills idle days?)'); print(line('FAST first/day',T.FAST,True)); print(line('LVN first/day',T.LVN,True)); print(line('SP first/day',T.SP,True))
print('-- longs / shorts for FAST aligned'); print(line('FAST long',T.FAST&(T.dr>0))); print(line('FAST short',T.FAST&(T.dr<0)))
