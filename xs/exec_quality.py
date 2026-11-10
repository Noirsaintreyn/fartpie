"""Nov 10 2026: EXECUTION QUALITY on the noise engine. DECLARED BEFORE RESULTS.
Baseline = market at next 5m open (nz_core.noise). Variants: limit at next_open -/+ P pts (P=2,4,8,12), live N=6 bars (30m), fill only if price trades THROUGH the limit by 1 tick (0.25), fill at limit (open if gapped through).
Unfilled = trade skipped (adverse selection included). Cost 0.5 (market) vs 0.25 (limit, maker-ish) and 0.5 both. Stop 40. Reports pts/trade, total pts, fill rate, 2014-21 vs 2022-26. ADOPT only if total pts per period beats market in BOTH."""
import sys; sys.path.insert(0,'/home/claude/fartpie/xs')
import numpy as np, pandas as pd, nz_core as Z
from nz_core import A, OPEN, PC, sg, GRID, NB
STOP=40.0
def run(d, P=0.0, cost=0.5, N=6, tick=0.25):
    a,_=A[d]; o,h,l,c,v,vw=a; s=sg(d); op=OPEN[d]; pc=PC[d] if np.isfinite(PC[d]) else op; rows=[]; pos=0; e=em=None; worst=0.0; pend=None; ex=0
    for j in range(NB):
        if np.isnan(c[j]): continue
        if pend is not None and pos==0:
            dr,lim,exp=pend
            if j>exp: pend=None
            else:
                if P==0: fill=o[j]
                elif dr>0 and l[j]<=lim-tick: fill=min(lim,o[j])
                elif dr<0 and h[j]>=lim+tick: fill=max(lim,o[j])
                else: fill=None
                if fill is not None: pos,e,em,worst=dr,fill,j,0.0; pend=None
        if pos!=0:
            lo=(l[j]-e) if pos>0 else (e-h[j]); worst=min(worst,lo)
            if lo<=-STOP: rows.append((d,em,pos,-STOP-cost,worst-cost)); pos=0; continue
        if j==NB-1 or np.isnan(o[j+1]):
            if j==NB-1 and pos!=0: rows.append((d,em,pos,(c[j]-e)*pos-cost,worst-cost)); pos=0
            continue
        t=GRID[j]
        if (t+5)%30!=0 or t+5<600 or t+5>930 or not np.isfinite(s[j]): continue
        ub,lb=max(op,pc)*(1+s[j]),min(op,pc)*(1-s[j]); nx=o[j+1]
        if pos==1 and c[j]<max(ub,vw[j]): rows.append((d,em,1,nx-e-cost,worst-cost)); pos=0
        elif pos==-1 and c[j]>min(lb,vw[j]): rows.append((d,em,-1,e-nx-cost,worst-cost)); pos=0
        if pos==0 and pend is None:
            if c[j]>ub: pend=(1,nx-P,j+N)
            elif c[j]<lb: pend=(-1,nx+P,j+N)
            if pend is not None and P==0: pend=(pend[0],nx,j+1)
    return rows
days=[d for d in Z.DAYS if d>=pd.Timestamp('2014-02-01')]
def go(P,cost):
    R=[r for d in days for r in run(d,P,cost)]; D=pd.DataFrame(R,columns=['day','m','dir','pts','mae']); return D
base=go(0,0.5); nb={'a':(base.day<'2022-01-01').sum(),'b':(base.day>='2022-01-01').sum()}
def line(lab,D):
    a=D[D.day<'2022-01-01']; b=D[D.day>='2022-01-01']
    print(f' {lab:18} n {len(a):4}/{len(b):4}  pts/trade {a.pts.mean():+6.2f}/{b.pts.mean():+6.2f}  TOTAL {a.pts.sum():+8.0f}/{b.pts.sum():+8.0f}  fill {len(a)/nb["a"]:.0%}/{len(b)/nb["b"]:.0%}')
line('market c0.5',base); line('market c0.25',go(0,0.25))
for P in (2,4,8,12):
    line(f'limit -{P} c0.5',go(P,0.5)); line(f'limit -{P} c0.25',go(P,0.25))
