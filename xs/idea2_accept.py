"""Nov 10 2026 IDEA 2: ACCEPTANCE vs REJECTION after a noise-band break. DECLARED BEFORE RESULTS.
Re-run the noise engine (market entry next open, stop 40, cost .5) recording entry bar, exit bar/price, band edge at entry.
At entry+30m (6 bars) for trades still open: A1 = close beyond the band edge in trade direction (price accepted outside);
A2 = POC of the 30m volume profile (typical price x volume, 2.5pt bins) beyond the band edge. REMAINING pts = exit - open of next bar after the mark.
Question: do accepted trades have higher remaining pts than rejected? Rule 'cut rejected at +30m' gain = -remaining of rejected (+cost). PASS: acc-rej same sign, |t|>=2 both periods."""
import sys; sys.path.insert(0,'/home/claude/fartpie/xs')
import numpy as np, pandas as pd, nz_core as Z
from nz_core import A, OPEN, PC, sg, GRID, NB
STOP=40.0; COST=0.5
def run(d):
    a,_=A[d]; o,h,l,c,v,vw=a; s=sg(d); op=OPEN[d]; pc=PC[d] if np.isfinite(PC[d]) else op; R=[]; pos=0; pend=None; e=ej=None; edge=None
    for j in range(NB):
        if np.isnan(c[j]): continue
        if pend is not None: pos,e,ej,edge,pend=pend[0],o[j],j,pend[1],None
        if pos!=0:
            lo=(l[j]-e) if pos>0 else (e-h[j])
            if lo<=-STOP: R.append((d,ej,j,pos,e,e-pos*STOP,edge,'stop')); pos=0; continue
        if j==NB-1 or np.isnan(o[j+1]):
            if j==NB-1 and pos!=0: R.append((d,ej,j,pos,e,c[j],edge,'eod')); pos=0
            continue
        t=GRID[j]
        if (t+5)%30!=0 or t+5<600 or t+5>930 or not np.isfinite(s[j]): continue
        ub,lb=max(op,pc)*(1+s[j]),min(op,pc)*(1-s[j]); nx=o[j+1]
        if pos==1 and c[j]<max(ub,vw[j]): R.append((d,ej,j,1,e,nx,edge,'sig')); pos=0
        elif pos==-1 and c[j]>min(lb,vw[j]): R.append((d,ej,j,-1,e,nx,edge,'sig')); pos=0
        if pos==0 and c[j]>ub: pend=(1,ub)
        elif pos==0 and c[j]<lb: pend=(-1,lb)
    return R
rows=[r for d in Z.DAYS for r in run(d)]
T=pd.DataFrame(rows,columns=['day','ej','xj','pos','e','xp','edge','why']); T['pts']=(T.xp-T.e)*T.pos-COST
out=[]
for r in T.itertuples():
    m=r.ej+5
    if r.xj<=m: continue                      # exited before the +30m mark
    a=A[r.day][0]; o,h,l,c,v=a[0],a[1],a[2],a[3],a[4]; sl=slice(r.ej,m+1)
    if np.isnan(c[m]) or m+1>=NB or np.isnan(o[m+1]): continue
    tp=(h[sl]+l[sl]+c[sl])/3; vol=np.nan_to_num(v[sl]); b=np.round(tp/2.5)*2.5
    poc=pd.Series(vol).groupby(b).sum().idxmax() if vol.sum()>0 else np.nan
    A1=(c[m]-r.edge)*r.pos>0; A2=(poc-r.edge)*r.pos>0 if np.isfinite(poc) else False
    mfe=max(((h[sl].max()-r.e) if r.pos>0 else (r.e-l[sl].min())),0)
    out.append((r.day,A1,A2,mfe,(r.xp-o[m+1])*r.pos,(c[m]-r.e)*r.pos,r.pts))
D=pd.DataFrame(out,columns=['day','A1','A2','mfe','rem','run','pts']); D=D[D.day>='2018-02-01']
def tt(a,b): return (a.mean()-b.mean())/np.sqrt(a.var()/len(a)+b.var()/len(b)) if len(a)>=10 and len(b)>=10 else np.nan
print(f'trades open at +30m: {len(D)} of {len(T)}')
for k in ('A1','A2'):
    o=[]
    for lab,m in (('18-21',D.day.dt.year<2022),('22-26',D.day.dt.year>=2022)):
        x=D[m]; ac=x.rem[x[k]]; rj=x.rem[~x[k]]
        o.append(f'{lab}: acc rem {ac.mean():+6.1f} (n{len(ac)}) rej rem {rj.mean():+6.1f} (n{len(rj)}) t {tt(ac,rj):+.2f} | cut-rej gain/trade-all {-rj.sum()/len(x):+.2f}')
    print(f' {k}',' || '.join(o))
