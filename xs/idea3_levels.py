"""Nov 10 2026 IDEA 3: prior-day levels (H, L, POC, VAH, VAL) as exit magnets on the noise engine. DECLARED BEFORE RESULTS.
Baseline engine (market entry next open, stop 40, cost .5). Variant: take profit at the nearest prior-day level >= M pts beyond entry (M=10,20,40), fill at level if price trades through by 1 tick; stop first if both in one bar.
Null: same levels shifted by a random +-(30..80) pts per day, 30 draws. Adopt only if TP total beats baseline in BOTH periods AND the real levels beat the null's 95th pct."""
import sys; sys.path.insert(0,'/home/claude/fartpie/xs')
import numpy as np, pandas as pd, nz_core as Z
from nz_core import A, OPEN, PC, sg, GRID, NB
STOP=40.0; COST=0.5; rng=np.random.default_rng(7)
def prof(d):
    a=A[d][0]; h,l,c,v=a[1],a[2],a[3],np.nan_to_num(a[4]); ok=~np.isnan(c)
    tp=(h[ok]+l[ok]+c[ok])/3; b=np.round(tp/2.5)*2.5; s=pd.Series(v[ok]).groupby(b).sum()
    if s.sum()<=0: return None
    poc=s.idxmax(); order=s.sort_index(); tot=order.sum(); lo=hi=poc; acc=order[poc]; idx=list(order.index); i0=idx.index(poc); i,j=i0,i0
    while acc<0.7*tot and (i>0 or j<len(idx)-1):
        up=order.iloc[j+1] if j<len(idx)-1 else -1; dn=order.iloc[i-1] if i>0 else -1
        if up>=dn: j+=1; acc+=up
        else: i-=1; acc+=dn
    return [float(np.nanmax(h[ok])),float(np.nanmin(l[ok])),float(poc),float(idx[j]),float(idx[i])]
days=Z.DAYS; LV={}
for k,d in enumerate(Z.ALLD):
    if d in set(days) or True:
        p=prof(d)
        if p: LV[d]=p
prev={d:Z.ALLD[Z.ALLD.index(d)-1] for d in days if Z.ALLD.index(d)>0}
def run(d,M=None,shift=0.0):
    a,_=A[d]; o,h,l,c,v,vw=a; s=sg(d); op=OPEN[d]; pc=PC[d] if np.isfinite(PC[d]) else op; R=[]; pos=0; pend=None; e=None; tp=None
    lv=np.array(LV.get(prev.get(d),[]))+shift
    for j in range(NB):
        if np.isnan(c[j]): continue
        if pend is not None:
            pos,e=pend,o[j]; pend=None; tp=None
            if M is not None and len(lv):
                cand=lv[(lv-e)*pos>=M]
                if len(cand): tp=cand[np.argmin(np.abs(cand-e))]
        if pos!=0:
            lo=(l[j]-e) if pos>0 else (e-h[j])
            if lo<=-STOP: R.append(-STOP-COST); pos=0; continue
            if tp is not None and ((pos>0 and h[j]>=tp+.25) or (pos<0 and l[j]<=tp-.25)): R.append((tp-e)*pos-COST); pos=0; continue
        if j==NB-1 or np.isnan(o[j+1]):
            if j==NB-1 and pos!=0: R.append((c[j]-e)*pos-COST); pos=0
            continue
        t=GRID[j]
        if (t+5)%30!=0 or t+5<600 or t+5>930 or not np.isfinite(s[j]): continue
        ub,lb=max(op,pc)*(1+s[j]),min(op,pc)*(1-s[j]); nx=o[j+1]
        if pos==1 and c[j]<max(ub,vw[j]): R.append(nx-e-COST+0) if False else R.append((nx-e)*1-COST); pos=0
        elif pos==-1 and c[j]>min(lb,vw[j]): R.append((e-nx)-COST); pos=0
        if pos==0 and c[j]>ub: pend=1
        elif pos==0 and c[j]<lb: pend=-1
    return R
def tot(M=None,shift=None):
    r={'a':0.0,'b':0.0}
    for d in days:
        sh=0.0 if shift is None else shift*rng.integers(30,81)*rng.choice([-1,1])
        x=sum(run(d,M,sh)); r['a' if d<pd.Timestamp('2022-01-01') else 'b']+=x
    return r
base=tot(); print('baseline total pts 2014-21 / 2022-26:',round(base['a']),round(base['b']))
for M in (10,20,40):
    r=tot(M); nl=[tot(M,1.0) for _ in range(12)]; qa=np.percentile([x['a'] for x in nl],95); qb=np.percentile([x['b'] for x in nl],95)
    print(f' TP at nearest level >= {M:2}pts: {round(r["a"]):+6} / {round(r["b"]):+6}  vs base {round(r["a"]-base["a"]):+d}/{round(r["b"]-base["b"]):+d} | null p95 {round(qa)}/{round(qb)}')
