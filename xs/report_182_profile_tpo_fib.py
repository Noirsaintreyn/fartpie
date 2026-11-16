"""Report 182: volume-profile / TPO / ledge / volume-gap / fib level reactions on NQ RTH (Rey, Nov 15). DECLARED BEFORE RESULTS.
Could not read the Notion page (JS-rendered) or the YouTube videos (fetch fails) -> standard textbook definitions, stated here so they are fixed before results:
bin = 0.03% of prior close; volume spread uniformly over bins in each 5m bar's high-low; TPO = 30-min periods (6 bars) covering each bin.
Levels from the PRIOR session (1) or the PRIOR 5 sessions composite (5): POC, VAH/VAL (70% value area), HVN (smoothed peaks, prominence>=15% of max), LVN (smoothed troughs, same prominence),
GAP (run of >=2 bins with smoothed volume <=15% of max = volume gap, both edges), SHELF/ledge (boundary where a plateau (3-bin mean >=40% of max) drops to a void side (next 3-bin mean <=40% of the plateau), both directions),
SP (TPO single-print run >= 4 bins in the prior day, both edges), FIB (0.382/0.5/0.618/0.705/0.79 retracements of the prior day's range measured from its extreme in the direction of its close),
FIB+X = a fib level within 0.1% of a prior-day POC/VAH/VAL or composite HVN/SHELF (confluence).
Event: per day and level type, the FIRST touch (5m bar range contains the level, from 09:35, approach side = sign(level - previous close)). Outcome: continuation = approach_dir*(close 30 min later - level) (positive = price goes through, negative = rejected), 60 min as well. Cost not subtracted (reported as raw pts; a trade needs |mean| > 0.5).
Strict split: 2018-21 | 2022+; PASS = same sign, |t|>=2, n>=30 both."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from scipy.signal import find_peaks
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').sort_values('dt').drop_duplicates('dt'); b['d']=b.dt.dt.normalize(); b['m']=b.dt.dt.hour*60+b.dt.dt.minute
b=b[(b.m>=570)&(b.m<=955)]; D={d:g for d,g in b.groupby('d') if len(g)>=70}; days=sorted(D)
A={d:(g.open.values,g.high.values,g.low.values,g.close.values,g.volume.values.astype(float),((g.m.values-570)//30)) for d,g in D.items()}
def prof(ds,bs):
    lo=int(min(A[d][2].min() for d in ds)/bs)-1; hi=int(max(A[d][1].max() for d in ds)/bs)+2; n=hi-lo; V=np.zeros(n); T=np.zeros(n)
    for d in ds:
        o,h,l,c,v,p=A[d]
        for k in range(len(h)):
            i0=int(l[k]/bs)-lo; i1=int(h[k]/bs)-lo; V[i0:i1+1]+=v[k]/(i1-i0+1)
        for q in np.unique(p):
            s=p==q; i0=int(l[s].min()/bs)-lo; i1=int(h[s].max()/bs)-lo; T[i0:i1+1]+=1
    return lo,V,T
def va(V):
    j=int(V.argmax()); lo=hi=j; tot=V.sum(); cur=V[j]
    while cur<.7*tot and (lo>0 or hi<len(V)-1):
        a=V[lo-1]+(V[lo-2] if lo>1 else 0) if lo>0 else -1; c=V[hi+1]+(V[hi+2] if hi<len(V)-2 else 0) if hi<len(V)-1 else -1
        if a>=c: lo-=1; cur+=V[lo]
        else: hi+=1; cur+=V[hi]
    return j,lo,hi
def levels(d_idx):
    pd_=days[d_idx-1]; bs=A[pd_][3][-1]*0.0003; L={}
    lo1,V1,T1=prof([pd_],bs); lo5,V5,T5=prof(days[d_idx-5:d_idx],bs)
    px=lambda lo,i:(lo+i+.5)*bs; edge=lambda lo,i:(lo+i)*bs
    j,l_,h_=va(V1); L['POC1']=[px(lo1,j)]; L['VAH1']=[edge(lo1,h_+1)]; L['VAL1']=[edge(lo1,l_)]
    j,l_,h_=va(V5); L['POC5']=[px(lo5,j)]; L['VAH5']=[edge(lo5,h_+1)]; L['VAL5']=[edge(lo5,l_)]
    sm=np.convolve(V5,[.25,.5,.25],'same'); mx=sm.max()
    pk=find_peaks(sm,prominence=.15*mx,distance=5)[0]; L['HVN5']=[px(lo5,i) for i in pk]
    tr=find_peaks(-sm,prominence=.15*mx,distance=5)[0]; L['LVN5']=[px(lo5,i) for i in tr]
    nz=np.where(V5>0)[0]; a0,a1=nz.min(),nz.max(); void=(sm<=.15*mx); gp=[]; i=a0
    while i<=a1:
        if void[i]:
            k=i
            while k<=a1 and void[k]: k+=1
            if k-i>=2: gp+= [edge(lo5,i),edge(lo5,k)]
            i=k
        else: i+=1
    L['GAP5']=gp; sh=[]
    for i in range(3,len(sm)-4):
        left=sm[i-2:i+1].mean(); right=sm[i+1:i+4].mean()
        if left>=.4*mx and right<=.4*left and sm[i]>=.4*mx: sh.append(edge(lo5,i+1))
        if right>=.4*mx and left<=.4*right and sm[i+1]>=.4*mx: sh.append(edge(lo5,i+1))
    L['SHELF5']=sh; sp=[]; s1=(T1==1)&(V1>0); i=0
    while i<len(s1):
        if s1[i]:
            k=i
            while k<len(s1) and s1[k]: k+=1
            if k-i>=4: sp+=[edge(lo1,i),edge(lo1,k)]
            i=k
        else: i+=1
    L['SP1']=sp; o,h,l,c,v,p=A[pd_]; H,Lw=h.max(),l.min(); up=c[-1]>o[0]
    for f in (.382,.5,.618,.705,.79): L[f'FIB{f}']=[H-f*(H-Lw)] if up else [Lw+f*(H-Lw)]
    base=L['POC1']+L['VAH1']+L['VAL1']+L['HVN5']+L['SHELF5']; fb=[x for f in (.382,.5,.618,.705,.79) for x in L[f'FIB{f}']]
    L['FIBxVOL']=[x for x in fb if any(abs(x-y)<=.001*x for y in base)]
    return L
rows=[]
for di in range(6,len(days)):
    d=days[di]
    if d<pd.Timestamp('2018-01-01') or d>pd.Timestamp('2026-01-30') or (d-days[di-5]).days>12: continue
    try: L=levels(di)
    except Exception as e: continue
    o,h,l,c,v,p=A[d]; n=len(h)
    for ty,lv in L.items():
        best=None
        for x in lv:
            for t in range(1,n-1):
                if l[t]<=x<=h[t]:
                    if c[t-1]!=x and (best is None or t<best[0]): best=(t,np.sign(x-c[t-1]),x)
                    break
        if best and best[0]<n-6:
            t,a,x=best; rows.append((d,ty,t,a*(c[t+6]-x),a*(c[min(t+12,n-1)]-x),a))
E=pd.DataFrame(rows,columns=['d','ty','t','c30','c60','a']); E.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/profile_events182.pkl')
def st(s): n=len(s); return (n,s.mean(),s.mean()/(s.std()/np.sqrt(n)) if n>2 and s.std()>0 else 0)
print(f'events {len(E)}; days {E.d.nunique()}')
print('type      | 2018-21 cont30 (t, n) | 2022+ cont30 (t, n) | 60min A | 60min B | tag')
for ty in E.ty.unique():
    r=[]
    for m in (E.d<'2022-01-01',E.d>='2022-01-01'):
        x=E[(E.ty==ty)&m]; r.append((st(x.c30),st(x.c60)))
    ok=lambda k:(np.sign(r[0][k][1])==np.sign(r[1][k][1]) and all(abs(q[k][2])>=2 and q[k][0]>=30 for q in r))
    tag=('PASS30 ' if ok(0) else '')+('PASS60' if ok(1) else '')
    print(f'{ty:8s} | {r[0][0][1]:+6.2f} (t{r[0][0][2]:+.1f},n{r[0][0][0]}) | {r[1][0][1]:+6.2f} (t{r[1][0][2]:+.1f},n{r[1][0][0]}) | {r[0][1][1]:+6.2f} (t{r[0][1][2]:+.1f}) | {r[1][1][1]:+6.2f} (t{r[1][1][2]:+.1f}) | {tag}')
