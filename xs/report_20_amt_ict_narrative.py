"""Nov 13 2026: ONE NARRATIVE = AMT context (where the day opens vs prior value area, overnight inventory) gating the ICT chain (sweep->MSS+FVG->limit, icc_T.pkl). DECLARED BEFORE RESULTS, 3 hypotheses, no tuning:
 H1 initiative: day opens OUTSIDE prior VA and chain direction = away from VA (open>VAH & long, open<VAL & short) - acceptance/continuation, the sweep was the stop-run before the move.
 H2 counter-auction: chain direction opposite overnight inventory (inv>=.9 -> short, <=.1 -> long) - 'about 70% counter-auction'.
 H3 balance: day opens INSIDE VA and chain direction points back toward the VA middle after sweeping an edge side (long if open in lower half of VA, short if upper half).
 Also: chain on same day/same direction as VA-B. Compare filtered vs unfiltered chain per era (NQ pts, 1 ct, cost already in icc_T)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('/home/claude/fartpie/xs/report_19_amt_papers.py').read().split("def sim(")[0].split('"""',2)[2])
C=pd.read_pickle(SC+'icc_T.pkl'); V=pd.read_pickle(SC+'amt_models_T.pkl'); VB=V[V.model=='VA-B']
vbd={(d,dr) for d,dr in zip(VB.d,VB.dr)}; vbdays={d for d,_ in vbd}
dl=[d for d in days if d in RTH]; ctx={}
for i,D in enumerate(dl):
    if i<21 or (D-dl[i-1]).days>5: continue
    P=dl[i-1]; pO,vah,val=PRO[P]; o=RTH[D].open.values[0]; settle=RTH[P].close.values[-1]
    on=by[P]; on=on[on.m>=1080]; on=pd.concat([on,by[D][by[D].m<570]])
    if len(on)<60: continue
    inv=float((on.close.values>settle).mean()); ctx[D]=(o,vah,val,inv)
rows=[]
for r in C.itertuples():
    if r.d not in ctx: continue
    o,vah,val,inv=ctx[r.d]; dr=r.dr
    h1=(o>vah and dr>0) or (o<val and dr<0)
    h2=(inv>=.9 and dr<0) or (inv<=.1 and dr>0)
    mid=(vah+val)/2; h3=(val<=o<=vah) and ((o<mid and dr>0) or (o>=mid and dr<0))
    vb=(r.d,dr) in vbd
    rows.append(dict(d=r.d,dr=dr,pts=r.pts,mae=r.mae,m=r.m,era=r.era,yr=r.yr,H1=h1,H2=h2,H3=h3,VB=vb,VBday=r.d in vbdays,out=not(val<=o<=vah)))
T=pd.DataFrame(rows); T.to_pickle(SC+'narr_T.pkl')
def st(x):
    if len(x)<5: return f'n={len(x):3d}'
    return f'n={len(x):3d} mean {x.pts.mean():+6.2f} win {np.mean(x.pts>0):.0%} t {x.pts.mean()/(x.pts.std()/len(x)**.5+1e-9):+.1f}'
print('ALL chain', ' A:',st(T[T.era=='A']),' | B:',st(T[T.era=='B']))
for k in ['H1','H2','H3','VB','VBday','out']:
    for v in (True,False):
        s=T[T[k]==v]; print(f'{k}={v!s:5}  A: {st(s[s.era=="A"])}  | B: {st(s[s.era=="B"])}')
s=T[T.H1|T.H3]; print('H1|H3 (AMT-aligned)  A:',st(s[s.era=='A']),'| B:',st(s[s.era=='B']))
s=T[~(T.H1|T.H3)]; print('not aligned          A:',st(s[s.era=='A']),'| B:',st(s[s.era=='B']))
s=T[(T.H1|T.H3)&T.H2]; print('aligned & counter-inv A:',st(s[s.era=='A']),'| B:',st(s[s.era=='B']))
