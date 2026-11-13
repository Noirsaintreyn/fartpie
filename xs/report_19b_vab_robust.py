"""VA-B robustness (Nov 13 2026). VA-B = open INSIDE prior-day value area (70% VA of RTH volume profile), first touch of VAH/VAL after 10:00 -> GO WITH the break (long at VAH / short at VAL). Found as the exact mirror of a losing fade (VA-F), so every number below is post-hoc: declared checks only, no tuning on the answer.
Checks: (a) strict fill: need price to trade 0.25 THROUGH the edge, entry = edge+0.25 (breakout buy-stop), slippage 1 and 2 pts; (b) by year; (c) long/short; (d) exit variants: target 0.10/0.15/0.25 ADR with stop at POC / at the VA midpoint / fixed 25 pts; (e) touch before vs after 10:30; (f) day-of-week. Eras A 2018-02..2021-12 / B 2022-01..2026-01, NQ points, cost 0.5 + slippage."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
exec(open('xs/report_19_amt_papers.py').read().split("def sim(")[0].split('"""',2)[2])   # loads data, RTH, PRO
def sim(h,l,c,f,ent,stop,dr,xend,tgt=None,cost=.5):
    for q in range(f,xend+1):
        if (l[q]<=stop) if dr>0 else (h[q]>=stop): return -abs(ent-stop)-cost
        if tgt is not None and ((h[q]>=tgt) if dr>0 else (l[q]<=tgt)): return abs(tgt-ent)-cost
    return (c[xend]-ent)*dr-cost
dl=[d for d in days if d in RTH]; R=[]
for i,D in enumerate(dl):
    if i<21 or D<pd.Timestamp('2018-02-01') or D>pd.Timestamp('2026-01-30'): continue
    P=dl[i-1]
    if (D-P).days>5: continue
    g=RTH[D]; o,h,l,c=[g[k].values.astype(float) for k in ('open','high','low','close')]; adr=np.mean([RTH[d].high.max()-RTH[d].low.min() for d in dl[i-20:i]]); pO,pvah,pval=PRO[P]
    if not(pval<=o[0]<=pvah): continue
    for k in range(6,76):
        up=h[k]>=pvah-.25; dn=l[k]<=pval+.25
        if up or dn:
            dr=1 if up else -1; lvl=pvah if up else pval; strict=(h[k]>=lvl+.25) if up else (l[k]<=lvl-.25)
            base=dict(d=D,dr=dr,k=k,dow=D.dayofweek,risk=abs(lvl-pO),adr=adr,strict=strict)
            d2=min(.15*adr,40)
            if abs(lvl-pO)>=6:
                R.append(dict(base,v='base (edge fill, 1pt slip)',pts=sim(h,l,c,k,lvl,pO,dr,77,lvl+dr*d2,.5)-1.0))
                if strict:
                    for sl_ in (1.0,2.0): R.append(dict(base,v='strict buy-stop +0.25, slip %d'%sl_,pts=sim(h,l,c,k,lvl+dr*.25,pO,dr,77,lvl+dr*d2,.5)-sl_))
                    for tg in (.10,.25): R.append(dict(base,v='strict, target %.2f ADR, stop POC'%tg,pts=sim(h,l,c,k,lvl+dr*.25,pO,dr,77,lvl+dr*min(tg*adr,60),.5)-1.0))
                    mid=(pvah+pval)/2; R.append(dict(base,v='strict, stop VA-mid, target .15',pts=sim(h,l,c,k,lvl+dr*.25,mid,dr,77,lvl+dr*d2,.5)-1.0 if abs(lvl-mid)>=6 else np.nan))
                    R.append(dict(base,v='strict, stop 25 fixed, target .15',pts=sim(h,l,c,k,lvl+dr*.25,lvl+dr*.25-dr*25,dr,77,lvl+dr*d2,.5)-1.0))
                    R.append(dict(base,v='strict, no target (hold 15:55), stop POC',pts=sim(h,l,c,k,lvl+dr*.25,pO,dr,77,None,.5)-1.0))
            break
T=pd.DataFrame(R).dropna(subset=['pts']); T['era']=np.where(T.d<pd.Timestamp('2022-01-01'),'A','B'); T['yr']=T.d.dt.year
def t_(a): return a.mean()/(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>5 and a.std()>0 else np.nan
def line(nm,g):
    a,b_=g[g.era=='A'].pts,g[g.era=='B'].pts
    if len(a)<8 or len(b_)<8: return '%-44s n%d/%d'%(nm,len(a),len(b_))
    return '%-44s A n %3d mean %+6.2f win %2.0f%% t %+5.1f | B n %3d mean %+6.2f win %2.0f%% t %+5.1f | yrs>0 %3.0f%%'%(nm,len(a),a.mean(),(a>0).mean()*100,t_(a),len(b_),b_.mean(),(b_>0).mean()*100,t_(b_),(g.groupby('yr').pts.mean()>0).mean()*100)
for v,g in T.groupby('v',sort=False): print(line(v,g))
B=T[T.v=='base (edge fill, 1pt slip)']; S=T[T.v=='strict buy-stop +0.25, slip 1']
print('\n-- base VA-B by year (mean pts, n)'); print('  '+' | '.join('%d: %+.1f (%d)'%(y,g.pts.mean(),len(g)) for y,g in B.groupby('yr')))
print('-- base VA-B long/short'); [print(line('  '+n,B[B.dr==d_])) for d_,n in ((1,'long (VAH break)'),(-1,'short (VAL break)'))]
print('-- touch time'); [print(line('  '+n,B[f])) for n,f in (('touch before 10:30',B.k<12),('touch 10:30-12:00',(B.k>=12)&(B.k<30)),('touch after 12:00',B.k>=30))]
print('-- risk (POC distance) terciles'); q=B.risk.quantile([.33,.67]).values
for n,f in (('POC close (<%.0f)'%q[0],B.risk<q[0]),('mid',(B.risk>=q[0])&(B.risk<q[1])),('POC far (>%.0f)'%q[1],B.risk>=q[1])): print(line('  '+n,B[f]))
print('-- avg win / avg loss (base): win %+.1f  loss %+.1f  | share of days with a VA touch among inside-open days: see n'%(B[B.pts>0].pts.mean(),B[B.pts<=0].pts.mean()))
T.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/vab_T.pkl')
