"""report 159: GATES for the S1 premium fade (report 156: tight prior VA + open above VAH -> short to VAL, stop .5 daily ATR). Rey: find the gate, don't toss it. DECLARED BEFORE RESULTS. Cuts frozen on 2018-21 trades; pass = same sign, |t|>=2, n>=30 in BOTH eras for the IN group.
Mechanisms (AMT): G1 INRANGE open <= prior-day high (open auction inside yesterday's range returns to value; open above PDH = initiative/acceptance). G2 FILL40 trailing 40-day share of gaps >=5 pts filled by close >= A median (mean-reversion regime). G3 NEARVAH (open-VAH)/ATR <= A median (small excursion above value reverts, large = breakout). G4 PDUP prior day closed up (premium after a rally = exhaustion?). G5 ABOVE20 open above 20-day range midpoint (premium in the larger auction). G6 LOWVOL prior-day range/ATR below A median. G7 GAPUP open above prior close (gap, not rotation). G8 TIGHT25 tightness <= A 25th pct (stricter VA).
Report each gate IN vs OUT, then the single best by pre-declared rule (both eras IN pass) with year table and top-10-trade concentration."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; BL=pd.Timestamp('2022-01-01')
exec(open('xs/report_156_va_tightness.py').read().split("A=dd[dd.index<BL]")[0].split('"""',2)[2])
A_t=dd[dd.index<BL].t.dropna(); q25,q50=A_t.quantile([.25,.5])
RTd=pd.DataFrame({d:dict(o=by[d].open.iloc[0],h=by[d].high.max(),l=by[d].low.min(),c=by[d].close.iloc[-1]) for d in days}).T; RTd.index=pd.to_datetime(RTd.index)
T=pd.read_pickle(S+'va156.pkl'); t=T[T.cfg=='S1_premium'].copy(); t['d']=pd.to_datetime(t.d)
prev=RTd.shift(1); gap=RTd.o-prev.c; filled=((gap<0)&(RTd.h>=prev.c))|((gap>0)&(RTd.l<=prev.c)); big=gap.abs()>=5
fr=pd.Series(np.where(big,filled.astype(float),np.nan),index=RTd.index).rolling(40,min_periods=15).mean().shift(1)
rows=[]
for _,r in t.iterrows():
    D=r.d; p=P[days[days.index(D)-1]] if D in days else None
    i=list(RTd.index).index(D); pdh=RTd.h.iloc[i-1]; pdc,pdo=RTd.c.iloc[i-1],RTd.o.iloc[i-1]; atr=dd.atr[RTd.index[i-1]]; o=RTd.o.iloc[i]
    lo20=RTd.l.iloc[i-20:i].min(); hi20=RTd.h.iloc[i-20:i].max()
    rows.append(dict(d=D,pts=r.pts,era=r.era,inrange=o<=pdh,fill=fr.iloc[i],near=(o-p['vah'])/atr,pdup=pdc>pdo,above20=o>(lo20+hi20)/2,pr=p['tr']/atr,gapup=o>pdc,t=p['t']))
X=pd.DataFrame(rows); fa=X[X.era=='A']
cuts=dict(fill=fa.fill.median(),near=fa.near.median(),pr=fa.pr.median())
G={'G1_INRANGE':X.inrange,'G2_FILL40':X.fill>=cuts['fill'],'G3_NEARVAH':X.near<=cuts['near'],'G4_PDUP':X.pdup,'G5_ABOVE20':X.above20,'G6_LOWVOL':X.pr<=cuts['pr'],'G7_GAPUP':X.gapup,'G8_TIGHT25':X.t<=q25}
def st(s):
    n=len(s); 
    if n==0: return 'n   0'
    return f'n{n:4d} {s.mean():+7.1f} t{(s.mean()/(s.std()/np.sqrt(n)) if n>2 and s.std()>0 else 0):+5.1f} w{(s>0).mean()*100:3.0f}%'
print('cuts',{k:round(v,2) for k,v in cuts.items()}); print('S1 base: A',st(X[X.era=='A'].pts),'| B',st(X[X.era=='B'].pts))
best=[]
for g,m in G.items():
    a1,b1=X[(X.era=='A')&m].pts,X[(X.era=='B')&m].pts; a0,b0=X[(X.era=='A')&~m].pts,X[(X.era=='B')&~m].pts
    ps=lambda s:(len(s)>=30 and abs(s.mean()/(s.std()/np.sqrt(len(s))))>=2)
    ok=ps(a1) and ps(b1) and np.sign(a1.mean())==np.sign(b1.mean())
    print(f'{g:11s} IN  A {st(a1)} | B {st(b1)} {"PASS" if ok else ""}\n{"":11s} OUT A {st(a0)} | B {st(b0)}')
    if ok: best.append(g)
print('PASSING GATES:',best)
for g in best:
    m=G[g]; Y=X[m]; print(f'\n{g} by year:'); print(Y.groupby(Y.d.dt.year).pts.agg(['size','mean']).round(1).T.to_string())
    for e in 'AB':
        y=Y[Y.era==e].pts.sort_values(ascending=False); print(f'era {e} top-5 trades share of total: {y.head(5).sum()/y.sum():.0%}; mean w/o top5 {y.iloc[5:].mean():+.1f}')
