"""report 161: S1 premium fade as a REGIME model (Rey: it turns on right after 2020). DECLARED BEFORE RESULTS.
Part 1 REGIME EVIDENCE from measures that do NOT use S1: for every day with |gap|>=5 pts, GF = -sign(gap)*(close-open)/ATR14 (open->close fade of the opening gap in ATR units), plus gap-fill rate and 09:30-10:00 reversal of the gap. Yearly means, and A vs B Welch t. If the same break date shows up in independent measures it is a regime, not an S1 accident.
Part 2 CAUSAL REGIME SWITCH: reg60 = trailing 60-trading-day mean of GF (shifted 1 day, uses every gap day, not S1 trades). Switch S1 on when reg60 > 0 (cut fixed a priori at zero). Report S1 ON vs OFF per era.
Part 3 save S1 trades with MAE for the Lucid sim (report 162)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; BL=pd.Timestamp('2022-01-01'); COST=.5
exec(open('xs/report_156_va_tightness.py').read().split("A=dd[dd.index<BL]")[0].split('"""',2)[2])
A_t=dd[dd.index<BL].t.dropna(); q50=A_t.quantile(.5)
rows=[]
for i,D in enumerate(days):
    if i<15: continue
    x=by[D]; pc=by[days[i-1]].close.iloc[-1]; o=x.open.iloc[0]; gap=o-pc; atr=dd.atr[days[i-1]]
    if x.m.iloc[0]!=570 or x.m.iloc[-1]!=955 or abs(gap)<5 or not atr>0: continue
    c=x.close.iloc[-1]; sg=np.sign(gap); x30=x[x.m<=595]; c30=x30.close.iloc[-1]
    filled=(x.high.max()>=pc) if gap<0 else (x.low.min()<=pc)
    rows.append(dict(d=pd.Timestamp(D),GF=-sg*(c-o)/atr,fill=float(filled),rev30=-sg*(c30-o)/atr))
G=pd.DataFrame(rows).set_index('d'); G=G[G.index>='2018-02-01']
print('PART 1 yearly regime measures (|gap|>=5 days):'); print(G.groupby(G.index.year).agg(n=('GF','size'),GF=('GF','mean'),fill=('fill','mean'),rev30=('rev30','mean')).round(3).T.to_string())
from scipy import stats
for c in ('GF','rev30','fill'):
    a,b=G[G.index<BL][c],G[G.index>=BL][c]; t,p=stats.ttest_ind(b,a,equal_var=False); print(f'{c:6s} A {a.mean():+.3f} -> B {b.mean():+.3f}  Welch t {t:+.2f} p {p:.3f}')
reg=G.GF.rolling(60,min_periods=30).mean().shift(1)
# S1 trades with MAE
def s1sim(D,i):
    p=P[days[i-1]]; atr=dd.atr[days[i-1]]; x=by[D].reset_index(drop=True); o=x.open[0]
    if x.m.iloc[0]!=570 or x.m.iloc[-1]!=955 or not atr>0 or not (p['t']<=q50 and o>p['vah']): return None
    ent=o; st=ent+.5*atr; tgt=p['val']; lock=False; worst=0.0; poc=p['poc']
    for q in range(len(x)):
        hq,lq=x.high[q],x.low[q]; worst=max(worst,hq-ent)
        if hq>=st: return dict(pts=-(st-ent)-COST,mae=-(st-ent)-COST,risk=st-ent)
        if lq<=tgt: return dict(pts=(ent-tgt)-COST,mae=-worst-COST,risk=st-ent)
        if not lock and lq<=poc<=hq: lock=True; st=ent-.25*atr
    return dict(pts=(ent-x.close.iloc[-1])-COST,mae=-worst-COST,risk=.5*atr)
R=[]
for i,D in enumerate(days):
    if i<15 or D<pd.Timestamp('2018-02-01') or D>pd.Timestamp('2026-01-30'): continue
    r=s1sim(D,i)
    if r: r.update(day=pd.Timestamp(D),m=570.0,dir=-1.0,era='A' if D<BL else 'B'); R.append(r)
X=pd.DataFrame(R); X['reg']=reg.reindex(X.day).values; X.to_pickle(S+'s1_trades162.pkl')
def st(s):
    n=len(s); return f'n{n:4d} {s.mean():+7.1f} t{(s.mean()/(s.std()/np.sqrt(n)) if n>2 and s.std()>0 else 0):+5.1f} w{(s>0).mean()*100:3.0f}%' if n else 'n   0'
print('\nPART 2 S1 base A',st(X[X.era=='A'].pts),'| B',st(X[X.era=='B'].pts),'| median risk',round(X.risk.median(),0))
on=X.reg>0
for e in 'AB': print(f'  reg60>0 ON  {e} {st(X[(X.era==e)&on].pts)}   OFF {st(X[(X.era==e)&~on&X.reg.notna()].pts)}')
print('  ON share of S1 days:',round(on.mean(),2),' yearly ON share:',X.groupby(X.day.dt.year).apply(lambda g:(g.reg>0).mean()).round(2).to_dict())
