"""report_122 (Nov 15 2026): REAL WYCKOFF, ALONE AND WITH THE VECTOR, small caps ($1-5, dv20>=$500k). Definitions fixed BEFORE results. Train 2019-21 / test 2022+, 30bp, entry next open.
Structure at bar i (all known at close i): RANGE = prior 40 bars (excl. i), width <= 12 ATR14. SPRING = in the last 25 bars a bar whose low pierced the prior-20-bar low and CLOSED back above it. DRY = mean volume of last 10 bars <= 0.8 x mean volume of the 40-bar range (supply exhausted). SOS = close i > range high on volume >= 1.5 x 20-bar mean.
WYCKOFF EVENT = RANGE & SPRING & DRY & SOS. Controls: SOS alone (plain volume breakout of a 40d range), SOS&SPRING, SOS&DRY. 
Part A (Wyckoff ALONE): event -> hold 20 bars, excess over the average $1-5 stock-day that month (same 20d window). Split by vector state at bar i: FRESH flip (flip within 5 bars), BULL older, BEAR/none.
Part B (WITH vector): the 30,323 daily flips, hold to bear flip; flips whose flip bar IS a Wyckoff SOS after spring+dry vs the rest. Pass bar: same sign both eras, month-clustered t>=2 on the test, n>=30 per era."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
X='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; D=pd.read_pickle(X+'wide_daily_ohlcv.pkl'); F=pd.read_pickle(X+'stack_flips.pkl'); COST=.003; H=20
rows=[]; allx=[]
for tk,q in D.items():
    if len(q)<300: continue
    q=q[['open','high','low','close','volume']].copy(); q=q[~q.index.duplicated()].dropna(); q.index=pd.to_datetime(q.index).tz_localize(None).normalize()
    if len(q)<300: continue
    reg,lev,atr,_=run_state_machine(q.assign(datetime=q.index)); reg=np.asarray(reg)
    o,h,l,c,v=[q[k].values.astype(float) for k in ('open','high','low','close','volume')]; n=len(q)
    pc=np.r_[c[0],c[:-1]]; tr=np.maximum(h-l,np.maximum(abs(h-pc),abs(l-pc))); a14=pd.Series(tr).rolling(14).mean().shift(1).values
    RH=pd.Series(h).rolling(40).max().shift(1).values; RL=pd.Series(l).rolling(40).min().shift(1).values
    L20=pd.Series(l).rolling(20).min().shift(1).values; spr=((l<L20)&(c>L20)).astype(float); SP=pd.Series(spr).rolling(25).max().shift(1).values
    v10=pd.Series(v).rolling(10).mean().shift(1).values; v40=pd.Series(v).rolling(40).mean().shift(1).values; v20=pd.Series(v).rolling(20).mean().shift(1).values
    dv20=pd.Series(c*v).rolling(20).mean().shift(1).values
    flipd=np.r_[False,(reg[1:]==1)&(reg[:-1]!=1)]; fl5=pd.Series(flipd.astype(float)).rolling(6).max().values
    rng_ok=(RH-RL)/a14<=12; dry=v10<=.8*v40; sos=(c>RH)&(v>=1.5*v20); spring=SP>0
    elig=(c>=1)&(c<5)&(dv20>=5e5)
    idx=np.arange(n); okf=idx+H+1<n
    ent=np.r_[o[1:],np.nan]; ex=np.full(n,np.nan); ex[:n-H-1]=o[H+1:]; ret=ex/ent-1-COST
    # bear-flip exit return for vector flips handled via F.r
    ev=pd.DataFrame({'tk':tk,'d':q.index,'ret':ret,'elig':elig,'rng':rng_ok,'spring':spring,'dry':dry,'sos':sos,'bull':reg==1,'fresh':fl5>0,'px':c})
    allx.append(ev[ev.elig&ev.ret.notna()][['d','ret']]); rows.append(ev[ev.elig&ev.ret.notna()&ev.sos&ev.rng])
E=pd.concat(rows); U=pd.concat(allx); U['mo']=U.d.dt.to_period('M'); mmean=U.groupby('mo').ret.mean()
E['mo']=E.d.dt.to_period('M'); E['ex']=E.ret-E.mo.map(mmean); E['test']=E.d>='2022-01-01'; E=E[E.d>='2019-01-01']
def tstat(z,col='ex'):
    m=z.groupby('mo')[col].mean(); return m.mean()/(m.std()/np.sqrt(len(m))) if len(m)>5 and m.std()>0 else np.nan
def line(name,z,col='ex'):
    out=[]
    for era,zz in (('TR',z[~z.test]),('TE',z[z.test])):
        out.append(f'{era}: n {len(zz):4d} mean {zz[col].mean()*100:+6.2f}% med {zz[col].median()*100:+6.2f}% big30 {np.mean(zz.ret>=.3)*100:4.1f}% t {tstat(zz,col):+.2f}')
    print(f'  {name:42s} '+' | '.join(out))
print(f'universe $1-5 stock-day mean 20d net: train {U[U.d<"2022-01-01"].ret.mean()*100:+.2f}% test {U[U.d>="2022-01-01"].ret.mean()*100:+.2f}%')
print('\nPART A: Wyckoff ALONE (20d hold, excess vs same-month $1-5 average); all rows already require 40d range + SOS')
line('SOS alone (range breakout on volume)',E)
line('SOS & SPRING',E[E.spring]); line('SOS & DRY',E[E.dry]); line('FULL WYCKOFF (SOS&SPRING&DRY)',E[E.spring&E.dry])
FW=E[E.spring&E.dry]
line('  FULL, vector FRESH flip (<=5 bars)',FW[FW.fresh]); line('  FULL, vector BULL older',FW[FW.bull&~FW.fresh]); line('  FULL, vector BEAR/none',FW[~FW.bull])
line('  SOS only no spring/dry, FRESH',E[~(E.spring&E.dry)&E.fresh]); line('  SOS only no spring/dry, BEAR/none',E[~(E.spring&E.dry)&~E.bull])
# PART B: flips
Fk=F.copy(); Fk['key']=list(zip(Fk.tk,Fk.d)); S=E.set_index(['tk','d']); FWi=set(map(tuple,FW[['tk','d']].values)); SOSi=set(map(tuple,E[['tk','d']].values))
Fk['wy']=[k in FWi for k in Fk.key]; Fk['sos']=[k in SOSi for k in Fk.key]; Fk['n']=Fk.r-COST; Fk['mo']=Fk.d.dt.to_period('M'); Fk['ex']=Fk.n
B=Fk[(Fk.px>=1)&(Fk.px<5)]; B=B[B.d>='2019-01-01']
print('\nPART B: vector flips $1-5, hold to bear flip, net 30bp (ex = net return)')
def lineb(name,z):
    out=[]
    for era,zz in (('TR',z[~z.test]),('TE',z[z.test])): out.append(f'{era}: n {len(zz):4d} mean {zz.n.mean()*100:+6.2f}% med {zz.n.median()*100:+6.2f}% big30 {np.mean(zz.n>=.3)*100:4.1f}% huge {np.mean(zz.n>=1)*100:4.1f}% t {tstat(zz,"n"):+.2f}')
    print(f'  {name:42s} '+' | '.join(out))
lineb('all flips $1-5',B); lineb('flip bar = SOS (range break, vol)',B[B.sos]); lineb('flip bar = FULL WYCKOFF',B[B.wy]); lineb('flip, not SOS',B[~B.sos]); lineb('flip, not FULL WYCKOFF',B[~B.wy])
C=B[B.lr>=F[~F.test].lr.quantile(.8)]; lineb('top-20% score $1-5',C); lineb('  + FULL WYCKOFF',C[C.wy]); lineb('  + SOS',C[C.sos])
E.to_pickle(X+'wyckoff_events.pkl')
