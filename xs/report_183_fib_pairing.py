"""Report 183: fib pairing (Rey: 'pairing fib with stuff if it works now'). DECLARED BEFORE RESULTS.
Mechanism: after a trending prior day, the 0.62-0.79 retracement (OTE) of that day's range is where a pullback ends; post-0DTE dealer hedging makes round/proportional levels sticky. Prior-day direction = sign(close-open) of RTH.
A) report-182 fib touches (0.618/0.705/0.79/0.5/0.382): trade in the PRIOR-DAY direction at first touch while retracing (approach against prior direction), 30/60 min, raw pts.
B) IFVG trades (report 166 universe, 656): ENT inside OTE band [0.62,0.79] retracement of prior-day range (from its extreme in the prior direction) AND trade dir == prior-day dir (pullback entry) vs rest; also OTE band regardless of direction.
C) S1 premium fade regime-ON: not applicable (fades a gap).
Strict split 2018-21 | 2022+, PASS = same sign |t|>=2 n>=30 both; also shown: 2022+ alone because Rey's regime argument (0DTE)."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').sort_values('dt').drop_duplicates('dt'); b['d']=b.dt.dt.normalize(); b['m']=b.dt.dt.hour*60+b.dt.dt.minute
R=b[(b.m>=570)&(b.m<=955)].groupby('d').agg(o=('open','first'),h=('high','max'),l=('low','min'),c=('close','last')); R['dir']=np.sign(R.c-R.o); P=R.shift(1)
def st(s): n=len(s); return n,s.mean(),(s.mean()/(s.std()/np.sqrt(n)) if n>2 and s.std()>0 else 0)
def show(lab,x,col):
    r=[st(x[col][x.d<'2022-01-01']),st(x[col][x.d>='2022-01-01'])]; ok=np.sign(r[0][1])==np.sign(r[1][1]) and all(abs(q[2])>=2 and q[0]>=30 for q in r)
    print(f'  {lab:34s} A {r[0][1]:+6.2f} (t{r[0][2]:+.1f},n{r[0][0]}) | B {r[1][1]:+6.2f} (t{r[1][2]:+.1f},n{r[1][0]}){"  <== PASS" if ok else ""}')
E=pd.read_pickle(SP+'ext/profile_events182.pkl'); E=E[E.ty.str.startswith('FIB')&(E.ty!='FIBxVOL')].copy(); E['pd']=E.d.map(P.dir); E=E[E.pd!=0]
E['ret']=(E.a==-E.pd); E['t30']=-E.c30; E['t60']=-E.c60
print('A) fib first touch, trade in the PRIOR-DAY direction when the touch is a retracement (raw pts at 30 | 60 min)')
for ty in sorted(E.ty.unique()):
    x=E[(E.ty==ty)&E.ret]; show(ty+' 30m',x,'t30'); show(ty+' 60m',x,'t60')
x=E[E.ty.isin(['FIB0.618','FIB0.705','FIB0.79'])&E.ret]; x=x.sort_values('t').groupby('d').head(1); show('OTE any (first/day) 30m',x,'t30'); show('OTE any (first/day) 60m',x,'t60')
I=pd.read_pickle(SP+'ext/uni_ifvg.pkl'); I=I[I.model=='IFVG'].copy(); I['d']=pd.to_datetime(I.d).dt.normalize()
for c in ('h','l','dir'): I['p'+c]=I.d.map(P[c])
rr=I.ph-I.pl; I['rt']=np.where(I.pdir>0,(I.ph-I.ent)/rr,np.where(I.pdir<0,(I.ent-I.pl)/rr,np.nan)); I['ote']=(I.rt>=.62)&(I.rt<=.79); I['pull']=I.ote&(I.dr==I.pdir); I['inret']=(I.rt>=.38)&(I.rt<=.79)&(I.dr==I.pdir)
print('\nB) IFVG trades (pts per trade)')
show('all IFVG',I,'pts'); show('ENT in OTE .62-.79',I[I.ote],'pts'); show('OTE & dir = prior-day dir',I[I.pull],'pts'); show('retrace .38-.79 & dir = prior dir',I[I.inret],'pts'); show('not OTE',I[~I.ote&I.rt.notna()],'pts'); show('dir = prior dir (any)',I[I.dr==I.pdir],'pts'); show('dir against prior dir',I[I.dr==-I.pdir],'pts')
