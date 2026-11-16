"""report 164: EMA grid, timeframes 5m/15m/1H/4H/daily x lengths 9/21/50/100/200 (Rey: 'different time frames ema and different counts'). DECLARED BEFORE RESULTS; whole grid reported, nothing picked.
Use 1 CONTEXT: IFVG trade (ext/uni_ifvg.pkl) kept only if its direction agrees with sign(entry price - EMA) using the last COMPLETED bar of that timeframe. Use 2 STANDALONE: at 10:00 ET take the side of price vs EMA (last completed bar), hold to 15:55, cost .5.
25 cells x 2 uses; PASS = same sign, |t|>=2, n>=30 in A (2018-21) and B (2022+). With this many cells a few false PASSes are expected by chance, so read PASS counts against that."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
S='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; COST=.5; BL=pd.Timestamp('2022-01-01')
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').sort_values('dt').drop_duplicates('dt').reset_index(drop=True); b['d']=b.dt.dt.normalize(); b['m']=b.dt.dt.hour*60+b.dt.dt.minute
LEN=(9,21,50,100,200); EM={}
def hbars(key):
    return b.groupby(key).close.last()
for tf,key in (('5m',b.dt),('15m',b.dt.dt.floor('15min')),('1H',b.dt.dt.floor('60min')),('4H',(b.dt-pd.Timedelta(hours=18)).dt.floor('4h')+pd.Timedelta(hours=18))):
    c=hbars(key)
    for L in LEN:
        e=c.ewm(span=L,adjust=False).mean().shift(1)   # last COMPLETED bar's value is visible from the next bar on
        EM[(tf,L)]=e.reindex(b.dt.values if tf=='5m' else key.values).values   # value aligned to each 5m bar via its HTF bar start
rt=b[(b.m>=570)&(b.m<=955)]; dc=rt.groupby('d').close.last()
for L in LEN: EM[('D',L)]=b.d.map(dc.ewm(span=L,adjust=False).mean().shift(1)).values
ix=pd.Series(np.arange(len(b)),index=b.dt.values)
t=pd.DataFrame(pd.read_pickle(S+'ext/uni_ifvg.pkl')); t=t[t.model=='IFVG'].copy(); t['d']=pd.to_datetime(t.d)
t['j']=[ix.get(r.d+pd.Timedelta(minutes=int(r.m)),-1) for _,r in t.iterrows()]; t=t[t.j>0]
# standalone: 10:00 entries
cl=b.close.values; op=b.open.values; mday=b.m.values; dd=b.d.values
e10=b[(b.m==600)&(b.d>='2018-02-01')&(b.d<'2026-02-01')]; i10=e10.index.values; dayN=e10.d.values
x15=[]; 
for i,dn in zip(i10,dayN):
    seg=b[(b.d==dn)&(b.m<=955)&(b.m>=600)]
    if len(seg) and seg.m.iloc[-1]==955: x15.append((i,dn,seg.open.iloc[0],seg.close.iloc[-1]))
X=pd.DataFrame(x15,columns=['i','d','o','c']); X['d']=pd.to_datetime(X.d)
def st(s):
    n=len(s); return (n,s.mean(),(s.mean()/(s.std()/np.sqrt(n)) if n>2 and s.std()>0 else 0)) if n else (0,np.nan,0)
def cell(df):
    a,bb=st(df[df.d<BL].pts),st(df[df.d>=BL].pts); ok=a[0]>=30 and bb[0]>=30 and abs(a[2])>=2 and abs(bb[2])>=2 and np.sign(a[1])==np.sign(bb[1]); return a,bb,ok
base=cell(t.assign(pts=t.pts)); print(f'IFVG all: A n{base[0][0]} {base[0][1]:+.2f} t{base[0][2]:+.1f} | B n{base[1][0]} {base[1][1]:+.2f} t{base[1][2]:+.1f}\n')
print('USE 1: IFVG kept when direction agrees with price vs EMA  (A mean t | B mean t, n)'); npass1=0
for tf in ('5m','15m','1H','4H','D'):
    row=[]
    for L in LEN:
        e=EM[(tf,L)]; ev=e[t.j.values]; ag=(t.dr.values*(t.ent.values-ev)>0)&~np.isnan(ev); r=cell(t[ag]); npass1+=r[2]
        row.append(f'{L:3d}: {r[0][1]:+5.1f}/{r[0][2]:+4.1f} | {r[1][1]:+5.1f}/{r[1][2]:+4.1f} n{r[0][0]:3d},{r[1][0]:3d}{"*" if r[2] else " "}')
    print(tf.ljust(4),'  '.join(row))
print('(* = PASS)  PASS cells:',npass1,'of 25  (IFVG-all itself passes, so agreeing cells that keep ~half the trades pass easily; compare with the AGAINST table)')
print('\nUSE 1b: IFVG trades AGAINST the EMA side'); npass1b=0
for tf in ('5m','15m','1H','4H','D'):
    row=[]
    for L in LEN:
        e=EM[(tf,L)]; ev=e[t.j.values]; ag=(t.dr.values*(t.ent.values-ev)<0)&~np.isnan(ev); r=cell(t[ag]); npass1b+=r[2]
        row.append(f'{L:3d}: {r[0][1]:+5.1f}/{r[0][2]:+4.1f} | {r[1][1]:+5.1f}/{r[1][2]:+4.1f} n{r[0][0]:3d},{r[1][0]:3d}{"*" if r[2] else " "}')
    print(tf.ljust(4),'  '.join(row))
print('PASS cells:',npass1b,'of 25')
print('\nUSE 2: standalone 10:00 -> 15:55, side = price vs EMA  (A mean/t | B mean/t)'); npass2=0
for tf in ('5m','15m','1H','4H','D'):
    row=[]
    for L in LEN:
        e=EM[(tf,L)][X.i.values]; side=np.sign(X.o.values-e); pts=(X.c.values-X.o.values)*side-COST; df=X.assign(pts=pts)[~np.isnan(e)]; r=cell(df); npass2+=r[2]
        row.append(f'{L:3d}: {r[0][1]:+5.1f}/{r[0][2]:+4.1f} | {r[1][1]:+5.1f}/{r[1][2]:+4.1f}{"*" if r[2] else " "}')
    print(tf.ljust(4),'  '.join(row))
print('PASS cells:',npass2,'of 25')
