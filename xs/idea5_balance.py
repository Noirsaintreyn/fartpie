"""Nov 10 2026 IDEA 5: BALANCE TIME (multi-day value-area overlap) + open location vs prior VA. DECLARED BEFORE RESULTS.
ov5 = mean over last 5 completed sessions of overlap(VA_k, VA_k-1) = |inter|/|union| of the 70% value areas (known at 09:30).
open_loc = prior-VA position of today's RTH open: -1 below VAL, 0 inside, +1 above VAH.
Tested on noise-engine breaks (G_noise pts), S6 day (non-overnight legs) pts/day; terciles of ov5 (cut within each period's own history, trailing 252) and open_loc groups.
PASS: high-vs-low tercile difference same sign, |t|>=2 in both 2018-21 and 2022-26."""
import sys; sys.path.insert(0,'/home/claude/fartpie/xs')
import numpy as np, pandas as pd, pickle, nz_core as Z
from nz_core import A, OPEN
src=open('/home/claude/fartpie/xs/idea3_levels.py').read(); ns={}
exec(src[:src.index("days=Z.DAYS; LV={}")].replace("rng=np.random","rng=np.random"),ns); prof=ns['prof']
SP=Z.SP; VA={}
for d in Z.ALLD:
    p=prof(d)
    if p: VA[d]=(p[4],p[3])             # VAL, VAH
ds=sorted(VA); ov=pd.Series({ds[i]:(max(0,min(VA[ds[i]][1],VA[ds[i-1]][1])-max(VA[ds[i]][0],VA[ds[i-1]][0])))/(max(VA[ds[i]][1],VA[ds[i-1]][1])-min(VA[ds[i]][0],VA[ds[i-1]][0])+1e-9) for i in range(1,len(ds))})
ov5=ov.rolling(5).mean().shift(1)           # known at today's open: sessions through yesterday
oloc=pd.Series({d:(-1 if OPEN[d]<VA[ds[i-1]][0] else 1 if OPEN[d]>VA[ds[i-1]][1] else 0) for i,d in enumerate(ds) if i>0})
def tt(a,b): return (a.mean()-b.mean())/np.sqrt(a.var()/len(a)+b.var()/len(b)) if len(a)>=10 and len(b)>=10 else np.nan
G=pd.read_pickle(SP+'/G_noise.pkl'); G['day']=pd.to_datetime(G.day).dt.normalize()
O,N=pickle.load(open(SP+'/s6.pkl','rb')); allr=[]
for nm,(df,_,_) in zip(N,O['V2']):
    if nm.startswith('ON'): continue
    df=df.copy(); df['day']=pd.to_datetime(df.day).dt.normalize(); df['w']=df.pts*df.sz; allr.append(df[['day','w']])
S6=pd.concat(allr).groupby('day').w.sum().reset_index()
NZd=G.groupby('day').pts.sum().reset_index().rename(columns={'pts':'w'})
def rep(name,df,wcol='w'):
    df=df[df.day>='2018-02-01'].copy(); df['ov']=ov5.reindex(df.day).values; df['ol']=oloc.reindex(df.day).values; o=[]
    for lab,m in (('18-21',df.day.dt.year<2022),('22-26',df.day.dt.year>=2022)):
        x=df[m].dropna(subset=['ov']); q=np.nanquantile(x.ov,[1/3,2/3]); lo=x[wcol][x.ov<=q[0]]; hi=x[wcol][x.ov>=q[1]]
        o.append(f'{lab}: ov5 lo(balance-low) {lo.mean():+5.1f} hi(balance) {hi.mean():+5.1f} t {tt(hi,lo):+.2f}')
    print(f' {name:6} BALANCE ',' | '.join(o)); o=[]
    for lab,m in (('18-21',df.day.dt.year<2022),('22-26',df.day.dt.year>=2022)):
        x=df[m].dropna(subset=['ol']); ins=x[wcol][x.ol==0]; out=x[wcol][x.ol!=0]
        o.append(f'{lab}: open inside VA {ins.mean():+5.1f} (n{len(ins)}) outside {out.mean():+5.1f} (n{len(out)}) t {tt(out,ins):+.2f}')
    print(f' {name:6} OPENLOC ',' | '.join(o))
rep('NOISE/trade',G.rename(columns={'pts':'w'})[['day','w']]); rep('NZ/day',NZd); rep('S6day',S6)
