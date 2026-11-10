"""Nov 10 2026 IDEA 4: BREADTH of the pivot-ratchet vector across the names cached locally (54 large caps, DAILY bars 2017-06..2026-10; the screener's 15m/500-name cache lives on the Mac).
breadth_t = share of names whose daily vector regime is bullish at the PRIOR close (known at 09:30). Test: NQ noise engine pts/day and S6 day pts split by breadth terciles, and WITH vs AGAINST (noise trade side vs breadth>.5), 2018-21 vs 2022-26.
PASS: same sign and |t|>=2 in both periods. Also dbreadth = change over 5 sessions."""
import sys; sys.path.insert(0,'/home/claude/fartpie/discord_bot'); sys.path.insert(0,'/home/claude/fartpie/xs')
import glob, os, numpy as np, pandas as pd, pickle, warnings; warnings.filterwarnings('ignore')
from pivot_ratchet_core import run_state_machine
R={}
for f in sorted(glob.glob('/home/claude/fartpie/data/yahoo/daily/*.parquet')):
    t=os.path.basename(f)[:-8]; d=pd.read_parquet(f).dropna()
    if len(d)<500: continue
    try: reg,_,_,_=run_state_machine(d); R[t]=pd.Series(np.asarray(reg).astype(int),index=pd.to_datetime(d.index).normalize())
    except Exception as e: print(t,'skip',str(e)[:60])
B=pd.DataFrame(R); print('names',B.shape[1],'| state values',sorted(set(B.stack().unique())))
br=(B>0).sum(axis=1)/B.notna().sum(axis=1); br=br.shift(1)      # prior close -> known at the open
d5=br-br.shift(5)
SP='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad'
G=pd.read_pickle(SP+'/G_noise.pkl'); G['day']=pd.to_datetime(G.day).dt.normalize(); NZd=G.groupby('day').pts.sum()
O,N=pickle.load(open(SP+'/s6.pkl','rb')); a=[]
for nm,(df,_,_) in zip(N,O['V2']):
    if nm.startswith('ON'): continue
    df=df.copy(); df['day']=pd.to_datetime(df.day).dt.normalize(); df['w']=df.pts*df.sz; a.append(df[['day','w']])
S6=pd.concat(a).groupby('day').w.sum()
def tt(x,y): return (x.mean()-y.mean())/np.sqrt(x.var()/len(x)+y.var()/len(y)) if len(x)>=10 and len(y)>=10 else np.nan
for nm,ser in (('NZ/day',NZd),('S6day',S6)):
    ser=ser[ser.index>='2018-02-01']; o=[]
    for lab,m in (('18-21',ser.index.year<2022),('22-26',ser.index.year>=2022)):
        x=ser[m]; b=br.reindex(x.index); q=np.nanquantile(b,[1/3,2/3]); lo=x[b<=q[0]]; hi=x[b>=q[1]]; o.append(f'{lab}: breadth lo {lo.mean():+5.1f} hi {hi.mean():+5.1f} t {tt(hi,lo):+.2f}')
    print(f' {nm:7}',' | '.join(o))
G2=G[G.day>='2018-02-01'].copy(); G2['b']=br.reindex(G2.day).values; G2['with']=np.where(G2.b>.5,1,-1)*G2.side>0; o=[]
for lab,m in (('18-21',G2.day.dt.year<2022),('22-26',G2.day.dt.year>=2022)):
    x=G2[m].dropna(subset=['b']); w=x.pts[x['with']]; ag=x.pts[~x['with']]; o.append(f'{lab}: WITH breadth {w.mean():+5.1f} (n{len(w)}) AGAINST {ag.mean():+5.1f} (n{len(ag)}) t {tt(w,ag):+.2f}')
print(' NOISE trades',' | '.join(o))
for lab,m in (('18-21',G2.day.dt.year<2022),('22-26',G2.day.dt.year>=2022)):
    x=G2[m].dropna(subset=['b']); x['d']=d5.reindex(x.day).values; x=x.dropna(subset=['d']); w=x.pts[np.sign(x.d)*x.side>0]; ag=x.pts[np.sign(x.d)*x.side<0]; print(f'   {lab} WITH 5d breadth change {w.mean():+5.1f} (n{len(w)}) AGAINST {ag.mean():+5.1f} (n{len(ag)}) t {tt(w,ag):+.2f}')
