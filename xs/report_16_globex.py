"""Nov 13 2026: STANDALONE GLOBEX-SESSION MODELS for NQ (Asia 18:00-03:00 ET, London 03:00-08:30). Rey: need something that makes money on days the main RTH system doesn't; vendor products trade 'around the clock'. DECLARED BEFORE RESULTS.
Data: NQ 5m ETH (nqdrive/nq5_drive_et.parquet, ET). Trade date D = the session ending 09:30 D. Entry = next 5m bar open after the decision bar, exit 08:25 bar close (flat well before NY open, no overnight hold), hard stop 25 pts (stop-first), cost 0.5 pt/trade (per side-pair), 1 contract, results in NQ points.
Models (mechanism):
 L1/L1f  London-open drive: net move 03:00->03:25 bars >= 0.10 x ADR20(RTH); follow / fade.  (Europe cash open starts institutional flow)
 A1/A1f  Asia-trend at Europe arrival: net move 18:00->03:00 >= 0.30 ADR20; fade (thin-liquidity move reverts) / follow, enter 03:05.
 S1      Asia-range sweep reversal: a London bar (03:00-06:00) wicks past the Asia high/low and CLOSES back inside -> fade; stop = wick extreme +/-2 (skip risk>40), exit 08:25.
 B1/B1f  London 03:00-04:00 range break: first 5m close beyond the range between 04:00-06:00 -> follow / fade (stop opposite side capped 40).
 P1      Pre-NY run-in: London (03:00->08:00) net move >= 0.25 ADR20 -> follow into 08:25? (enter 08:05, exit 09:25: pre-open continuation) / P1f fade.
Eras: A 2018-02..2021-12 (train), B 2022-01..2026-01 (test). t = mean/(sd/sqrt n). 9 configs -> ~0.4 expected |t|>2 hits per era by chance; require the same sign and |t|>=2 in BOTH eras, positive in >=70% of years."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').sort_values('dt').drop_duplicates('dt').reset_index(drop=True); b['d']=b.dt.dt.normalize(); b['m']=b.dt.dt.hour*60+b.dt.dt.minute
rth=b[(b.m>=570)&(b.m<960)]; Dd=rth.groupby('d').agg(h=('high','max'),l=('low','min')); adr=(Dd.h-Dd.l).rolling(20).mean().shift()
by=dict(tuple(b.groupby('d'))); days=sorted(by); COST=.5; STOP=25
def sess(D,i):
    """returns arrays for trade date D: prior-evening 18:00+ bars and D 00:00-09:25 bars, concatenated with minute-of-session index (m_s: 18:00=0)"""
    if i==0: return None
    p=by[days[i-1]]; p=p[p.m>=1080]; c=by[D]; c=c[c.m<565]
    if len(p)<30 or len(c)<100: return None
    x=pd.concat([p,c]); x['ms']=np.where(x.m>=1080,x.m-1080,x.m+360); return x.reset_index(drop=True)
def run(x,e,ex,dr,stop,ent_price=None):
    """enter at bar e open, exit at close of bar ex; stop in pts from entry (or abs price); returns pts net"""
    ent=x.open[e] if ent_price is None else ent_price; S=ent-dr*stop
    for k in range(e,ex+1):
        if (x.low[k]<=S) if dr>0 else (x.high[k]>=S): return (S-ent)*dr-COST
    return (x.close[ex]-ent)*dr-COST
R=[]
for i,D in enumerate(days):
    if D<pd.Timestamp('2018-02-01') or D>pd.Timestamp('2026-01-30') or D not in adr.index or not np.isfinite(adr.get(D,np.nan)): continue
    x=sess(D,i)
    if x is None: continue
    A=adr[D]; ms=x.ms.values; idx=lambda t:int(np.searchsorted(ms,t))
    i03=idx(540); i0325=idx(565); i0330=idx(570); i0305=idx(545); i04=idx(600); i06=idx(720); i08=idx(840); i0805=idx(845); i0825=idx(865)-1; i0925=idx(925)-1
    if i0925>=len(x) or i0825>=len(x) or i03>=i0325: continue
    # L1
    mv=x.close[i0325]-x.open[i03]
    if abs(mv)>=.10*A:
        dr=np.sign(mv); R.append((D,'L1',run(x,i0330,i0825,dr,STOP))); R.append((D,'L1f',run(x,i0330,i0825,-dr,STOP)))
    # A1
    mv=x.close[i03-1]-x.open[0]
    if abs(mv)>=.30*A:
        dr=np.sign(mv); R.append((D,'A1',run(x,i0305,i0825,-dr,STOP))); R.append((D,'A1f',run(x,i0305,i0825,dr,STOP)))
    # S1
    ah=x.high[:i03].max(); al=x.low[:i03].min(); done=False
    for k in range(i03,i06):
        if x.high[k]>ah and x.close[k]<ah: dr=-1; ext=x.high[k]; done=True
        elif x.low[k]<al and x.close[k]>al: dr=1; ext=x.low[k]; done=True
        if done:
            e=k+1; ent=x.open[e]; risk=abs(ent-ext)+2
            if 4<=risk<=40: R.append((D,'S1',run(x,e,i0825,dr,risk)))
            break
    # B1
    rh=x.high[i03:i04].max(); rl=x.low[i03:i04].min()
    for k in range(i04,i06):
        up=x.close[k]>rh; dn=x.close[k]<rl
        if up or dn:
            dr=1 if up else -1; e=k+1; ent=x.open[e]; rsk=min(abs(ent-(rl if up else rh)),40)
            R.append((D,'B1',run(x,e,i0825,dr,max(rsk,8)))); R.append((D,'B1f',run(x,e,i0825,-dr,max(rsk,8)))); break
    # P1
    mv=x.close[i08-1]-x.open[i03]
    if abs(mv)>=.25*A:
        dr=np.sign(mv); R.append((D,'P1',run(x,i0805,i0925,dr,STOP))); R.append((D,'P1f',run(x,i0805,i0925,-dr,STOP)))
T=pd.DataFrame(R,columns=['d','m','pts']); T['era']=np.where(T.d<pd.Timestamp('2022-01-01'),'A','B'); T['yr']=T.d.dt.year
def t_(a): return a.mean()/(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>5 and a.std()>0 else np.nan
print('days',len(set(T.d)),'trades',len(T))
for m,g in T.groupby('m',sort=False):
    a,bb=g[g.era=='A'].pts,g[g.era=='B'].pts; yp=(g.groupby('yr').pts.mean()>0).mean()
    ok='  <== PASS (same sign, |t|>=2 both eras)' if (a.mean()*bb.mean()>0 and abs(t_(a))>=2 and abs(t_(bb))>=2) else ''
    print('%-4s n %4d | A n %4d mean %+6.2f win %2.0f%% t %+5.1f | B n %4d mean %+6.2f win %2.0f%% t %+5.1f | yrs>0 %2.0f%% | sum A %+7.0f B %+7.0f%s'%(m,len(g),len(a),a.mean(),(a>0).mean()*100,t_(a),len(bb),bb.mean(),(bb>0).mean()*100,t_(bb),yp*100,a.sum(),bb.sum(),ok))
T.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/globex_T.pkl')
