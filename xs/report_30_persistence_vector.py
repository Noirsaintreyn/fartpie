"""Nov 13 2026: ONE MODEL UNDER EVERYTHING WE FOUND. Hypothesis: price is a persistence process. The vector (pivot ratchet = first-passage of a trailing-drawdown process) only earns when returns are POSITIVELY autocorrelated at its horizon (variance ratio VR>1) and loses/vanishes when VR<1 (mean-reverting tape). This would tie: (i) vector 1h edge but 15m none, (ii) strong overnight inventory days (persistent imbalance) as the chain's only context, (iii) counter-auction vs with-inventory flipping between 2018-21 and 2022+ (the sign of persistence changed), (iv) fades of VA/VWAP losing (tape is persistent, not reverting), (v) pump/gap runners fading (VR<1 after a spike).
VR(q) = Var(sum of q 1h log-returns, overlapping) / (q Var(1 bar)), trailing 420 bars (~60 sessions), PIT at the flip bar. q=7 (~1 session) and q=35 (~1 week).
Test A (declared): 1h vector flips (A: flip while close>1h Maximin pivot L=20), forward +8 / +16 bars and hold-until-bear, by VR(7) bucket (>=1.05 persistent / 0.95-1.05 / <=0.95 reverting), eras 2018-21 vs 2022+. Controls: all bull-regime bars by the same VR bucket (drift only). PASS = persistent bucket beats reverting bucket in BOTH eras, same sign.
Test B: yearly mean VR (does the persistence regime itself change?). Data: 9 LSE names (1h from 15m RTH) in bp, and NQ 5m->1h RTH in points."""
import sys, glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'/home/claude/fartpie/xs')
from vector_pine import run_state_machine
L=20
def load(f):
    d=pd.read_parquet(f); d['datetime']=pd.to_datetime(d.datetime); d=d.drop_duplicates('datetime').set_index('datetime').sort_index()
    d.index=d.index.tz_localize('UTC').tz_convert('America/New_York').tz_localize(None); return d[['open','high','low','close','volume']].astype(float).between_time('09:30','15:59')
def h1(d): return d.resample('60min',offset='30min').agg(dict(open='first',high='max',low='min',close='last',volume='sum')).dropna(subset=['close'])
def vr(c,q,W=420):
    r=np.log(c).diff(); rq=np.log(c).diff(q)
    return rq.rolling(W).var()/(q*r.rolling(W).var())
def events(z,tag,pts=False):
    reg,lev,_,_=run_state_machine(z.assign(datetime=z.index)); reg=np.asarray(reg); lb=z.low.rolling(L).min().rolling(L).max().values; c=z.close.values; o=z.open.values; n=len(z)
    v7=vr(z.close,7).values; v35=vr(z.close,35).values; out=[]
    for i in range(500,n-17):
        if np.isnan(v7[i]): continue
        flipA=reg[i]==1 and reg[i-1]!=1 and c[i]>lb[i]; ctl=reg[i]==1 and i%10==0
        if not (flipA or ctl): continue
        e=i+1; ent=o[e]; res={}
        for N in (8,16): res[str(N)]=(c[i+N]-ent) if pts else (c[i+N]/ent-1)*1e4
        x=e
        while x<n-1 and reg[x]==1 and c[x]>=lev[x]: x+=1
        res['nat']=(c[x]-ent) if pts else (c[x]/ent-1)*1e4
        out.append(dict(t=z.index[i],kind='A flip|over pivot' if flipA else 'ctl bull bars',vr7=v7[i],vr35=v35[i],**res))
    return out
def bucket(v): return np.where(v>=1.05,'persistent VR>=1.05',np.where(v<=.95,'reverting VR<=0.95','neutral'))
def rep(R,unit):
    R['b']=bucket(R.vr7.values)
    for kind in ('A flip|over pivot','ctl bull bars'):
        print(f'  -- {kind}')
        for col in ('8','16','nat'):
            line=f'    exit {col:3s}'
            for b in ('persistent VR>=1.05','neutral','reverting VR<=0.95'):
                seg=[]
                for era,(a,e) in {'18-21':('2018-01-01','2022-01-01'),'22+':('2022-01-01','2027-01-01')}.items():
                    s=R[(R.kind==kind)&(R.b==b)&(R.t>=a)&(R.t<e)][col]; seg.append(f'{era} {s.mean():+7.1f} (n{len(s)})')
                line+=f' | {b[:10]}: '+' '.join(seg)
            print(line)
print('######## 9 LSE names, 1h, bp')
R=[];VRY=[]
for f in sorted(glob.glob('/home/claude/fartpie/lse_clean/*.parquet')):
    z=h1(load(f)); R+=events(z,'x'); v=vr(z.close,7).dropna(); VRY.append(v.groupby(v.index.year).mean())
R=pd.DataFrame(R); rep(R,'bp'); print('  yearly mean VR(7), 9-name avg:',' '.join(f'{y}:{v:.3f}' for y,v in pd.concat(VRY,axis=1).mean(axis=1).items()))
print('\n######## NQ 5m->1h RTH, points/contract')
nq=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); nq['dt']=pd.to_datetime(nq.dt); nq=nq.set_index('dt').sort_index()[['open','high','low','close','volume']].astype(float).between_time('09:30','15:59')
z=h1(nq); RN=pd.DataFrame(events(z,'nq',pts=True)); rep(RN,'pt'); v=vr(z.close,7).dropna(); print('  yearly mean VR(7), NQ:',' '.join(f'{y}:{x:.3f}' for y,x in v.groupby(v.index.year).mean().items()))
R.to_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/persist_R.pkl')
