"""ORDER-FLOW SETUP + TIGHTER STOPS + MORE SIZE on a Lucid Flex 50K eval (target 3000, MLL 2000 trailing->lock +100, day stop -1000, consistency 50%, cap 40 MNQ).
Setups (sign-corrected delta, thresholds from PRIOR days only): C10:30 = first-hour delta agrees with price & |delta ratio| in top third of prior days -> follow; A10:00 = first-30min delta+price agree & above prior median -> follow.
Stops are checked on 1-min CLOSES (no highs/lows in file) so they are optimistic; cost 0.5 pt. Starts every day 2024-12..2025-09. Nov 10 2026."""
import pandas as pd, numpy as np
X=pd.read_parquet('/mnt/user-data/uploads/fartpie/mnq_orderflow_1min.parquet')
X['et']=X.datetime.dt.tz_convert('America/New_York').dt.tz_localize(None); X['day']=X.et.dt.normalize(); X['m']=X.et.dt.hour*60+X.et.dt.minute
X['delta']=X.sell_vol-X.buy_vol; X['vol']=X.buy_vol+X.sell_vol
B={d:g.set_index('m') for d,g in X[(X.m>=570)&(X.m<960)].groupby('day') if len(g)>300}
days=sorted(B); rows=[]
for d in days:
    g=B[d]; w30=g[g.index<600]; w60=g[g.index<630]
    rows.append(dict(day=d,dr30=w30.delta.sum()/max(w30.vol.sum(),1),dr60=w60.delta.sum()/max(w60.vol.sum(),1),r30=w30.last_price.iloc[-1]-w30.last_price.iloc[0],r60=w60.last_price.iloc[-1]-w60.last_price.iloc[0]))
F=pd.DataFrame(rows).set_index('day')
F['th30']=F.dr30.abs().expanding().median().shift(1); F['th60']=F.dr60.abs().expanding().quantile(2/3).shift(1)
def path(g,m0,dr,S,T,cost=.5):
    e=g.last_price.get(m0,np.nan)
    if np.isnan(e): return None
    w=g.last_price[(g.index>m0)&(g.index<=955)]; worst=0.0
    for v in w.values:
        x=(v-e)*dr; worst=min(worst,x)
        if x<=-S: return -S-cost,-S-cost
        if T and x>=T: return T-cost,worst
    return (w.iloc[-1]-e)*dr-cost,worst
def trades(S,T,which):
    out={}
    for d in days[30:]:
        f=F.loc[d]; g=B[d]
        if 'C' in which and np.sign(f.dr60)==np.sign(f.r60) and abs(f.dr60)>=f.th60 and f.r60!=0: m0,dr=630,int(np.sign(f.dr60))
        elif 'A' in which and np.sign(f.dr30)==np.sign(f.r30) and abs(f.dr30)>=f.th30 and f.r30!=0: m0,dr=600,int(np.sign(f.dr30))
        else: continue
        r=path(g,m0,dr,S,T)
        if r: out[d]=r
    return out
def eval_sim(tr,n,s,days_):
    bal=0.0; floor=-2000.0; peak=0.0; best=0.0
    for i in range(s,len(days_)):
        d=days_[i]
        if d in tr:
            pts,worst=tr[d]; mult=2.0*n
            if bal+worst*mult<=floor: return 'fail',i-s+1
            pnl=max(pts*mult,-1000.0); 
        else: pnl=0.0
        bal+=pnl; best=max(best,pnl)
        if bal>=3000 and best<=.5*bal: return 'pass',i-s+1
        if floor<100: peak=max(peak,bal); floor=min(max(floor,peak-2000),100.0)
    return 'open',len(days_)-s
D_=days[30:]; starts=[i for i,d in enumerate(D_) if d<=pd.Timestamp('2025-08-15')]
print(f'{len(D_)} days {D_[0].date()}..{D_[-1].date()}, {len(starts)} starts')
print(f'{"setup":10} {"stop":>4} {"tgt":>4} {"MNQ":>3} | trades  mean pts  win | pass fail open | days to pass median | worst single-trade $')
for which in ('C','CA'):
    for S,T in ((10,None),(15,None),(20,None),(30,None),(20,40),(15,45)):
        tr=trades(S,T,which); arr=np.array([v[0] for v in tr.values()])
        for n in (10,20,30,40):
            R=[eval_sim(tr,n,s,D_) for s in starts]; p=[t for r,t in R if r=='pass']
            print(f'{which:10} {S:4d} {str(T or "-"):>4} {n:3d} | {len(arr):4d} {arr.mean():+8.1f} {np.mean(arr>0):4.0%} | {np.mean([r=="pass" for r,_ in R]):4.0%} {np.mean([r=="fail" for r,_ in R]):4.0%} {np.mean([r=="open" for r,_ in R]):4.0%} | {np.median(p) if p else float("nan"):5.0f} | {min(min(v[0],v[1]) for v in tr.values())*2*n:,.0f}')
