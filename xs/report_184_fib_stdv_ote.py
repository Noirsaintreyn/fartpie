"""Report 184: fib retracement (OTE) + standard-deviation projection strategies on NQ, many anchors. DECLARED BEFORE RESULTS (Rey Nov 15: 'try different fibs and fib stdv, asia range, NY open first 5m, first 15m, start of move').
Anchors (box/leg origin): ASIA = 20:00-24:00 ET prior evening range; NY5 = 09:30-09:35 range; NY15 = 09:30-09:45 range; SOM = start of move: first 5m close >= 0.5*ATR14 away from the 09:30 open, origin = extreme opposite to the impulse from 09:30 to that bar.
LEG: for boxes, first 5m close beyond a box edge (entries only from the bar after the box completes, until 12:00 ET) gives the direction; origin = opposite edge; end = running extreme in the direction (updated after the bar's fill check).
RETRACE entry: limit at fib f (0.5,0.618,0.705,0.79) of the leg origin->end (end excludes the current bar); stop at the origin; risk must be 6..60 pts else the day is skipped; TARGET = end + k*(leg) with k in (0, 1, 2) (0 = retest of the extreme, 1 and 2 = 'standard deviation' projections of the leg size). Stop checked before target in each bar after the fill bar; fill bar counts stop only. EOD exit 15:55 close. Cost 0.5. One trade per config per day.
EXTENSION FADE (boxes only): fade at box edge +/- k*W (W = box width, k in 1.5, 2, 2.5), stop 0.5 W beyond, target back 1.0 W; same risk rules; same window.
Strict split: 2018-21 | 2022+. PASS = same sign, |t|>=2, n>=30 both. 57 configs tested; chance PASS rate under the null is reported by shuffling the sign of daily outcomes (200 reps) so the count of real PASSes is judged against chance. Daily correlation of every PASS config with the system's daily points (variant-I legs + IFVG + S1 regime-ON, unweighted)."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/claude/fartpie/xs')
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet').sort_values('dt').drop_duplicates('dt'); b['d']=b.dt.dt.normalize(); b['m']=b.dt.dt.hour*60+b.dt.dt.minute
asia=b[b.m>=1200].assign(d=lambda x:(x.dt+pd.Timedelta(days=1)).dt.normalize()).groupby('d').agg(ah=('high','max'),al=('low','min'),n=('high','size'))
rt=b[(b.m>=570)&(b.m<=955)]; DAYS={d:g for d,g in rt.groupby('d') if len(g)>=70}; days=sorted(DAYS)
rg=pd.Series({d:g.high.max()-g.low.min() for d,g in DAYS.items()}); ATR=rg.rolling(14).mean().shift(1)
COST=.5; FIBS=(.5,.618,.705,.79); KS=(0,1,2); EXT=(1.5,2.,2.5); WEND=720
def run_dir(o,h,l,c,t0,d,origin,f,k):
    """long if d>0 (mirror via sign). returns (pts,mae) or None"""
    n=len(h); end=(h[t0] if d>0 else l[t0]); 
    for t in range(t0+1,n):
        if o[t] and (t*5+570)>WEND: return None
        lev=end-d*f*(d*(end-origin)); risk=d*(lev-origin)
        hit=(l[t]<=lev) if d>0 else (h[t]>=lev)
        if hit:
            if risk<6 or risk>60: return None
            ent=lev if (o[t]>=lev if d>0 else o[t]<=lev) else o[t]
            risk=d*(ent-origin); 
            if risk<=0: return None
            tgt=end+d*k*d*(end-origin); 
            if (l[t]<=origin) if d>0 else (h[t]>=origin): return (-risk-COST,-risk-COST)
            worst=0.0
            for u in range(t+1,n):
                lo=(l[u]-ent)*d if d>0 else (ent-h[u]); 
                if (l[u]<=origin) if d>0 else (h[u]>=origin): return (-risk-COST,-risk-COST)
                worst=min(worst,lo)
                if (h[u]>=tgt) if d>0 else (l[u]<=tgt): return (d*(tgt-ent)-COST,worst-COST)
            return (d*(c[-1]-ent)-COST,worst-COST)
        end=max(end,h[t]) if d>0 else min(end,l[t])
    return None
R={}
def add(key,d,res):
    if res is not None: R.setdefault(key,[]).append((d,res[0],res[1]))
for d in days:
    if d<pd.Timestamp('2018-01-01') or d>pd.Timestamp('2026-01-30'): continue
    g=DAYS[d]; o,h,l,c=g.open.values,g.high.values,g.low.values,g.close.values; n=len(h); at=ATR.get(d,np.nan)
    boxes={}
    if d in asia.index and asia.n[d]>=30: boxes['ASIA']=(asia.ah[d],asia.al[d],0)
    boxes['NY5']=(h[:1].max(),l[:1].min(),1); boxes['NY15']=(h[:3].max(),l[:3].min(),3)
    for nm,(H,L,ts) in boxes.items():
        W=H-L
        if W<3: continue
        t0=None
        for t in range(ts,n):
            if (t*5+570)>WEND: break
            if c[t]>H: t0,dr=t,1; break
            if c[t]<L: t0,dr=t,-1; break
        if t0 is not None:
            origin=L if dr>0 else H
            for f in FIBS:
                for k in KS: add((nm,'RET',f,k),d,run_dir(np.ones(n),h,l,c,t0,dr,origin,f,k) if False else run_dir(o,h,l,c,t0,dr,origin,f,k))
        # extension fade
        for k in EXT:
            for sg,lvl in ((1,H+k*W),(-1,L-k*W)):
                tt=None
                for t in range(ts,n):
                    if (t*5+570)>WEND: break
                    if (h[t]>=lvl if sg>0 else l[t]<=lvl): tt=t; break
                if tt is None: continue
                stop=lvl+sg*.5*W; tgt=lvl-sg*W; risk=abs(stop-lvl)
                if risk<6 or risk>60: break
                if (h[tt]>=stop if sg>0 else l[tt]<=stop): add((nm,'EXT',k,0),d,(-risk-COST,-risk-COST)); break
                res=None; worst=0.0
                for u in range(tt+1,n):
                    if (h[u]>=stop if sg>0 else l[u]<=stop): res=(-risk-COST,-risk-COST); break
                    worst=min(worst,(lvl-h[u]) if sg>0 else (l[u]-lvl))
                    if (l[u]<=tgt if sg>0 else h[u]>=tgt): res=(W-COST,worst-COST); break
                if res is None: res=(sg*(lvl-c[-1])-COST,worst-COST)
                add((nm,'EXT',k,0),d,res); break
    # start of move
    if np.isfinite(at):
        for t in range(1,n):
            if (t*5+570)>WEND: break
            if abs(c[t]-o[0])>=.5*at:
                dr=1 if c[t]>o[0] else -1; origin=l[:t].min() if dr>0 else h[:t].max(); origin=min(origin,o[0]) if dr>0 else max(origin,o[0])
                for f in FIBS:
                    for k in KS: add(('SOM','RET',f,k),d,run_dir(o,h,l,c,t,dr,origin,f,k))
                break
rows=[]
for key,v in R.items():
    X=pd.DataFrame(v,columns=['d','pts','mae']); rows.append((key,X))
pd.to_pickle(dict(rows),'/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/fibstdv184.pkl')
def st(s): n=len(s); return n,s.mean(),(s.mean()/(s.std()/np.sqrt(n)) if n>2 and s.std()>0 else 0)
def passed(X):
    a=st(X.pts[X.d<'2022-01-01']); bb=st(X.pts[X.d>='2022-01-01']); return a,bb,(np.sign(a[1])==np.sign(bb[1]) and abs(a[2])>=2 and abs(bb[2])>=2 and a[0]>=30 and bb[0]>=30)
rng=np.random.default_rng(1); null=[]
for _ in range(200):
    c_=0
    for key,X in rows:
        Y=X.copy(); Y['pts']=Y.pts.values*rng.choice([-1,1],len(Y))*0+rng.permutation(Y.pts.values)*rng.choice([-1,1],len(Y)); c_+=passed(Y)[2]
    null.append(c_)
npass=0; print('anchor kind fib k | 2018-21 n mean t | 2022+ n mean t'); out=[]
for key,X in sorted(rows,key=lambda r:r[0]):
    a,bb,ok=passed(X); npass+=ok
    print(f'{key[0]:5s} {key[1]:3s} {key[2]:5} {key[3]} | {a[0]:4d} {a[1]:+6.2f} t{a[2]:+5.1f} | {bb[0]:4d} {bb[1]:+6.2f} t{bb[2]:+5.1f}{"  <== PASS" if ok else ""}')
print(f'\nreal PASS {npass} of {len(rows)} configs; null (sign-shuffled) mean {np.mean(null):.1f}, 95th pct {np.percentile(null,95):.0f}')
