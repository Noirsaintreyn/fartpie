"""Nov 13 2026: is a VWAP z-score a better 'fair value' than fundamentals? Session VWAP (reset 09:30 ET, typical price x volume), z = (close - VWAP)/volume-weighted std of typical price around VWAP (cumulative, standard VWAP bands). DECLARED BEFORE RESULTS: first touch per side per day with z >= T or <= -T (T in 1.5, 2, 3), only after >= 8 bars of session; FADE = trade back toward VWAP at next bar open; the same table read with the sign flipped is CONTINUATION. Exits: +4 bars, +8 bars, session close. Data: LSE 15m cache (9 names, 2018-2026, UTC->ET) in bp/trade, and NQ 5m ETH->RTH in points/contract (cost 0.5 pt). PASS = same sign, |t|>=2 in BOTH eras (2018-21, 2022+), n>=30."""
import glob, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
def sess_events(z,T,Ns,pts=False,minbars=8):
    out=[]
    for d,g in z.groupby(z.index.normalize()):
        if len(g)<minbars+12: continue
        tp=((g.high+g.low+g.close)/3).values; v=g.volume.values.astype(float); c=g.close.values; o=g.open.values
        cv=np.cumsum(v); vw=np.cumsum(tp*v)/np.where(cv>0,cv,np.nan); var=np.cumsum(v*tp**2)/np.where(cv>0,cv,np.nan)-vw**2; sd=np.sqrt(np.maximum(var,1e-12))
        zz=(c-vw)/sd; seen=set()
        for i in range(minbars,len(g)-2):
            for sgn in (1,-1):
                if sgn in seen: continue
                if sgn*zz[i]>=T:
                    seen.add(sgn); e=i+1; ent=o[e]; dr=-sgn   # fade
                    r={}
                    for N in Ns: x=min(i+N,len(g)-1); r[N]=dr*((c[x]-ent) if pts else (c[x]/ent-1)*1e4)
                    r['close']=dr*((c[-1]-ent) if pts else (c[-1]/ent-1)*1e4)
                    out.append(dict(d=d,side=sgn,**{str(k):v for k,v in r.items()}))
    return pd.DataFrame(out)
def report(R,unit,cost,cols=('4','8','close')):
    for col in cols:
        line=f'    exit {col:5s}'
        for era,(a,b) in {'2018-21':('2018-01-01','2022-01-01'),'2022+':('2022-01-01','2027-01-01')}.items():
            s=R[(R.d>=a)&(R.d<b)][col]; t=s.mean()/(s.std()/len(s)**.5) if len(s)>5 else np.nan
            line+=f' | {era}: n={len(s):5d} fade {s.mean():+7.2f}{unit} (t {t:+.1f}, win {np.mean(s>0):.0%}) net {s.mean()-cost:+7.2f}'
        print(line)
def load(f):
    d=pd.read_parquet(f); d['datetime']=pd.to_datetime(d.datetime); d=d.drop_duplicates('datetime').set_index('datetime').sort_index()
    d.index=d.index.tz_localize('UTC').tz_convert('America/New_York').tz_localize(None); return d[['open','high','low','close','volume']].astype(float).between_time('09:30','15:59')
stocks={f.split('/')[-1].split('_')[0]:load(f) for f in sorted(glob.glob('/home/claude/fartpie/lse_clean/*.parquet'))}
nq=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); nq['dt']=pd.to_datetime(nq.dt); nq=nq.set_index('dt').sort_index()[['open','high','low','close','volume']].astype(float).between_time('09:30','15:59')
for T in (1.5,2,3):
    print(f'\n######## STOCKS (9 LSE names, 15m) |z|>={T}: FADE back to VWAP (bp/trade gross; net = -2bp)')
    R=pd.concat([sess_events(z,T,[4,8]).assign(tk=t) for t,z in stocks.items()]); report(R,'bp',2.0)
    print(f'######## NQ (5m RTH) |z|>={T}: FADE back to VWAP (points/contract gross; net = -0.5)')
    R=sess_events(nq,T,[12,24],pts=True,minbars=12); report(R,'pt',0.5,('12','24','close'))
