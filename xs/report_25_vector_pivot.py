"""Nov 13 2026: VECTOR (pivot-ratchet state machine, xs/vector_pine.run_state_machine) x MAXIMIN PIVOT (LB=highest(lowest(low,20),20)). Long only. DECLARED BEFORE RESULTS:
 A = ALREADY OVER PIVOT (close>LB), VECTOR JUST FLIPPED to bull on this bar   -> buy next bar open
 B = ALREADY OVER VECTOR (regime bull on the prior bar too), PRICE JUST CROSSED above the pivot (close>LB, prior close<=LB) -> buy next bar open
 controls: V-flip alone (any pivot position), P-cross alone (any vector state), all bull-regime bars (unconditional), A-complement V-flip while BELOW pivot.
 Exits: fixed N bars (close of bar i+N; 15m capped at that day's last bar) and NATURAL (hold until regime turns bear / close < vector level; 15m also flat at the day's close).
 Universe S&P 500 + midsmall_universe (1,250). Data: Yahoo 1h 730d (RTH) and 15m = 5m 60d resampled (RTH). Costs: gross shown; net = minus 20 bp round trip. Split: first half vs second half of each sample (Yahoo intraday cannot reach 2018-21).  1h-tuned vector constants on 15m (flagged)."""
import sys, json, pickle, warnings, numpy as np, pandas as pd, yfinance as yf; warnings.filterwarnings('ignore'); sys.path.insert(0,'/home/claude/fartpie/xs')
from vector_pine import run_state_machine
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'; L=20; COST=20.0
sp=json.load(open('/home/claude/fartpie/sp500_tickers.json')); sp=list(sp.keys()) if isinstance(sp,dict) else [s if isinstance(s,str) else s.get('ticker') for s in sp]
ms=[l.strip() for l in open('/home/claude/fartpie/midsmall_universe.txt') if l.strip() and not l.startswith('#')]
U=sorted(set(t.replace('.','-') for t in sp+ms))
def fetch(period,interval,fn):
    try: return pickle.load(open(SC+fn,'rb'))
    except Exception: pass
    D={}
    for i in range(0,len(U),100):
        try:
            d=yf.download(U[i:i+100],period=period,interval=interval,prepost=False,group_by='ticker',auto_adjust=False,progress=False,threads=True)
            for t in U[i:i+100]:
                try:
                    x=d[t].dropna(subset=['Close'])
                    if len(x)>200: D[t]=x
                except Exception: pass
        except Exception as e: print('err',i,e,flush=True)
        print(fn,i,len(D),flush=True)
    pickle.dump(D,open(SC+fn,'wb')); return D
D1=fetch('730d','1h','vp_1h.pkl'); D5=fetch('60d','5m','vp_5m.pkl')
def prep(x):
    x=x.rename(columns=str.lower); x.index=pd.to_datetime(x.index); x.index=(x.index.tz_convert('America/New_York') if x.index.tz else x.index.tz_localize('UTC').tz_convert('America/New_York')).tz_localize(None)
    return x[['open','high','low','close','volume']].astype(float)
def to15(x): return x.resample('15min',origin='start_day').agg(dict(open='first',high='max',low='min',close='last',volume='sum')).dropna(subset=['close'])
def trades(z,tk,tf,Ns):
    if len(z)<120: return []
    reg,lev,_,_=run_state_machine(z.assign(datetime=z.index)); reg=np.asarray(reg); lev=np.asarray(lev,float)
    lb=z.low.rolling(L).min().rolling(L).max().values; c=z.close.values; o=z.open.values; n=len(z)
    day=z.index.normalize().values; lastbar={}; 
    for i,d in enumerate(day): lastbar[d]=i
    out=[]
    for i in range(L*2,n-2):
        if np.isnan(lb[i]) or np.isnan(lb[i-1]): continue
        over=c[i]>lb[i]; vflip=reg[i]==1 and reg[i-1]!=1; pcross=over and c[i-1]<=lb[i-1]
        tags=[]
        if vflip and over: tags.append('A flip|over pivot')
        if reg[i]==1 and reg[i-1]==1 and pcross: tags.append('B cross|over vector')
        if vflip: tags.append('ctl V-flip any')
        if vflip and not over: tags.append('ctl V-flip BELOW pivot')
        if pcross: tags.append('ctl P-cross any')
        if reg[i]==1 and i%10==0: tags.append('ctl bull bars (every 10th)')
        if not tags: continue
        e=i+1; ent=o[e]
        if not np.isfinite(ent) or ent<=0: continue
        res={}
        for N in Ns:
            x=min(i+N, lastbar[day[e]]) if tf=='15m' else min(i+N,n-1)
            if x<=e-1: continue
            res[N]=(c[x]/ent-1)*1e4
        x=e
        while x<n-1 and reg[x]==1 and c[x]>=lev[x] and not (tf=='15m' and day[x+1]!=day[e]): x+=1
        res['nat']=(c[x]/ent-1)*1e4; res['natbars']=x-e
        for tg in tags: out.append(dict(tk=tk,t=z.index[i],tag=tg,**{str(k):v for k,v in res.items()}))
    return out
def run(tf,D,Ns,prepf):
    R=[]
    for t,x in D.items():
        z=prepf(x); z=z.between_time('09:30','15:59') if tf=='15m' else z[(z.index.hour*60+z.index.minute>=570)&(z.index.hour*60+z.index.minute<960)]
        R+=trades(z,t,tf,Ns)
    R=pd.DataFrame(R); mid=R.t.sort_values().iloc[len(R)//2]; R['half']=np.where(R.t<mid,'H1','H2'); return R
def rep(R,tf,Ns):
    print(f'\n######## {tf}: trades {len(R)} | {R.t.min():%Y-%m-%d} -> {R.t.max():%Y-%m-%d}  (bp per trade, gross; net = -{COST:.0f}bp)')
    for tag,g in R.groupby('tag'):
        for col in [str(n) for n in Ns]+['nat']:
            s=g[col].dropna(); 
            if len(s)<30: continue
            t_=s.mean()/(s.std()/len(s)**.5); h1=g[g.half=='H1'][col].dropna(); h2=g[g.half=='H2'][col].dropna()
            extra=f' avg hold {g.natbars.mean():.1f} bars' if col=='nat' else ''
            print(f'{tag:28s} {("exit nat" if col=="nat" else "exit +"+col+" bars"):14s} n={len(s):6d} mean {s.mean():+7.1f} net {s.mean()-COST:+7.1f} win {np.mean(s>0):.0%} t {t_:+5.1f} | H1 {h1.mean():+7.1f} (n{len(h1)}) H2 {h2.mean():+7.1f} (n{len(h2)}){extra}')
R1=run('1h',D1,[4,8,16],prep); R1.to_pickle(SC+'vp_R1h.pkl'); rep(R1,'1h',[4,8,16])
R15=run('15m',D5,[4,8,16],lambda x: to15(prep(x))); R15.to_pickle(SC+'vp_R15.pkl'); rep(R15,'15m',[4,8,16])
