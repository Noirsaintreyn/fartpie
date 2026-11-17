"""Report 232: exits for the 1H vector flip trades (Rey Nov 17: a winner that comes back to the stop costs twice when the trailing drawdown counts open profit, so test break-even stops and trails instead of fixed 1:1). DECLARED before results.
Same entries as the 1:1 model (report 229-231: entry = open of the 5m bar after the 1H flip bar closes, initial stop = vector level, R >= 3 pts, flat 15:55, cost 1 pt, one trade at a time, 09:35-15:30). Exits tested: FIX 1R / 2R (reference); BE: stop to entry once +0.5R (target 1R) and once +1R (no target); VEC: stop trails the 1H vector level (moves with it, never back); ATR k=2,3: stop = best price -/+ k x 1H ATR(14) of the last closed 1H bar; EMA: close crosses 5m EMA(20) against after +0.5R, exit next open; VWAP: close crosses the VWAP anchored at the entry bar against after +0.5R, exit next open.
Metrics (train 2019-21 | test 2018+2022-26): n, win, net pts, day-clustered t, avg win / avg loss, mean MFE, 'round trips' = share of trades with MFE >= 1R that finish <= 0, and mean open-profit giveback (MFE - exit) which is what an intraday trailing drawdown that counts open profit pays."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/claude/fartpie/xs'); from vector_pine import run_state_machine
b=pd.read_parquet('/home/claude/nqdrive/nq5_drive_et.parquet'); b=b[b.dt>='2017-06-01'].reset_index(drop=True); o,h,l,c,v=[b[x].values for x in ('open','high','low','close','volume')]; n=len(b)
m=(b.dt.dt.hour*60+b.dt.dt.minute).values; day=b.dt.dt.normalize().values; t5=b.dt.values
G=pd.read_csv('/home/claude/fartpie/xs/ext/gex/QQQ.csv',parse_dates=['date']).set_index('date').net_gex.sort_index(); G.index=G.index.normalize()
g=b.set_index('dt').resample('60min',label='right',closed='left').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna(); res=run_state_machine(g.assign(datetime=g.index)); reg=np.asarray(res[0]); lev=np.asarray(res[1],float); ends=g.index.values
pc=g.close.shift(1); tr=np.maximum(g.high-g.low,np.maximum((g.high-pc).abs(),(g.low-pc).abs())); atr=tr.rolling(14).mean().values
ema=pd.Series(c).ewm(span=20,adjust=False).mean().values
hi1=np.searchsorted(ends,t5,'right')-1   # last closed 1H bar for each 5m bar
sig=[]
for i in range(1,len(g)):
    s=1 if (reg[i]==1 and reg[i-1]!=1) else -1 if (reg[i]==-1 and reg[i-1]!=-1) else 0
    if s==0 or not np.isfinite(lev[i]): continue
    e=np.searchsorted(t5,ends[i])
    if e>=n or t5[e]!=ends[i]: continue
    R=abs(o[e]-lev[i])
    if R>=3: sig.append((e,s,R,lev[i]))
def trade(e,s,R,mode,par):
    pe=o[e]; stop=pe-s*R; mfe=0.0; cross_ready=False; vp=0.0; vv=0.0; j=e; exitp=None; pend_exit=False
    while j<n:
        if pend_exit: exitp=o[j]; break
        # stop check (stop wins ties)
        if (s==1 and l[j]<=stop) or (s==-1 and h[j]>=stop):
            exitp=(min(stop,o[j]) if s==1 else max(stop,o[j])); break
        best=(h[j]-pe) if s==1 else (pe-l[j]); mfe=max(mfe,best)
        if mode=='fix' and best>=par*R: exitp=pe+s*par*R; break
        if m[j]>=955 or day[j]!=day[e]: exitp=c[j]; break
        tp=(h[j]+l[j]+c[j])/3; vp+=tp*v[j]; vv+=v[j]; vw=vp/max(vv,1)
        k=hi1[j]
        if mode=='be':
            thr,tg=par
            if mfe>=thr*R: stop=max(stop,pe) if s==1 else min(stop,pe)
            if tg and best>=tg*R: exitp=pe+s*tg*R; break
        elif mode=='vec':
            L=lev[k]
            if np.isfinite(L): stop=max(stop,L) if s==1 else min(stop,L)
        elif mode=='atr':
            a=atr[k]
            if np.isfinite(a):
                ext=(pe+mfe) if s==1 else (pe-mfe); ns=ext-s*par*a; stop=max(stop,ns) if s==1 else min(stop,ns)
        elif mode=='ema' and mfe>=.5*R and ((s==1 and c[j]<ema[j]) or (s==-1 and c[j]>ema[j])): pend_exit=True
        elif mode=='vwap' and mfe>=.5*R and ((s==1 and c[j]<vw) or (s==-1 and c[j]>vw)): pend_exit=True
        j+=1
    if exitp is None: exitp=c[min(j,n-1)]
    return s*(exitp-pe),mfe,j
MODES=[('FIX 1R','fix',1.0),('FIX 2R','fix',2.0),('BE@.5R tgt 1R','be',(.5,1.0)),('BE@1R no tgt','be',(1.0,None)),('VECTOR trail','vec',0),('ATR trail 2','atr',2.0),('ATR trail 3','atr',3.0),('EMA20 5m exit','ema',0),('anchored VWAP exit','vwap',0)]
def tstat(x):
    d=x.groupby('day').pts.sum(); return d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if len(d)>5 else np.nan
for gname,gf in (('all days',lambda d:True),('neg gamma',lambda d:True)):
    pass
rows={}
for nm,mode,par in MODES:
    out=[]; last=-1
    for e,s,R,L in sig:
        if e<=last or e>=n-2 or not (575<=m[e]<=930) or day[e]<np.datetime64('2018-01-01'): continue
        gp,mfe,j=trade(e,s,R,mode,par); d=pd.Timestamp(day[e]); kk=G.index.searchsorted(d)-1; a=G.iloc[kk] if kk>=0 else np.nan
        out.append((d,gp-1.0,gp,mfe,R,bool(np.isfinite(a) and a<0))); last=j
    rows[nm]=pd.DataFrame(out,columns=['day','pts','gross','mfe','R','neg'])
for gname in ('all','neg'):
    print(f'\n--- {gname} days ---')
    for nm,_,_ in MODES:
        D=rows[nm]; D=D[D.neg] if gname=='neg' else D; y=D.day.dt.year; cells=[]
        for era,mk in (('train',(y>=2019)&(y<=2021)),('test',(y==2018)|(y>=2022))):
            x=D[mk]; w=x[x.pts>0].pts; lo=x[x.pts<=0].pts; big=x[x.mfe>=x.R]; rt=(big.gross<=0).mean() if len(big) else np.nan; gb=(x.mfe-x.gross).mean()
            cells.append(f'{era}: n{len(x):4d} win{np.mean(x.pts>0)*100:3.0f}% net {x.pts.mean():+6.1f} t{tstat(x):+5.1f} W {w.mean():+5.0f} L {lo.mean():+5.0f} MFE {x.mfe.mean():4.0f} roundtrip {rt*100:3.0f}% giveback {gb:4.0f}')
        print(f'{nm:20s}',' || '.join(cells),flush=True)
pd.to_pickle(rows,'/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/exits232.pkl')
