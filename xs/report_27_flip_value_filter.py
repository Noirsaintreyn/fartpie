"""Nov 13 2026: does value cheapness (report_26) improve DAILY VECTOR FLIPS? Events = daily vector regime flip to bull (xs/vector_pine.run_state_machine on daily OHLC, S&P PIT tickers with SEC sales/share), optionally also over the daily Maximin pivot (A). Cheapness measured at the flip: cross-sectional percentile (as of the latest month start before the flip) of (i) raw log P/S, (ii) macro z (fixed 2018-21 fit; OOS only valid 2022+), (iii) own-history z. Buckets: cheapest third / middle / richest third. Outcome: forward 21d/63d return minus the universe equal-weight return over the same window. Eras: 2018-21 vs 2022+. Declared: PASS = cheap-third beats rich-third in BOTH eras (same sign, |t|>=2 in at least the test era)."""
import sys, pickle, warnings, numpy as np, pandas as pd, yfinance as yf; warnings.filterwarnings('ignore'); sys.path.insert(0,'/home/claude/fartpie/xs')
from vector_pine import run_state_machine
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
Z=pd.read_pickle(SC+'macroz_Z.pkl'); tk=sorted(Z.tk.unique())
try: O=pickle.load(open(SC+'flipval_ohlc.pkl','rb'))
except Exception:
    O={}
    for i in range(0,len(tk),100):
        d=yf.download([t.replace('.','-') for t in tk[i:i+100]],start='2017-01-01',interval='1d',auto_adjust=True,group_by='ticker',progress=False,threads=True)
        for t in tk[i:i+100]:
            try:
                x=d[t.replace('.','-')].dropna(subset=['Close']).rename(columns=str.lower)[['open','high','low','close']]
                if len(x)>300: O[t]=x
            except Exception: pass
        print('px',i,len(O),flush=True)
    pickle.dump(O,open(SC+'flipval_ohlc.pkl','wb'))
C=pd.DataFrame({t:x.close for t,x in O.items()}).sort_index(); H=(21,63)
EW={h:(C.shift(-h)/C-1).mean(axis=1) for h in H}
for c in ('z','z0','lm'): Z[c+'_pct']=Z.groupby('d')[c].rank(pct=True)
Zi=Z.set_index(['tk','d']).sort_index()
rows=[]
for t,x in O.items():
    if len(x)<120: continue
    reg,lev,_,_=run_state_machine(x.assign(datetime=x.index)); reg=np.asarray(reg); lb=x.low.rolling(20).min().rolling(20).max().values; c=x.close.values
    zt=Z[Z.tk==t].set_index('d')[['z_pct','z0_pct','lm_pct']].sort_index()
    for i in range(60,len(x)-64):
        if reg[i]==1 and reg[i-1]!=1 and x.index[i]>=pd.Timestamp('2018-02-01'):
            d=x.index[i]; j=zt.index.searchsorted(d,side='right')-1
            if j<0: continue
            r=dict(tk=t,d=d,over=bool(c[i]>lb[i]),**zt.iloc[j].to_dict())
            for h in H: r[f'r{h}']=(c[i+h]/c[i]-1)-EW[h].iloc[i]
            rows.append(r)
E_=pd.DataFrame(rows); E_.to_pickle(SC+'flipval_events.pkl'); print('flip events',len(E_),flush=True)
def tt(s): return s.mean()/(s.std()/len(s)**.5) if len(s)>5 else np.nan
for sub,nm in ((E_,'ALL daily vector flips'),(E_[E_.over],'A: flips already over daily pivot')):
    for h in H:
        print(f'\n=== {nm} | fwd {h}d excess vs universe | n={len(sub)} ===')
        for era,(a,b) in {'2018-21':('2018-01-01','2022-01-01'),'2022+':('2022-01-01','2027-01-01')}.items():
            s=sub[(sub.d>=a)&(sub.d<b)]; print(f'  {era} all flips: mean {s[f"r{h}"].mean()*100:+.2f}% t {tt(s[f"r{h}"]):+.1f} n={len(s)}')
            for col in ('lm_pct','z0_pct','z_pct'):
                if col=='z_pct' and era=='2018-21': continue
                g=s.dropna(subset=[col]); lo=g[g[col]<=1/3][f'r{h}']; mi=g[(g[col]>1/3)&(g[col]<2/3)][f'r{h}']; hi=g[g[col]>=2/3][f'r{h}']
                d_=lo.mean()-hi.mean(); se=np.sqrt(lo.var()/len(lo)+hi.var()/len(hi))
                print(f'    {col:7s} cheap {lo.mean()*100:+6.2f}% (n{len(lo)}) | mid {mi.mean()*100:+6.2f}% | rich {hi.mean()*100:+6.2f}% (n{len(hi)}) | cheap-rich {d_*100:+.2f}% t {d_/se:+.1f}')
