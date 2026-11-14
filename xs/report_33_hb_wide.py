"""Nov 14 2026: confirm H-b (vector flips only count when the index is above its 200d SMA) on the WIDE universe: 12k daily vector flips on S&P names (report_27 events), QQQ state PIT (prior close vs 200d SMA), forward 21d/63d excess return vs universe EW, by the five declared regimes. H-b predicts above-200d flips beat below-200d flips in most regimes, and flips above the 200d beat the average flip. PASS = above>below in >=4 of 5 regimes and pooled t>=2."""
import pickle, numpy as np, pandas as pd, yfinance as yf, warnings; warnings.filterwarnings('ignore')
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
E=pd.read_pickle(SC+'flipval_events.pkl')
q=yf.download('QQQ',start='2016-01-01',auto_adjust=True,progress=False)['Close'].squeeze().dropna(); up=(q>q.rolling(200).mean()).shift(1)
E['up']=up.reindex(E.d).values
RG=[('R1','2018-01-01','2020-03-01'),('R2','2020-03-01','2022-01-01'),('R3','2022-01-01','2023-04-01'),('R4','2023-04-01','2025-04-01'),('R5','2025-04-01','2027-01-01')]
E['rg']=[next((n for n,a,b in RG if pd.Timestamp(a)<=t<pd.Timestamp(b)),None) for t in E.d]
def tt(a,b): return (a.mean()-b.mean())/np.sqrt(a.var()/len(a)+b.var()/len(b))
for sub,nm in ((E,'ALL daily flips'),(E[E.over],'A: flips over daily pivot')):
    for h in (21,63):
        print(f'\n=== {nm} | fwd {h}d excess vs universe | QQQ above vs below 200d ===')
        wins=0
        for n,_,_ in RG:
            s=sub[sub.rg==n]; a=s[s.up==True][f'r{h}']; b=s[s.up==False][f'r{h}']
            if len(a)>20 and len(b)>20: wins+=a.mean()>b.mean()
            print(f'  {n}: above {a.mean()*100:+.2f}% (n{len(a)}) | below {b.mean()*100:+.2f}% (n{len(b)}) | diff {(a.mean()-b.mean())*100 if len(b) else float("nan"):+.2f}%'+(f' t {tt(a,b):+.1f}' if len(a)>20 and len(b)>20 else ''))
        a=sub[sub.up==True][f'r{h}']; b=sub[sub.up==False][f'r{h}']; print(f'  POOLED: above {a.mean()*100:+.2f}% (n{len(a)}) below {b.mean()*100:+.2f}% (n{len(b)}) diff {(a.mean()-b.mean())*100:+.2f}% t {tt(a,b):+.1f} | regimes where above>below: {wins}')
