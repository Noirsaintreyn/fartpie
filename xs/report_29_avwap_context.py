"""Nov 13 2026: ANCHORED-VWAP CONTEXT for daily vector flips. Same 12k flip events as report_27 (daily vector flip to bull, S&P names). At each flip, is the close ABOVE or BELOW the stock's calendar-anchored VWAP (typical price x volume, anchor = start of week / month / quarter / year, needs >=3/5/10/20 sessions into the anchor), and is the stock's SECTOR ETF above its own monthly / quarterly VWAP. Outcome: forward 21d/63d return minus universe EW. Declared: PASS = above-vs-below spread same sign in BOTH eras and |t|>=2 in 2022+. 'Buy low' reading = flips BELOW the anchor."""
import json, pickle, warnings, numpy as np, pandas as pd, yfinance as yf; warnings.filterwarnings('ignore')
SC='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/'
E=pd.read_pickle(SC+'flipval_events.pkl'); tk=sorted(E.tk.unique())
try: V=pickle.load(open(SC+'avwap_ohlcv.pkl','rb'))
except Exception:
    V={}
    for i in range(0,len(tk),100):
        d=yf.download([t.replace('.','-') for t in tk[i:i+100]]+(['XLK','XLF','XLV','XLY','XLP','XLE','XLI','XLB','XLU','XLRE','XLC'] if i==0 else []),start='2017-01-01',interval='1d',auto_adjust=True,group_by='ticker',progress=False,threads=True)
        for t in list(d.columns.levels[0]):
            try: V[t.replace('-','.') if t not in ('XLK','XLF','XLV','XLY','XLP','XLE','XLI','XLB','XLU','XLRE','XLC') else t]=d[t].dropna(subset=['Close']).rename(columns=str.lower)[['high','low','close','volume']]
            except Exception: pass
        print('px',i,len(V),flush=True)
    pickle.dump(V,open(SC+'avwap_ohlcv.pkl','wb'))
sec=json.load(open('/home/claude/fartpie/sp500_sectors.json')); print('sector map sample',list(sec.items())[:3] if isinstance(sec,dict) else sec[:3])
MAP={'Technology':'XLK','Information Technology':'XLK','Financial Services':'XLF','Financials':'XLF','Healthcare':'XLV','Health Care':'XLV','Consumer Cyclical':'XLY','Consumer Discretionary':'XLY','Consumer Defensive':'XLP','Consumer Staples':'XLP','Energy':'XLE','Industrials':'XLI','Basic Materials':'XLB','Materials':'XLB','Utilities':'XLU','Real Estate':'XLRE','Communication Services':'XLC'}
def avwap(df,unit,minn):
    idx=df.index; g={'W':idx.to_period('W'),'M':idx.to_period('M'),'Q':idx.to_period('Q'),'Y':idx.to_period('Y')}[unit]
    tp=(df.high+df.low+df.close)/3; pv=(tp*df.volume).groupby(g).cumsum(); cv=df.volume.groupby(g).cumsum(); n=df.close.groupby(g).cumcount()+1
    w=pv/cv; w[n<minn]=np.nan; return w
A={}  # per ticker: dist to each anchor
for t,df in V.items():
    df=df[df.volume>0]
    if len(df)<250: continue
    A[t]={u:(df.close/avwap(df,u,m)-1) for u,m in (('W',3),('M',5),('Q',10),('Y',20))}
ab={u:[] for u in 'WMQY'}; sab={'M':[],'Q':[]}
for r in E.itertuples():
    a=A.get(r.tk)
    for u in 'WMQY': ab[u].append(a[u].get(r.d,np.nan) if a is not None else np.nan)
    s=MAP.get(sec.get(r.tk) if isinstance(sec,dict) else None); 
    for u in 'MQ': sab[u].append(A[s][u].get(r.d,np.nan) if s in A else np.nan)
for u in 'WMQY': E['d_'+u]=ab[u]
for u in 'MQ': E['sec_'+u]=sab[u]
E.to_pickle(SC+'avwap_events.pkl')
def tt(a,b):
    se=np.sqrt(a.var()/len(a)+b.var()/len(b)); return (a.mean()-b.mean())/se
print('events',len(E),'| sector-ETF coverage',E.sec_M.notna().mean().round(2))
for h in (21,63):
    print(f'\n=== forward {h}d excess return, daily vector flips; ABOVE vs BELOW anchored VWAP ===')
    for col,nm in (('d_W','stock vs WEEKLY VWAP'),('d_M','stock vs MONTHLY VWAP'),('d_Q','stock vs QUARTERLY VWAP'),('d_Y','stock vs ANNUAL VWAP'),('sec_M','SECTOR ETF vs its monthly VWAP'),('sec_Q','SECTOR ETF vs its quarterly VWAP')):
        line=f'{nm:34s}'
        for era,(a,b) in {'2018-21':('2018-01-01','2022-01-01'),'2022+':('2022-01-01','2027-01-01')}.items():
            s=E[(E.d>=a)&(E.d<b)].dropna(subset=[col]); up=s[s[col]>0][f'r{h}']; dn=s[s[col]<=0][f'r{h}']
            line+=f' | {era}: above {up.mean()*100:+.2f}% (n{len(up)}) below {dn.mean()*100:+.2f}% (n{len(dn)}) up-dn {(up.mean()-dn.mean())*100:+.2f}% t {tt(up,dn):+.1f}'
        print(line)
print('\n=== deep-below / far-above terciles of stock-vs-anchor distance (2022+), fwd 63d ===')
for col in ('d_M','d_Q','d_Y'):
    for era,(a,b) in {'2018-21':('2018-01-01','2022-01-01'),'2022+':('2022-01-01','2027-01-01')}.items():
        s=E[(E.d>=a)&(E.d<b)].dropna(subset=[col]); q=s.groupby('d')[col].rank(pct=True); lo=s[q<=1/3].r63; hi=s[q>=2/3].r63
        print(f'  {col} {era}: lowest third (relative low) {lo.mean()*100:+.2f}% | highest third {hi.mean()*100:+.2f}% | low-high {(lo.mean()-hi.mean())*100:+.2f}% t {tt(lo,hi):+.1f}')
