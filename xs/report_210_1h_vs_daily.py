"""Report 210: 1H vs daily vector flips on sub-$5 names over three regimes. DECLARED before results.
Data: Yahoo 1H bars 2023-12-18..2026-11-13 (unadjusted, RTH) for 317 names of the wide universe that closed < $5 on 20+ days since Dec 2023 (survivorship: names alive today); daily bars from the wide daily file. Same state machine on both. Entry = open of the bar after the bull-flip bar, exit = open of the bar after the bear-flip bar (or last close if open). Flips with entry >= 2024-02-01 and entry price < $5. Trades containing a bar-to-bar jump > 2x or < 0.5x are dropped (splits). Cost 100bp round trip (30bp shown too).
Split fixed in advance: H1 = 2024-02-01..2025-06-30, H2 = 2025-07-01..2026-11-13.
Rey's hypothesis: 1H catches the biggest moves earlier. Tests: (A) per-flip net return distribution, 1H vs daily, each half. (B) EPISODES defined without either signal: a name's close doubles within 60 trading days from a 10-day-low close < $5 (non-overlapping). For each episode and timeframe: is there a flip entered between 5 trading days before the trough and the peak; LATENESS = share of the trough-to-peak move already gone at entry; realized net return of that trade; share of episodes where the trade made >= +50%. H: 1H lateness < daily lateness AND 1H realized >= daily realized, in both halves."""
import pickle, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from vector_pine import run_state_machine
E='/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/'; SP=E[:-4]
P=pickle.load(open(E+'lowpx_1h_3y.pkl','rb')); D=pd.read_pickle(E+'wide_daily_ohlcv.pkl'); S0=pd.Timestamp('2024-02-01'); MID=pd.Timestamp('2025-07-01')
def trades(B,tf):
    reg,_,_,_=run_state_machine(B.assign(datetime=B.index)); reg=np.asarray(reg); o,c=B.open.values,B.close.values; idx=B.index; n=len(B); out=[]; i=1
    while i<n-1:
        if reg[i]==1 and reg[i-1]!=1:
            e=i+1; j=i+1
            while j<n and reg[j]!=-1: j+=1
            op=j>=n-1; x=n-1 if op else j+1; xp=c[n-1] if op else o[x]
            g=o[e+1:x+1]/c[e:x] if x>e else np.array([1.]); disc=bool(len(g) and (g.max()>2 or g.min()<.5))
            out.append(dict(tf=tf,entry=idx[e],exit=idx[min(x,n-1)],px=o[e],xpx=xp,ret=xp/o[e]-1,peak=c[e:x+1].max()/o[e]-1,open=op,disc=disc)); i=j
        else: i+=1
    return out
rows=[]; EP=[]
for tk,h in P.items():
    x=h.copy(); x.columns=['open','high','low','close','volume']; x.index=x.index.tz_convert('America/New_York').tz_localize(None); x=x[~x.index.duplicated()].dropna(); m=x.index.hour*60+x.index.minute; x=x[(m>=570)&(m<960)]
    if len(x)<1500 or tk not in D: continue
    d=D[tk]; d=d[~d.index.duplicated()]; d=d[d.index>='2022-06-01'][['open','high','low','close','volume']].dropna()
    for t in trades(x,'1H'): rows.append(dict(tk=tk,**t))
    for t in trades(d,'D'): rows.append(dict(tk=tk,**t))
    dd=d[d.index>='2024-01-15']; c=dd.close.values; ix=dd.index; i=10
    while i<len(c)-5:
        w=c[i+1:i+61]
        if c[i]<5 and len(w) and c[i]<=c[i-10:i+1].min()+1e-12 and w.max()/c[i]>=2:
            j=i+1+int(np.argmax(w)); EP.append(dict(tk=tk,t0=ix[i],t1=ix[j],p0=c[i],p1=c[j],t0m5=ix[max(0,i-5)])); i=j+1
        else: i+=1
T=pd.DataFrame(rows); T=T[(T.entry>=S0)&(T.px<5)&(~T.disc)].copy(); T['half']=np.where(T.entry<MID,'H1','H2'); EPd=pd.DataFrame(EP); EPd=EPd[EPd.t0>=S0]; EPd['half']=np.where(EPd.t0<MID,'H1','H2')
print('names',T.tk.nunique(),'| flips',T.groupby(['tf','half']).size().to_dict(),'| doubling episodes',EPd.groupby('half').size().to_dict())
print('\n(A) per-flip, net of 100bp (30bp mean in brackets)')
for tf in ('D','1H'):
    for hf in ('H1','H2'):
        s=T[(T.tf==tf)&(T.half==hf)]; r=s.ret-.01; rs=r.sort_values(); hold=(s.exit-s.entry).dt.days.median()
        print(f'  {tf:2s} {hf} n{len(s):5d} mean {r.mean()*100:6.2f}% [{(s.ret-.003).mean()*100:6.2f}%] x3 {rs.iloc[:-3].mean()*100:6.2f}% med {r.median()*100:6.2f}% win {np.mean(r>0)*100:3.0f}% >=+30% {np.mean(r>=.3)*100:4.1f}% >=+50% {np.mean(r>=.5)*100:4.1f}% >=+100% {np.mean(r>=1)*100:4.1f}% | sum of returns {r.sum()*100:7.0f}% | median hold {hold:.0f}d')
print('\n(B) doubling episodes: capture by timeframe')
G={k:g.sort_values('entry') for k,g in T.groupby(['tk','tf'])}
for hf in ('H1','H2'):
    e=EPd[EPd.half==hf]
    for tf in ('D','1H'):
        late=[]; real=[]; hit=0
        for _,ep in e.iterrows():
            g=G.get((ep.tk,tf))
            if g is None: continue
            # daily entries carry a date; 1H a timestamp. window: from 5 trading days before the trough to the peak day end
            w=g[(g.entry>=ep.t0m5)&(g.entry<=ep.t1+pd.Timedelta(hours=23))]
            if not len(w): continue
            hit+=1; best=w.iloc[0]; late.append(np.clip((best.px/ep.p0-1)/(ep.p1/ep.p0-1),-.5,1.5)); real.append(w.ret.max()-.01)
        late=np.array(late); real=np.array(real)
        print(f'  {hf} {tf:2s}: episodes {len(e):3d}, with a flip {hit:3d} ({hit/max(len(e),1)*100:3.0f}%) | first-entry lateness median {np.median(late)*100:4.0f}% of the move gone | best trade in episode: median {np.median(real)*100:6.1f}% mean {real.mean()*100:6.1f}% share >=+50% {np.mean(real>=.5)*100:4.0f}% >=+100% {np.mean(real>=1)*100:4.0f}%')
T.to_pickle(SP+'flips_1h_vs_d210.pkl'); EPd.to_pickle(SP+'episodes210.pkl')
