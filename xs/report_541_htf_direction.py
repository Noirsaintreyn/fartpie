"""Report 541 (Nov 28 2026): Rey: high vol goes either way - do we just take longs, or should a higher timeframe vector decide bull/bear? DECLARED BEFORE RESULTS.
Mechanism: a daily up-flip inside a weekly vector uptrend has the larger frame behind it (long); inside a weekly downtrend it is a counter-trend bounce that fails at the weekly level. Weekly vector = same run_state_machine on W-FRI bars, regime read from the LAST COMPLETED week (no lookahead). 
Test A: sub-$5 daily up-flips (1,504 in flip_history) split by weekly regime up/down, within HI-ATRP (top third, train cut) and all; reach (PEAK>=10%), mean/median r; train 2019-21 | test 2022+; perm null vs random same-size draws.
Test B (the short side): all sub-$5 daily DOWN flips, HI-ATRP, weekly regime down vs up; short return to the next up-flip (-(c_exit/c_entry-1), no borrow cost) and max further drop; same split. Pass: weekly-agreeing arm beats disagreeing arm both eras, same sign, perm p<.05.
Lower timeframe is NOT tested as a decider: 1H/4H vector flips happen after the daily flip bar and only on 124 cached tickers (report 531 showed they add nothing as entries).
"""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0,'xs')
from vector_core import run_state_machine
H=pd.read_csv('flip_history.csv',parse_dates=['d']); H=H[H.px<5]; D=pd.read_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/wide_daily_ohlcv.pkl'); UP=[]; DN=[]
for tk in H.tk.unique():
    q=D.get(tk)
    if q is None: continue
    q=q.dropna(subset=['close']).copy(); q.index=pd.DatetimeIndex(q.index).astype('datetime64[ns]'); q=q[q.index>='2017-06-01']
    if len(q)<200: continue
    W=q.resample('W-FRI').agg(dict(open='first',high='max',low='min',close='last',volume='sum')).dropna()
    if len(W)<80: continue
    rw=np.asarray(run_state_machine(W)[0]); wr=pd.Series(rw,index=W.index)
    rd,lv,atr,fl=run_state_machine(q); c=q.close.values; h=q.high.values; l=q.low.values
    for k,(i,dr) in enumerate(fl):
        d=q.index[i]
        if not(c[i]<5) or d<pd.Timestamp('2019-01-01'): continue
        prev=wr[wr.index<d-pd.Timedelta(days=d.weekday()+0)]  # weeks ended strictly before this week's Friday
        prev=wr[wr.index<=d-pd.Timedelta(days=(d.weekday()-4)%7 if d.weekday()<4 else 0)] if False else wr[wr.index<(d-pd.Timedelta(days=d.weekday()))]
        if not len(prev): continue
        j=next((jj for jj,d2 in fl[k+1:] if d2==-dr),None)
        if j is None: continue
        row=dict(tk=tk,d=d,ATRP=atr[i]/c[i],wk=int(prev.iloc[-1]),yr=d.year)
        if dr==1: row.update(r=c[j]/c[i]-1,hit=float(h[i+1:j+1].max()/c[i]-1>=.10) if j>i else 0.); UP.append(row)
        else: row.update(r=-(c[j]/c[i]-1),hit=float(1-l[i+1:j+1].min()/c[i]>=.10) if j>i else 0.); DN.append(row)
rng=np.random.default_rng(2)
def rep(G,title,sign):
    G=pd.DataFrame(G); tr=G.yr.between(2019,2021); te=G.yr>=2022; cut=G.loc[tr,'ATRP'].quantile(2/3); G['HA']=G.ATRP>cut
    print(f'\n{title}: n{len(G)} train{tr.sum()} test{te.sum()} HI-ATRP cut {cut:.3f}; weekly regime values {sorted(G.wk.unique())}')
    for nm,mk in (('TRAIN',tr),('TEST',te)):
        for ha in (True,False):
            base=G[mk&(G.HA==ha)]; out=f'{nm} {"HI-ATRP" if ha else "ALL    "}'
            for w,lab in ((sign,'agree'),(-sign,'disagr')):
                d=base[base.wk==w]
                if len(d)<8: out+=f' | {lab} n{len(d)} few'; continue
                R=np.array([(lambda s:(s.hit.mean(),s.r.mean()))(base.sample(len(d),random_state=int(rng.integers(1e9)))) for _ in range(1500)])
                out+=f' | {lab} n{len(d):4d} reach {d.hit.mean()*100:4.0f}% mean {d.r.mean()*100:+6.1f} med {d.r.median()*100:+6.1f} p({(R[:,0]>=d.hit.mean()).mean():.2f},{(R[:,1]>=d.r.mean()).mean():.2f})'
            print(out)
rep(UP,'A. LONG: daily up-flips by weekly regime (agree = weekly up)',1)
rep(DN,'B. SHORT: daily down-flips by weekly regime (agree = weekly down), returns are short-side',-1)
