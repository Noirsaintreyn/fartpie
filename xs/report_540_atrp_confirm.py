"""Report 540 (Nov 28 2026): Rey: high ATRP + high flip volume / fast flip = confirmed move (vs bigger/smaller)? DECLARED BEFORE RESULTS.
Features at flip day close: ATRP (given), VOLR = flip-day volume / mean of prior 20 days, S_BODY = |C-O|/ATR14, S_RANGE = (H-L)/ATR14. Cuts from TRAIN 2019-21 (top third = HI). Arms inside HI-ATRP: +VOL, +FAST(body), +BOTH(VOL&FAST), +EITHER; compared to HI-ATRP alone and to LO-ATRP. Metrics: reach (PEAK>=+10%), mean r, median r, by era (train 2019-21 | test 2022+). Null: random same-size draws from HI-ATRP in the same era (10k), p = share of draws with reach/mean >= arm. Mechanism: a volatile stock breaking up on heavy volume / fast displacement = participation commits the move. Pass: beats HI-ATRP alone in both eras, perm p<.05 both. 5 arms x 2 metrics = 10 trials, Bonferroni .005.
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
H=pd.read_csv('flip_history.csv',parse_dates=['d']); H=H[H.px<5].copy(); D=pd.read_pickle('/tmp/claude-0/-home-claude/6d3c83e1-1a4a-5a0a-8647-c2590d1d30a4/scratchpad/ext/wide_daily_ohlcv.pkl'); rows=[]
for tk,g in H.groupby('tk'):
    q=D.get(tk)
    if q is None: continue
    q=q.dropna(subset=['close']).copy(); q.index=pd.DatetimeIndex(q.index).astype('datetime64[ns]'); pc=q.close.shift(); a=pd.concat([q.high-q.low,(q.high-pc).abs(),(q.low-pc).abs()],axis=1).max(axis=1).rolling(14).mean()
    f=pd.DataFrame(dict(VOLR=q.volume/q.volume.shift().rolling(20).mean(),S_BODY=(q.close-q.open).abs()/a,S_RANGE=(q.high-q.low)/a)); rows.append(g.merge(f,left_on='d',right_index=True))
G=pd.concat(rows).dropna(subset=['VOLR','S_BODY']); G['yr']=G.d.dt.year; G['hit']=G.hit.astype(float); tr=G.yr.between(2019,2021); te=G.yr>=2022
c={k:G.loc[tr,k].quantile(2/3) for k in ('ATRP','VOLR','S_BODY','S_RANGE')}; lo=G.loc[tr,'ATRP'].quantile(1/3)
G['HA']=G.ATRP>c['ATRP']; G['V']=G.VOLR>c['VOLR']; G['F']=G.S_BODY>c['S_BODY']; G['FR']=G.S_RANGE>c['S_RANGE']
arms={'LO-ATRP':G.ATRP<=lo,'HI-ATRP':G.HA,'HI&VOL':G.HA&G.V,'HI&FAST':G.HA&G.F,'HI&FASTR':G.HA&G.FR,'HI&VOL&FAST':G.HA&G.V&G.F,'HI&(VOL|FAST)':G.HA&(G.V|G.F)}
rng=np.random.default_rng(1)
print('n',len(G),'train',tr.sum(),'test',te.sum(),'cuts',{k:round(v,3) for k,v in c.items()})
for nm,mk in (('TRAIN 2019-21',tr),('TEST 2022+',te)):
    print('\n'+nm); base=G[mk&G.HA]
    for a,m in arms.items():
        d=G[mk&m]
        if len(d)<8: print(f'{a:14} n{len(d)} few'); continue
        p=''
        if a not in('LO-ATRP','HI-ATRP'):
            R=[(base.sample(len(d),random_state=int(rng.integers(1e9))).hit.mean(),base.sample(len(d),random_state=int(rng.integers(1e9))).r.mean()) for _ in range(2000)]; R=np.array(R); p=f'perm p reach {(R[:,0]>=d.hit.mean()).mean():.3f} mean {(R[:,1]>=d.r.mean()).mean():.3f}'
        print(f'{a:14} n{len(d):4d} reach {d.hit.mean()*100:5.1f}% mean r {d.r.mean()*100:+6.1f} med r {d.r.median()*100:+6.1f}  {p}')
